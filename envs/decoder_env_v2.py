"""
New version of a python environment that simulates the T8 decompressor

now implements:
    - the TDM mode (135 bits gets sent to a subdecoder at a time)
    - actual stalling if a subdecoder is full
    - FIFO buffer behavior
    - cycle modelling based on CMD type and size
    - CMD decoding (partially)

still missing:
    - row barrier mechanism
"""
from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np
import gymnasium as gym

BEATS_ON_BUS = 8
NUM_DECODERS = 8
CMD_TYPE=0
LIT_TYPE=1
RLE_TYPE=2
MAX_STALL = 200 # for deadlock detection

class SubDecoder():
    def __init__(self):
        # NOTE: 1 chunk = 16 bit
        # 24 * 4 chunk LIT FIFO buffers
        self.LIT_FIFO = deque(maxlen=24)
        # 16 chunk CMD FIFO buffer
        self.CMD_FIFO = deque(maxlen=16)
        # keep track of cycles left to process curr CMD
        self.cycles_left = 0
        self.current_cmd = 0
        # accumulated active cycles (time spent executing commands)
        self.active_cycles = 0
        self.cmd_credit = 0

    def debug_dump_buffers(self):
        """Return a short debug string describing the current FIFO contents.

        Shows the type of each token in the CMD FIFO and a summary of the LIT FIFO.
        """
        cmd_types = []
        for tok in self.CMD_FIFO:
            try:
                t = self.get_cmd_type(tok)
            except Exception:
                t = "?"
            cmd_types.append(t)

        # LIT_FIFO stores 16-bit payloads; report count and sample values
        lit_preview = list(self.LIT_FIFO)[:8]
        return (
            f"CMD_FIFO_types={cmd_types}, LIT_count={len(self.LIT_FIFO)}, "
            f"LIT_preview={lit_preview}, cycles_left={self.cycles_left}, cmd_credit={self.cmd_credit}"
        )

    def get_cmd_type(self, token: int) -> str:
        """Return the token type for a token encoded as an int.

        The trial compressor emits tokens as 5-hex-digit words `T PPPP` where
        `T` (4 bits) is the type: 0=CMD, 1=LIT, 2=RLE and `PPPP` is the 16-bit
        payload. This helper accepts that integer form (e.g. 0x107FE) and
        returns the string `'CMD'`, `'LIT'`, `'RLE'`, or `'UNKNOWN'`.
        """
        try:
            t = int(token)
        except Exception:
            raise TypeError("token must be an int-like value")

        # Expect a 20-bit word (T:4 bits, payload:16 bits). If only a 16-bit
        # value is provided, the caller didn't include the type nibble.
        if t <= 0xFFFF:
            return "UNKNOWN"

        type_code = (t >> 16) & 0xF
        if type_code == CMD_TYPE:
            return "CMD"
        if type_code == LIT_TYPE:
            return "LIT"
        if type_code == RLE_TYPE:
            return "RLE"
        
        return "UNKNOWN"
    
    def is_free(self, cmd: int, required_lit_chunks: int = 1):
        """
        Helper function to evaluate if the decoder can accept a new command given the command `cmd`
        Returns `True` if the subdecoder can accept more work, `False` otherwise
        """
        cmd_type = self.get_cmd_type(cmd)

        if cmd_type in ["CMD", "RLE", "UNKNOWN"]:
            # The CMD FIFO strictly holds 16 commands
            return self.cmd_credit < 8
                
        elif cmd_type == "LIT":
            # You must check if your LIT FIFO banks have enough combined/bank-specific
            # capacity to hold the new required_lit_chunks. 
            # (Assuming self.get_lit_capacity() calculates space based on the 24/24/18/12 limits)
            available_space = self.LIT_FIFO.maxlen - len(self.LIT_FIFO)
            return available_space >= required_lit_chunks
            
        return False
    
    def calculate_cycles(self, cmd: int):
        """
        given a token, calculate how many cycles it would take to execute based on:

        Cmd layout (16 b):
        [15]    : type (0=LC, 1=RLE)
        LC  : [14:11] lit_field (lit_bytes/2, max 15)
                [10: 7] copy_len  (0..14 = 2..16 byte copy ; 15 = no copy)
                [ 6: 5] buf_sel   (0=current, 1..3 = last/last_last/last_last_last)
                [ 4: 1] offset    (byte offset in selected history buf)
                [   0] reserved
        RLE : [14:11] rle_length (length-1, so max 16 bytes)
                [10: 8] period_class (7 = FAST mode, 16b pattern as-is)
                [ 7: 0] reserved

        LC:  IF lit_field >= 4: (lit_field / 4) + (lit_field % 4); ELSE 1
        RLE: IF rle_len > 7: 2; ELSE: 1
        """
        cmd_type = self.get_cmd_type(cmd)

        payload = int(cmd) & 0xFFFF
        is_rle = (cmd_type == "RLE") or (payload >> 15) & 0x1

        if is_rle:
            rle_length = (payload >> 11) & 0xF
            return 2 if rle_length > 7 else 1

        lit_field = (payload >> 11) & 0xF
        if lit_field >= 4:
            return (lit_field // 4) + (1 if (lit_field % 4) else 0)

        return 1
    
    def assign_work(self, cmd: int):
        """
        helper to update the status of the sub-d once work has been assigned

        ASSUMES that the decoder is free (needs a call to is_free before hand)
        other functions should stall the decoder before this gets called
        """
        req_lit_chunks = 0
        cmd_type = self.get_cmd_type(cmd)

        # determine required literal chunks for CMD tokens
        if cmd_type == "CMD":
            payload = int(cmd) & 0xFFFF
            lit_field = (payload >> 11) & 0xF
            req_lit_chunks = int(lit_field)
        elif cmd_type == "LIT":
            req_lit_chunks = 1

        # check availability by passing the original token and required chunks
        if not self.is_free(cmd, req_lit_chunks):
            return 

        # enqueue the actual token/payload into the appropriate FIFO
        if cmd_type == "CMD":
            self.CMD_FIFO.append(cmd)
            self.cmd_credit += 1
        elif cmd_type == "LIT":
            # store only the 16-bit payload for literals
            self.LIT_FIFO.append(int(cmd) & 0xFFFF)
        elif cmd_type in ["RLE", "UNKNOWN"]:
            # RLE seed patterns are stored in the CMD FIFO after their RLE-CMD
            self.CMD_FIFO.append(cmd)
            self.cmd_credit += 1

    
    def step(self):
        """
        method to decrement the cycles left of current command to process and move on to the next command
        """
        # decrement the cycles remaining on current task
        if self.cycles_left > 0:
            self.cycles_left -= 1
            self.active_cycles +=1 # Only pull a new command if we aren't currently busy AND the FIFO isn't empty
            if self.cycles_left == 0:
                self.cmd_credit -= 1

        if self.cycles_left == 0 and self.CMD_FIFO:
            # PEEK at the next command instead of popping it immediately
            next_cmd = self.CMD_FIFO[0]
            cmd_type = self.get_cmd_type(next_cmd)
            payload = int(next_cmd) & 0xFFFF
            
            # --- DEPENDENCY STALL CHECKS ---
            
            # 1. RLE Seed Check
            is_rle = (cmd_type == "RLE")
            period_class = (payload >> 8) & 0x7 if is_rle else 0
            requires_seed = is_rle and (period_class == 7)
            
            if requires_seed and len(self.CMD_FIFO) < 2:
                # Stall: Waiting for the bus to deliver the seed
                return
                
            # 2. CMD Literal Check (THE NEW FIX)
            if cmd_type == "CMD":
                req_lit_chunks = (payload >> 11) & 0xF
                if len(self.LIT_FIFO) < req_lit_chunks:
                    # Stall: Waiting for the bus to deliver all required literals
                    return 
            
            # --- SAFE EXECUTION ---
            
            # Now that all data dependencies are met, it is safe to pop and execute
            new_cmd = self.CMD_FIFO.popleft()
            
            if cmd_type == "RLE":
                if requires_seed:
                    RLE_seed = self.CMD_FIFO.popleft()
                
            elif cmd_type == "CMD":
                # We already verified len >= req_lit_chunks, so this is guaranteed to pop everything
                print("req lit chunks: ", req_lit_chunks)
                for _ in range(req_lit_chunks):
                    self.LIT_FIFO.popleft()

            # Calculate the cycles this command will take and update accordingly
            self.cycles_left += self.calculate_cycles(new_cmd)
            self.active_cycles += 1


class DecoderEnvV2(gym.Env):
    def __init__(self, cmd_path: str):
        """
        Initializes the environment
            -`cmd_path`: path to the `beats_hex` folder for example: `sample_data/test_11_lantern/beats_hex`
        """
        # bit arrays of commands (RLE, LC, ROW_REPEAT)
        # one array per subdecoder, each shape = (num_lines, bits_per_line)
        self.cmds = self._load_hex_folder(cmd_path)
        # array of subdecoder objects
        self.decoders = []
        for i in range(NUM_DECODERS):
            self.decoders.append(SubDecoder())
        # num cycles taken by the simulation
        self.num_cycles = 0
        # our action space is the number of decoders we have
        self.action_space = gym.spaces.Discrete(NUM_DECODERS)

        # TODO: design the observation space ()
        # generous version
        self.observation_space = {
            # Dictionary of FIFO statuses by sub-d
            # List of cycles left per sub-d
            # lookahead window for each subdecoder (8 CMDs)
                # cycle counts (approx)
                # cmd types
        }

        # limited version
        self.observation_space = {
            # cycles taken (LMFAOOOOOO)
        }

    def _load_hex_folder(self, cmd_path: str) -> list:
        """
        Load all subdecoder hex files from a folder into 8 separate hex arrays.
        """

        path = Path(cmd_path)
        if not path.is_dir():
            raise ValueError(f"cmd_path must be a directory: {cmd_path}")

        files = sorted(path.glob('*.hex'))
        if len(files) != NUM_DECODERS:
            raise ValueError(
                f"Expected {NUM_DECODERS} subdecoder hex files in {cmd_path}, found {len(files)}"
            )

        outputs = []
        for file_path in files:
            raw_lines = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
            if not raw_lines:
                raise ValueError(f"Hex file is empty: {file_path}")

            parsed = deque()
            for ln in raw_lines:
                # try parsing as hex (files contain hex tokens), fallback to int()
                try:
                    val = int(ln, 16)
                except Exception:
                    try:
                        val = int(ln, 0)
                    except Exception:
                        raise ValueError(f"Unrecognized token in {file_path}: {ln}")
                parsed.append(val)

            outputs.append(parsed)

        return outputs

    def potential():
        # sum up cycles taken by remaining work
        # take the max of -(sub-d that takes the longest, sum / num_decoders)
        pass
    
    def step_sub_decoders(self):
        """
        helper to iterate and step through all the sub decoders
        """
        for sub_d in self.decoders:
            sub_d.step()

    def step(self, action):
        observation = None
        reward = 0
        terminated = False
        truncated = None
        info = None
        stall_cycles = 0

        # action: choose a decoder to send
        target_sub_d = action

        # each call to step represents one bus beat (cycle)
        self.num_cycles += 1

        # calculate current potential

        # consume the 135 bits from the selected cmd array if not empty
        if self.cmds[target_sub_d]:
            # unload 8 CMDs onto the subdecoder or stall until it can take more
            i = 0
            while i < BEATS_ON_BUS and self.cmds[target_sub_d]:
                # check if sub decoder is free
                if self.decoders[target_sub_d].is_free(self.cmds[target_sub_d][0]) == True:
                    new_cmd = self.cmds[target_sub_d].popleft()
                    # update the sub-d accordingly
                    self.decoders[target_sub_d].assign_work(new_cmd)
                    # only increment the counter if we can push a cmd on
                    i = i + 1
                # if not, stall until decoder becomes free
                else: 
                    while self.decoders[target_sub_d].is_free(self.cmds[target_sub_d][0]) == False: 
                        stall_cycles += 1
                        self.num_cycles += 1
                        self.step_sub_decoders()
                        # NOTE: for debugging deadlock
                        cmd_type = self.decoders[0].get_cmd_type(self.cmds[target_sub_d][0])
                        if stall_cycles > MAX_STALL:
                            # dump buffers for the target subdecoder to aid debugging
                            try:
                                dump = self.decoders[target_sub_d].debug_dump_buffers()
                                print(f"[DEADLOCK DEBUG] Subdecoder {target_sub_d}: {dump}")
                            except Exception:
                                print(f"[DEADLOCK DEBUG] Subdecoder {target_sub_d}: <failed to dump buffers>")
                            raise ValueError(f"Subdecoder {target_sub_d} permanently deadlocked. cmd type: {cmd_type}")

        # update every subdecoder accordingly
        self.step_sub_decoders()

        # check for termination condition: all per-decoder queues empty and all decoders have finished
        if all(len(q) == 0 for q in self.cmds) and all(sub_d.cycles_left == 0 for sub_d in self.decoders):
            terminated = True

        # calculate the reward
        # new_potential - prev_potential

        return observation, reward, terminated, truncated, info

    def _get_obs():
        # List of observations:
        # - state of each sub-d's FIFOs + outputs
        # - each sub-d's next 4 cmds
        # - cycles remaining for each sub-d's current assigned work
        pass