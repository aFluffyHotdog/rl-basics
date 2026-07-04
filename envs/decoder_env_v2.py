from pathlib import Path
from typing import Optional
from collections import deque

import numpy as np
import gymnasium as gym

NUM_DECODERS = 8

class SubDecoder():
    def __init__(self):
        # NOTE: 1 chunk = 16 bit
        # 4 chunk LIT FIFO buffers
        self.LIT_FIFO = deque(maxlen=4)
        # 16 chunk CMD FIFO buffer
        self.CMD_FIFO = deque(maxlen=16)
        # keep track of cycles left to process curr CMD
        self.cycles_left = 0
        self.current_cmd = 0

    def _get_cmd_type(self, token: int) -> str:
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
        if type_code == 0:
            return "CMD"
        if type_code == 1:
            return "LIT"
        if type_code == 2:
            return "RLE"
        
        raise TypeError("unknown command type")
    
    def is_free(self, cmd: int, required_lit_chunks: int = 0):
        """
        Helper function to evaluate if the decoder can accept a new command given the command `cmd`
        Returns `True` if the subdecoder can accept more work, `False` otherwise
        """
        cmd_type = self._get_cmd_type(cmd)

        if cmd_type == "CMD":
            # The CMD FIFO strictly holds 16 commands
            return len(self.CMD_FIFO) < 16
                
        elif cmd_type == "LIT":
            # You must check if your LIT FIFO banks have enough combined/bank-specific
            # capacity to hold the new required_lit_chunks. 
            # (Assuming self.get_lit_capacity() calculates space based on the 24/24/18/12 limits)
            return len(self.LIT_FIFO) >= required_lit_chunks
            
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
        cmd_type = self._get_cmd_type(cmd)
        if cmd_type != "CMD":
            raise ValueError("calculate_cycles expects a CMD token")

        payload = int(cmd) & 0xFFFF
        is_rle = (payload >> 15) & 0x1

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


        """
        req_lit_chunks = 0
        cmd_type = self._get_cmd_type(cmd)

        #TODO: add logic to read req_lit_chunks from the cmd 

        if self.is_free(cmd_type, req_lit_chunks) == True:
            self.CMD_FIFO.append()
            # if LIT cmd, append the Literal into lit buffer
            if cmd_type == "LIT":
                self.LIT_FIFO.append()
            # append the RLE payload into the FIFO command
            if cmd_type == "RLE":
                self.CMD_FIFO.append()

            # calculate the cycles this command will take and update accordingly
            self.cycles_left += self.calculate_cycles(cmd)

        pass
    
    def step(self):
        """
        method to decrement the cycles left of current command to process and move on to the next command
        """
        # decrement the cycles remaining on current task
        self.cycles_left -= 1
        if self.cycles_left == 0:
            # pop new commmand
            new_cmd = self.CMD_FIFO.pop()
            cmd_type = self._get_cmd_type(new_cmd)
            # if it's RLE, also pop the seed pattern
            if cmd_type == "RLE":
                RLE_seed = self.CMD_FIFO.pop()
            # if LC, pop from LIT fifo
            elif cmd_type == "LC":
                self.LIT_FIFO.pop()


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

    def _load_hex_folder(self, cmd_path: str) -> list[np.ndarray]:
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
            lines = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
            if not lines:
                raise ValueError(f"Hex file is empty: {file_path}")

            outputs.append(np.asarray(lines, dtype=int))

        return outputs

    def potential():
        # sum up cycles taken by remaining work
        # take the max of -(sub-d that takes the longest, sum / num_decoders)
        pass

    def step(self, action):
        observation = None
        reward = 0
        terminated = False
        truncated = None
        info = None

        # action: choose a decoder to send
        target_sub_d = action
        # calculate current potential

        # check if sub decoder is free
        if self.decoders[target_sub_d].is_free() == True:
            # TODO: add the correct parameters / bit numbers

            # consume the 135 bits from the selected cmd array
            self.cmds[target_sub_d].pop()
            # update the sub-d accordingly
            self.decoders[target_sub_d].assign_work()
        else:
            # give a reward score that is a punishment or something
            reward = -100

        # update every subdecoder accordingly
        for sub_d in self.decoders:
            sub_d.step()

        # calculate the reward
        # new_potential - prev_potential

        return observation, reward, terminated, truncated, info

    def _get_obs():
        # List of observations:
        # - state of each sub-d's FIFOs + outputs
        # - each sub-d's next 4 cmds
        # - cycles remaining for each sub-d's current assigned work
        pass