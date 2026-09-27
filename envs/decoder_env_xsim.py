"""
Hardware-in-the-Loop Environment for the T8 Decompressor.
Uses Tcl-Ping-Pong to step a live Vivado SystemVerilog simulation.
Includes a fully self-contained Python Shadow Model for Exact Adaptive Packing and Skew Bound Tracking.
"""
import os
import time
import math
import subprocess
from pathlib import Path
from collections import deque
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

VIVADO_VERBOSE = True

# Hardware Constants
NUM_DECODERS = 9      # 0-7 are LC/RLE, 8 is Bypass
NO_OP_ACTION = 9      # Action index to yield the bus (Valid=0)
BEATS_ON_BUS = 8      # Max 16-bit slots per beat
MAX_CYCLES_LEFT = 16 
LOOKAHEAD_WINDOW = 64

# FIFO Max Capacities for Normalization
CMD_FIFO_SIZE = 16
LIT_FIFO_MAX = 24     # Max of the 24, 24, 18, 12 banks
OUT_FIFO_SIZE = 4
BYPASS_IN_SIZE = 8
FIFO_HOARD_THRESH = 0.8 # for reward calculation

# Sync Barrier / Skew Constraints
SKEW_BOUND = 4        # Typically matches ROW_QUEUE_DEPTH

CMD_TYPE=0
LIT_TYPE=1
RLE_TYPE=2

@dataclass
class TokenInfo:
    raw_val: int
    category: str        
    req_lit_chunks: int  
    slots_needed: int
    out_bytes: int

def decode_token_slots(token: int) -> TokenInfo:
    """Decodes a token purely to calculate its bus footprint and output byte yield."""
    t = int(token)
    payload = t & 0xFFFF
    type_code = (t >> 16) & 0xF
    
    if type_code == LIT_TYPE:
        return TokenInfo(t, "LIT", 1, 1, 0) # Literals don't independently yield bytes, the CMD does
        
    if type_code not in [CMD_TYPE, LIT_TYPE, RLE_TYPE]:
        return TokenInfo(t, "SEED", 0, 1, 0)

    is_rle = (type_code == RLE_TYPE) or (type_code == CMD_TYPE and ((payload >> 15) & 0x1 == 1))

    if is_rle:
        rle_length = (payload >> 11) & 0xF
        out_bytes = rle_length + 1
        return TokenInfo(t, "RLE", 0, 2, out_bytes) # RLE consumes 2 slots (CMD + Pattern)
    else: 
        lit_field = (payload >> 11) & 0xF
        copy_len = (payload >> 7) & 0xF
        is_row_repeat = (lit_field == 0 and copy_len == 15)
        
        if is_row_repeat:
            # ROW_REPEAT Sentinel yields 0 bytes mathematically here but advances the row internally
            out_bytes = 0 
        else:
            lit_bytes = lit_field * 2
            copy_bytes = 0 if copy_len == 15 else copy_len + 2
            out_bytes = lit_bytes + copy_bytes
            
        return TokenInfo(t, "CMD", int(lit_field), 1 + int(lit_field), out_bytes)

class DecoderEnvXSim(gym.Env):
    def __init__(self, data_dir: str, sim_dir: str, is_inference: bool = False):
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.sim_dir = Path(sim_dir).resolve()
        self.run_dir = self.sim_dir / "xsim_run" / "run"
        self.action_file = self.run_dir / "action.txt"
        
        # Shadow Model Configurations
        self.CMD_MAX = CMD_FIFO_SIZE
        self.LIT_MAX = [24, 24, 18, 12]  # Banks 1 to 4
        self.ROW_Q_MAX = OUT_FIFO_SIZE
        self.BYPASS_Q_MAX = BYPASS_IN_SIZE
        self._init_shadow_model()

        # Load data
        if is_inference:
            cmds = self._load_hex_folder(self.data_dir)
            first_file = next(self.data_dir.glob('*.hex')).name
            stem_name = first_file.split('_subdec')[0] 
            stem_path = self.data_dir / stem_name
            self.dataset_pool = [(stem_path, cmds)]
            print(f"Loaded single dataset for inference from: {self.data_dir}")
        else:
            all_datasets = []
            for test_folder in sorted(self.data_dir.iterdir()):
                if test_folder.is_dir():
                    hex_dir = test_folder / "beats_hex"
                    if hex_dir.exists() and hex_dir.is_dir():
                        try:
                            cmds = self._load_hex_folder(hex_dir)
                            first_file = next(hex_dir.glob('*.hex')).name
                            stem_name = first_file.split('_subdec')[0]
                            stem_path = hex_dir / stem_name
                            all_datasets.append((stem_path, cmds))
                        except Exception as e:
                            print(f"Skipping {test_folder.name} due to error: {e}")

            self.dataset_pool = all_datasets
            if not all_datasets:
                raise ValueError(f"No valid datasets found in {data_dir}")
        
        # Action Space: 0-7 (Regular), 8 (Bypass), 9 (Yield Bus / NO_OP)
        self.action_space = gym.spaces.Discrete(10)
        
        # Observation Space combining Shadow Model State and Python Lookahead
        self.observation_space = gym.spaces.Dict({
            "top_in_ready": gym.spaces.Discrete(2),
            "top_row_valid": gym.spaces.Discrete(2),
            "cmd_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            "lit_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            "output_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            "bypass_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "lookahead_types": gym.spaces.Box(low=-1, high=3, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.int32),
            "global_progress": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "cycles_left": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
            "lookahead_cycles": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.float32),
            "lookahead_is_barrier": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.int32),
            # NEW OBSERVATION: How far ahead each sub-decoder's dispatch is relative to the barrier
            "channel_skew": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
        })

        self.sim_proc = None
        self.hw_state = {"top_in_ready": 1, "top_row_valid": 0, "mismatches": 0}
        self.completed_images = 0
        self.episodes_since_vivado_restart = 0
        self.max_episodes_per_vivado_process = 100

    def _restart_sim_if_needed(self):
        """Restart the live Vivado simulator once every N episodes to avoid drift/lockups."""
        self.episodes_since_vivado_restart += 1
        if self.episodes_since_vivado_restart < self.max_episodes_per_vivado_process:
            return False

        self.close()
        self._start_sim()
        self.episodes_since_vivado_restart = 0
        return True

    def _init_shadow_model(self):
        """Initializes the Python-side FIFO and execution tracking logic."""
        self.cmd_fifo_cnt = [0] * 8
        self.lit_fifo_cnt = [[0] * 4 for _ in range(8)]
        self.row_queue_cnt = [0] * 8
        self.bypass_row_cnt = 0
        
        self.active_cmd_cycles = [0] * 8
        self.byte_accumulators = [0] * 8
        self.total_rows_emitted = [0] * NUM_DECODERS 
        
        # --- SKEW TRACKING (v19 sync fix) ---
        self.topr_cnt = 0  # Consumed-row count (global barrier syncs)
        self.dispatched_rows = [0] * NUM_DECODERS # Rows committed to the bus for each channel
        self.dispatched_byte_acc = [0] * 8 # Byte accumulator to calculate row boundaries upon dispatch
        
        self.internal_cmd_q = [deque() for _ in range(8)]
        self.internal_lit_q = [deque() for _ in range(8)]

    def _load_hex_folder(self, cmd_path: Path) -> list:
        files = sorted(cmd_path.glob('*.hex'))
        if len(files) != NUM_DECODERS:
            raise ValueError(f"Expected {NUM_DECODERS} subdecoder hex files, found {len(files)}")

        outputs = []
        for file_path in files:
            raw_lines = [line.strip() for line in file_path.read_text().splitlines() if line.strip()]
            parsed = deque([int(ln, 16) if ln.startswith('0x') else int(ln, 16) for ln in raw_lines])
            outputs.append(parsed)
        return outputs

    def _decode_token(self, token: int, is_bypass: bool) -> dict:
        """Parses a raw integer ONCE and returns hardware execution traits."""
        if is_bypass:
            return {"category": "CMD", "is_barrier": True, "cycles": 1}

        t = int(token)
        payload = t & 0xFFFF
        type_code = (t >> 16) & 0xF
        
        if type_code == 1: 
            return {"category": "LIT", "is_barrier": False, "cycles": 0}
        if type_code not in [0, 1, 2]: 
            return {"category": "SEED", "is_barrier": False, "cycles": 0}

        is_rle = (type_code == 2) or (type_code == 0 and ((payload >> 15) & 0x1 == 1))

        if is_rle:
            rle_length = (payload >> 11) & 0xF
            cycles = 2 if rle_length > 7 else 1
            return {"category": "RLE", "is_barrier": False, "cycles": cycles}
        else: 
            lit_field = (payload >> 11) & 0xF
            copy_len = (payload >> 7) & 0xF
            is_barrier = (lit_field == 0 and copy_len == 15)
            
            cycles = 1
            if lit_field >= 4:
                cycles = (lit_field // 4) + (1 if (lit_field % 4) else 0)
                
            return {"category": "CMD", "is_barrier": is_barrier, "cycles": cycles}

    def _start_sim(self):
        self.close()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.sim_proc = subprocess.Popen(
            ["xsim.bat", "tb_rl_interactive_snap", "-tclbatch", "quiet_run.tcl", "-R"],
            cwd=str(self.run_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

    def _write_action(self, content: str):
        with open(self.action_file, "w") as f:
            f.write(f"{content}\n")

    def _resume_sim(self):
        self.sim_proc.stdin.write("run -all\n")
        self.sim_proc.stdin.flush()

    def _wait_for_obs(self):
        found_obs = False
        while True:
            line = self.sim_proc.stdout.readline()
            print(f"[XSIM] {line.strip()}")
            if not line:
                raise RuntimeError("Simulation crashed or closed unexpectedly.")

            if line.startswith("@HANDSHAKE_READY"):
                return line
                
            if line.startswith("@OBS"):
                self._parse_obs(line)
                found_obs = True
                continue
                
            if "$stop called at time" in line:
                if found_obs: return line
                else: continue

    def _parse_obs(self, obs_str: str):
        """Parses the minimum required physical signals from XSIM."""
        parts = obs_str.split()[1:]
        self.hw_state["top_in_ready"] = int(parts[0])
        self.hw_state["top_row_valid"] = int(parts[1])
        # We ignore hardware FIFO counts from XSIM as we track them cleanly in Python
        self.hw_state["mismatches"] = int(parts[3])

    def action_masks(self) -> np.ndarray:
        """
        Calculates the action mask. Only masks out a subdecoder if it physically 
        cannot accept the VERY NEXT command group in its queue OR if its dispatched
        rows exceed the SKEW_BOUND relative to the global barrier.
        """
        mask = np.zeros(10, dtype=bool)
        
        # 1. Hardware Back-pressure Fallback (Physical Stall forces NO_OP)
        if self.hw_state.get("top_in_ready", 0) == 0:
            mask[NO_OP_ACTION] = True
            return mask
            
        work_remaining = False
        for i in range(NUM_DECODERS):
            if len(self.cmds[i]) > 0:
                work_remaining = True

                # --- SKEW GATE: Prevent channels from starving the barrier ---
                if (self.dispatched_rows[i] - self.topr_cnt) >= SKEW_BOUND:
                    continue # Masked: Channel has raced too far ahead of the global barrier!

                # Protect Bypass Lane
                if i == 8:
                    if self.bypass_row_cnt <= (self.BYPASS_Q_MAX - 2): # Needs space for 2 rows
                        mask[i] = True
                    continue

                # Protect Regular Sub-decoders
                if self.row_queue_cnt[i] >= self.ROW_Q_MAX:
                    continue # Stalled at output barrier, cannot execute/accept properly
                
                # Check if the FIRST command group fits in the Shadow FIFOs
                raw_cmd = self.cmds[i][0]
                needed = decode_token_slots(raw_cmd).slots_needed
                
                # Simulate FIFO check for this group
                temp_cmd = self.cmd_fifo_cnt[i]
                temp_lit = list(self.lit_fifo_cnt[i])
                fits_fifo = True
                
                for offset in range(needed):
                    if offset >= len(self.cmds[i]):
                        break # Prevent out of bounds if dataset finishes cleanly mid-group
                        
                    look_tok = int(self.cmds[i][offset])
                    look_type = (look_tok >> 16) & 0xF
                    
                    if look_type == LIT_TYPE:
                        found_bank = -1
                        for b in range(4):
                            if temp_lit[b] < self.LIT_MAX[b]:
                                found_bank = b
                                break
                        if found_bank != -1:
                            temp_lit[found_bank] += 1
                        else:
                            fits_fifo = False
                            break
                    elif look_type in [CMD_TYPE, RLE_TYPE]:
                        if temp_cmd < self.CMD_MAX:
                            temp_cmd += 1
                        else:
                            fits_fifo = False
                            break
                            
                if fits_fifo:
                    mask[i] = True

        if not work_remaining:
            mask[NO_OP_ACTION] = True
            
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        restarted = self._restart_sim_if_needed()

        selected_idx = self.np_random.integers(0, len(self.dataset_pool))
        selected_stem_path, selected_dataset = self.dataset_pool[selected_idx]

        self.cmds = [deque(list(queue)) for queue in selected_dataset]
        self.initial_tokens = sum(len(cmd) for cmd in self.cmds)
        self.cycles = 0
        self.stall_cycles = 0
        self._init_shadow_model()

        if self.sim_proc is None:
            self._start_sim()
        elif not restarted:
            self.sim_proc.stdin.write("restart\n")
            self.sim_proc.stdin.flush()

        self._resume_sim()
        self._wait_for_obs() 
        
        stem_str = str(selected_stem_path).replace("\\", "/")
        stem_str = str(selected_stem_path).replace("beats_hex", "rows_hex")
        self._write_action(stem_str)
        self._resume_sim()
        self._wait_for_obs()
        
        return self._get_obs(), {}

    def _push_shadow_token(self, dest: int, token: int):
        """Updates internal Python FIFOs and Dispatched Skew Trackers with a single 16-bit token."""
        if dest == 8: return 
            
        t_val = int(token)
        payload = t_val & 0xFFFF
        type_code = (t_val >> 16) & 0xF
        
        if type_code == LIT_TYPE:
            self.internal_lit_q[dest].append(payload)
            for bank in range(4):
                if self.lit_fifo_cnt[dest][bank] < self.LIT_MAX[bank]:
                    self.lit_fifo_cnt[dest][bank] += 1
                    break
        elif type_code in [CMD_TYPE, RLE_TYPE]:
            self.internal_cmd_q[dest].append(t_val)
            self.cmd_fifo_cnt[dest] += 1
            
            # Update the theoretical dispatched rows for the skew limit calculation
            is_rle = (type_code == RLE_TYPE) or (type_code == CMD_TYPE and ((payload >> 15) & 0x1 == 1))
            lit_field = (payload >> 11) & 0xF
            copy_len = (payload >> 7) & 0xF

            if is_rle:
                rle_length = (payload >> 11) & 0xF
                out_bytes = rle_length + 1
            else:
                is_row_repeat = (lit_field == 0 and copy_len == 15)
                
                if is_row_repeat:
                    out_bytes = 0 
                    self.dispatched_rows[dest] += (payload & 0x7F) + 1 # ROW_REPEAT jumps the row counter
                    self.dispatched_byte_acc[dest] = 0
                else:
                    lit_bytes = lit_field * 2
                    copy_bytes = 0 if copy_len == 15 else copy_len + 2
                    out_bytes = lit_bytes + copy_bytes
                    
            if not (type_code == CMD_TYPE and lit_field == 0 and copy_len == 15):
                self.dispatched_byte_acc[dest] += out_bytes
                while self.dispatched_byte_acc[dest] >= 16:
                    self.dispatched_byte_acc[dest] -= 16
                    self.dispatched_rows[dest] += 1

    def _step_execution_cycle(self):
        """Ticks the hardware clock by 1 within the Python Shadow Model."""
        for i in range(8):
            if self.row_queue_cnt[i] >= self.ROW_Q_MAX:
                continue
                
            if self.active_cmd_cycles[i] > 1:
                self.active_cmd_cycles[i] -= 1
                continue
                
            if self.cmd_fifo_cnt[i] > 0:
                token = self.internal_cmd_q[i].popleft()
                self.cmd_fifo_cnt[i] -= 1
                
                payload = token & 0xFFFF
                type_code = (token >> 16) & 0xF
                is_rle = (type_code == RLE_TYPE) or (type_code == CMD_TYPE and ((payload >> 15) & 0x1 == 1))
                
                if is_rle:
                    rle_length = (payload >> 11) & 0xF
                    self.active_cmd_cycles[i] = 2 if rle_length > 7 else 1
                    out_bytes = rle_length + 1
                else:
                    lit_field = (payload >> 11) & 0xF
                    copy_len = (payload >> 7) & 0xF
                    
                    self.active_cmd_cycles[i] = 1
                    if lit_field >= 4:
                        self.active_cmd_cycles[i] = (lit_field // 4) + (1 if (lit_field % 4) else 0)
                        
                    for _ in range(lit_field):
                        if self.internal_lit_q[i]:
                            self.internal_lit_q[i].popleft()
                            for bank in range(3, -1, -1): # Drain from highest occupied bank
                                if self.lit_fifo_cnt[i][bank] > 0:
                                    self.lit_fifo_cnt[i][bank] -= 1
                                    break
                                    
                    if lit_field == 0 and copy_len == 15:
                        out_bytes = 0 
                        self.byte_accumulators[i] += (payload & 0x7F) * 16 + 16
                    else:
                        lit_bytes = lit_field * 2
                        copy_bytes = 0 if copy_len == 15 else copy_len + 2
                        out_bytes = lit_bytes + copy_bytes
                        
                self.byte_accumulators[i] += out_bytes
                
                while self.byte_accumulators[i] >= 16:
                    self.byte_accumulators[i] -= 16
                    self.row_queue_cnt[i] += 1
                    self.total_rows_emitted[i] += 1

        self._check_row_barrier()

    def _check_row_barrier(self):
        """Global synchronization barrier using Python states."""
        all_ready = True
        for i in range(8):
            if self.row_queue_cnt[i] < 1:
                all_ready = False
                break
        if self.bypass_row_cnt < 1:
            all_ready = False
            
        if all_ready:
            for i in range(8):
                self.row_queue_cnt[i] -= 1
            self.bypass_row_cnt -= 1
            self.topr_cnt += 1 # Advance the global barrier count

    def step(self, action):
        self.cycles += 1
        
        valid_bit = 1
        payload_128b = 0
        slots_used = 0
        commands_packed = 0

        if int(action) == NO_OP_ACTION:
            valid_bit = 0
            action_dest = 0
            print("[Env] NO_OP Action")
        else:
            print("[Env] Action:", action)
            action_dest = int(action)
            
            # Temporary predictive trackers to exact-pack during the loop
            if action_dest < 8:
                temp_cmd = self.cmd_fifo_cnt[action_dest]
                temp_lit = list(self.lit_fifo_cnt[action_dest])
            else:
                temp_bypass = self.bypass_row_cnt
            
            # Packing loop
            while slots_used < BEATS_ON_BUS and self.cmds[action_dest]:
                # bypass lane has special logic
                if action_dest == 8: 
                    needed = 1
                    if slots_used + needed <= BEATS_ON_BUS and temp_bypass + 2 <= self.BYPASS_Q_MAX:
                        temp_bypass += 2
                    else:
                        break
                else:
                    raw_cmd = self.cmds[action_dest][0]
                    needed = decode_token_slots(raw_cmd).slots_needed
                    
                    if slots_used + needed > BEATS_ON_BUS:
                        break 
                        
                    # Simulate FIFO space for the next command group
                    temp_cmd_future = temp_cmd
                    temp_lit_future = list(temp_lit)
                    fits_fifo = True
                    
                    for offset in range(needed):
                        look_tok = int(self.cmds[action_dest][offset])
                        look_type = (look_tok >> 16) & 0xF
                        if look_type == LIT_TYPE:
                            found_bank = -1
                            for b in range(4):
                                if temp_lit_future[b] < self.LIT_MAX[b]:
                                    found_bank = b
                                    break
                            if found_bank != -1:
                                temp_lit_future[found_bank] += 1
                            else:
                                fits_fifo = False
                                break
                        elif look_type in [CMD_TYPE, RLE_TYPE]:
                            if temp_cmd_future < self.CMD_MAX:
                                temp_cmd_future += 1
                            else:
                                fits_fifo = False
                                break
                                
                    if not fits_fifo:
                        break # Prevent silent data drop! Send partially filled beat.
                        
                    temp_cmd = temp_cmd_future
                    temp_lit = temp_lit_future

                # The group fits perfectly. Safely pop and pack it into the payload.
                for _ in range(needed):
                    token = self.cmds[action_dest].popleft()
                    # update the FIFO counts we keep track of in Python
                    self._push_shadow_token(action_dest, token)
                    clean_16b_token = token & 0xFFFF
                    payload_128b = payload_128b | (clean_16b_token << (slots_used * 16))
                    slots_used += 1

                commands_packed += 1
            
            if action_dest == 8 and slots_used > 0:
                self.bypass_row_cnt += (2 * slots_used)
                self.total_rows_emitted[8] += (2 * slots_used)
                self.dispatched_rows[8] += (2 * slots_used) # Update bypass skew

        dest_bin = f"{action_dest:04b}"
        
        if action_dest == 8 or commands_packed == 0:
            ncmd_bin = "000"
        else:
            ncmd_bin = f"{commands_packed - 1:03b}"
            
        payload_bin = f"{payload_128b:0128b}"
        full_135b_string = dest_bin + ncmd_bin + payload_bin
        action_hex = f"{int(full_135b_string, 2):034x}"
        print(f"[Env] Action hex: {action_hex}")
        # Advance Shadow Model & Ping-Pong
        self._step_execution_cycle()
        self._write_action(f"{valid_bit} {action_hex}")
        self._resume_sim()
        self._wait_for_obs()

        # Rewards calculation
        reward = -1.0 
        
        if self.hw_state["mismatches"] > 0:
            reward -= 20.0
            print(f"[Env] Hardware Mismatch Detected! Total Mismatches: {self.hw_state['mismatches']}")
            print(f"[Env] action_hex: {action_hex}")
            return self._get_obs(), reward, True, False, {"deadlock": False, "mismatch": True}

        if slots_used == 0 and any(len(q) > 0 for q in self.cmds):
            self.stall_cycles += 1
            reward -= 2.0 
        else:
            self.stall_cycles = 0

        # positive reward for fully utilizing the bus
        if slots_used == BEATS_ON_BUS:
            reward += 0.1

        # if stalled, exit early 
        if getattr(self, 'stall_cycles', 0) > 20:
            reward -= 100.0
            return self._get_obs(), reward, True, False, {"deadlock": True}

        # Reward Shaping
        row_spread = max(self.total_rows_emitted) - min(self.total_rows_emitted)
        imbalance_penalty = 2.0 if row_spread > 2 else 0.0

        # hoard_penalty = 0.0
        # for s in range(8):
        #     cmd_util = self.cmd_fifo_cnt[s] / CMD_FIFO_SIZE
        #     if cmd_util > FIFO_HOARD_THRESH:
        #         hoard_penalty += 0.5
                
        # reward -= (imbalance_penalty + hoard_penalty)
        reward -= imbalance_penalty

        terminated = False
        if sum(len(q) for q in self.cmds) == 0:
            hw_clear = True
            for s in range(8):
                if self.row_queue_cnt[s] > 0 or self.cmd_fifo_cnt[s] > 0:
                    hw_clear = False
            if hw_clear and self.bypass_row_cnt == 0:
                self.completed_images += 1
                terminated = True
                reward += 10.0

        info = {"makespan": self.cycles, "images_completed": self.completed_images} if terminated else {}
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        obs = {
            "top_in_ready": np.array(self.hw_state["top_in_ready"], dtype=np.int64),
            "top_row_valid": np.array(self.hw_state["top_row_valid"], dtype=np.int64),
            "cmd_fifo_fill": np.zeros(8, dtype=np.float32),
            "lit_fifo_fill": np.zeros(8, dtype=np.float32),
            "output_fifo_fill": np.zeros(8, dtype=np.float32),
            "bypass_fifo_fill": np.array([self.bypass_row_cnt / BYPASS_IN_SIZE], dtype=np.float32),
            "lookahead_types": np.full((NUM_DECODERS, LOOKAHEAD_WINDOW), -1, dtype=np.int32), 
            "global_progress": np.array([0.0], dtype=np.float32),
            "cycles_left": np.zeros(NUM_DECODERS, dtype=np.float32),
            "lookahead_cycles": np.full((NUM_DECODERS, LOOKAHEAD_WINDOW), -1, dtype=np.float32), 
            "lookahead_is_barrier": np.full((NUM_DECODERS, LOOKAHEAD_WINDOW), -1, dtype=np.int32), 
            "channel_skew": np.zeros(NUM_DECODERS, dtype=np.float32),
        }

        for i in range(NUM_DECODERS):
            is_bypass = (i == 8)
            bus_queue = self.cmds[i]
            
            if len(bus_queue) > 0:
                head_info = self._decode_token(bus_queue[0], is_bypass)
                obs["cycles_left"][i] = min(head_info["cycles"] / MAX_CYCLES_LEFT, 1.0)
            else:
                obs["cycles_left"][i] = 0.0

            window_size = min(len(bus_queue), LOOKAHEAD_WINDOW)
            for j in range(window_size):
                raw_token = bus_queue[j]
                token_info = self._decode_token(raw_token, is_bypass) 
                
                obs["lookahead_cycles"][i, j] = min(token_info["cycles"] / MAX_CYCLES_LEFT, 1.0)
                obs["lookahead_is_barrier"][i, j] = float(token_info["is_barrier"])
                
            # Populate Skew Observation (Normalized 0.0 to 1.0 based on SKEW_BOUND)
            skew_val = self.dispatched_rows[i] - self.topr_cnt
            obs["channel_skew"][i] = min(max(skew_val / SKEW_BOUND, 0.0), 1.0)

        for s in range(8):
            obs["cmd_fifo_fill"][s] = self.cmd_fifo_cnt[s] / CMD_FIFO_SIZE
            obs["output_fifo_fill"][s] = self.row_queue_cnt[s] / OUT_FIFO_SIZE
            
            max_lit_ratio = max(
                self.lit_fifo_cnt[s][0] / 24.0,
                self.lit_fifo_cnt[s][1] / 24.0,
                self.lit_fifo_cnt[s][2] / 18.0,
                self.lit_fifo_cnt[s][3] / 12.0
            )
            obs["lit_fifo_fill"][s] = max_lit_ratio
            
        for i in range(NUM_DECODERS):
            window_size = min(len(self.cmds[i]), LOOKAHEAD_WINDOW)
            for j in range(window_size):
                if i == 8:
                    obs["lookahead_types"][i, j] = CMD_TYPE 
                else:
                    tok = decode_token_slots(self.cmds[i][j])
                    type_mapping = {"CMD": 0, "LIT": 1, "RLE": 2, "SEED": 3}
                    obs["lookahead_types"][i, j] = type_mapping.get(tok.category, -1)
                    
        total_remaining = sum(len(q) for q in self.cmds)
        obs["global_progress"][0] = total_remaining / max(1, self.initial_tokens)

        return obs

    def close(self):
        if not self.sim_proc:
            return

        try:
            if self.sim_proc.poll() is None:
                try:
                    self._write_action("0 0000000000000000000000000000000000")
                    self._resume_sim()
                except Exception:
                    pass

                try:
                    if self.sim_proc.stdin and not self.sim_proc.stdin.closed:
                        self.sim_proc.stdin.write("quit\n")
                        self.sim_proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass

                try:
                    self.sim_proc.wait(timeout=2)
                except Exception:
                    try:
                        self.sim_proc.terminate()
                        self.sim_proc.wait(timeout=2)
                    except Exception:
                        pass
        finally:
            try:
                if self.sim_proc.stdin and not self.sim_proc.stdin.closed:
                    self.sim_proc.stdin.close()
            except Exception:
                pass
            self.sim_proc = None
