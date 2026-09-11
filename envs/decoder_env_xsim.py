"""
Hardware-in-the-Loop Environment for the T8 Decompressor.
Uses Tcl-Ping-Pong to step a live Vivado SystemVerilog simulation.
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

CMD_TYPE=0
LIT_TYPE=1
RLE_TYPE=2

@dataclass
class TokenInfo:
    raw_val: int
    category: str        
    req_lit_chunks: int  
    slots_needed: int

def decode_token_slots(token: int) -> TokenInfo:
    """Decodes a token purely to calculate its bus footprint."""
    t = int(token)
    payload = t & 0xFFFF
    type_code = (t >> 16) & 0xF
    
    if type_code == LIT_TYPE:
        return TokenInfo(t, "LIT", 1, 1)
        
    if type_code not in [CMD_TYPE, LIT_TYPE, RLE_TYPE]:
        return TokenInfo(t, "SEED", 0, 1)

    is_rle = (type_code == RLE_TYPE) or (type_code == CMD_TYPE and ((payload >> 15) & 0x1 == 1))

    if is_rle:
        return TokenInfo(t, "RLE", 0, 2) # RLE consumes 2 slots (CMD + Pattern)
    else: 
        lit_field = (payload >> 11) & 0xF
        return TokenInfo(t, "CMD", int(lit_field), 1 + int(lit_field))

class DecoderEnvXSim(gym.Env):
    def __init__(self, data_dir: str, sim_dir: str, is_inference: bool = False):
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.sim_dir = Path(sim_dir).resolve()
        self.run_dir = self.sim_dir / "xsim_run" / "run"
        self.action_file = self.run_dir / "action.txt"

        # Load data
        if is_inference:
            # ... (keep your existing inference folder check) ...
            
            cmds = self._load_hex_folder(hex_dir)
            # DYNAMIC STEM EXTRACTION:
            # Grab the prefix from a file like "test_11_lantern_subdec0_beats.hex"
            first_file = next(hex_dir.glob('*.hex')).name
            stem_name = first_file.split('_subdec')[0] 
            stem_path = hex_dir / stem_name
            
            # Store it as a tuple: (path, dataset)
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
                            # DYNAMIC STEM EXTRACTION:
                            first_file = next(hex_dir.glob('*.hex')).name
                            stem_name = first_file.split('_subdec')[0]
                            stem_path = hex_dir / stem_name
                            
                            all_datasets.append((stem_path, cmds)) # Append tuple
                        except Exception as e:
                            print(f"Skipping {test_folder.name} due to error: {e}")

            self.dataset_pool = all_datasets
            if not all_datasets:
                raise ValueError(f"No valid datasets found in {data_dir}")
        
        # Action Space: 0-7 (Regular), 8 (Bypass), 9 (Yield Bus)
        self.action_space = gym.spaces.Discrete(10)
        
        # Observation Space combining Hardware State and Python Lookahead
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
        })

        self.sim_proc = None
        self.hw_state = {}

    def _load_hex_folder(self, cmd_path: str) -> list:
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

    def _decode_token(self, token: int, is_bypass: bool) -> dict:
        """Parses a raw integer ONCE and returns hardware execution traits."""
        if is_bypass:
            return {"category": "CMD", "is_barrier": True, "cycles": 1}

        t = int(token)
        payload = t & 0xFFFF
        type_code = (t >> 16) & 0xF
        
        # 1. Identify LITs and raw SEEDs
        if type_code == 1: # LIT_TYPE
            return {"category": "LIT", "is_barrier": False, "cycles": 0}
        if type_code not in [0, 1, 2]: # CMD, LIT, RLE
            return {"category": "SEED", "is_barrier": False, "cycles": 0}

        is_rle = (type_code == 2) or (type_code == 0 and ((payload >> 15) & 0x1 == 1))

        # 2. Compute Cycles and Barrier Status
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
        
        print("Starting Vivado HIL Subprocess...")
        self.sim_proc = subprocess.Popen(
            ["xsim.bat", "tb_rl_interactive_snap"],
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
        while True:
            line = self.sim_proc.stdout.readline()
            if not line:
                raise RuntimeError("Simulation crashed or closed unexpectedly.")
            line_str = line.strip()
            
            if not line_str.startswith("@OBS"): 
                print(f"[Vivado] {line_str}")
            
            if line_str.startswith("@HANDSHAKE_READY"):
                print(f"[Vivado] {line_str}")
                return line_str
                
            if line_str.startswith("@OBS"):
                print(f"[Vivado] {line_str}")
                self._parse_obs(line_str)
                return line_str

    def _parse_obs(self, obs_str: str):
        """Parses the @OBS string directly from Verilog."""
        parts = obs_str.split()[1:] # Skip "@OBS"
        
        self.hw_state["top_in_ready"] = int(parts[0])
        self.hw_state["top_row_valid"] = int(parts[1])
        self.hw_state["bypass_in_cnt"] = int(parts[2])
        self.hw_state["mismatches"] = int(parts[3])
        
        self.hw_state["sub_decoders"] = []
        idx = 4
        for s in range(8):
            self.hw_state["sub_decoders"].append({
                "cmd": int(parts[idx]),
                "lit1": int(parts[idx+1]),
                "lit2": int(parts[idx+2]),
                "lit3": int(parts[idx+3]),
                "lit4": int(parts[idx+4]),
                "out": int(parts[idx+5])
            })
            idx += 6

    def action_masks(self) -> np.ndarray:
        mask = np.zeros(10, dtype=bool)
        
        # If hardware is stalled, all routes are blocked except Yield (No-Op)
        if self.hw_state.get("top_in_ready", 0) == 0:
            mask[NO_OP_ACTION] = True
            return mask
            
        # Standard software tracking
        work_remaining = False
        for i in range(NUM_DECODERS):
            if len(self.cmds[i]) > 0:
                mask[i] = True
                work_remaining = True
                
        if not work_remaining:
            mask[NO_OP_ACTION] = True
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Randomly select one dataset from the pool for this episode
        selected_idx = self.np_random.integers(0, len(self.dataset_pool))
        
        # Unpack the specific path stem and the data queues
        selected_stem_path, selected_dataset = self.dataset_pool[selected_idx]

        self.cmds = [deque(list(queue)) for queue in selected_dataset]
        self.initial_tokens = sum(len(cmd) for cmd in self.cmds)
        self.cycles = 0
        
        # Reboot the simulator for a clean hardware state
        self._start_sim()
        #
        self._resume_sim()
        # Wait for @HANDSHAKE_READY
        self._wait_for_obs() 
        
        # Send the exact initialization path for this specific dataset!
        stem_str = str(selected_stem_path).replace("\\", "/")
        # the simulation needs the rows file in order to check output, replace the path
        stem_str = str(selected_stem_path).replace("beats_hex", "rows_hex")
        print(f"[Env] Sending Handshake Stem: {stem_str}")
        self._write_action(stem_str)
        self._resume_sim()
        
        # Wait for reset sequence and first true observation
        self._wait_for_obs()
        
        return self._get_obs(), {}

    def step(self, action):
        self.cycles += 1
        
        valid_bit = 1
        payload_128b = 0
        slots_used = 0

        if action == NO_OP_ACTION:
            valid_bit = 0
            action_dest = 0
        else:
            action_dest = action
            
            # Pack up to 8 slots
            while slots_used < BEATS_ON_BUS and self.cmds[action]:
                if action == 8: # Bypass lane takes raw data
                    needed = 1
                else:
                    raw_cmd = self.cmds[action][0]
                    needed = decode_token_slots(raw_cmd).slots_needed
                    
                if slots_used + needed <= BEATS_ON_BUS:
                    # pop from the left to keep endian-ness
                    for _ in range(needed):
                        token = self.cmds[action].popleft()
                        
                        # Place token into the correct slot (Slot 0 = LSB) ---
                        payload_128b = payload_128b | (token << (slots_used * 16))
                        slots_used += 1
                else:
                    break

        # 135-bit payload: [134:131]=dest, [130:3]=payload, [2:0]=0
        # Format the 4-bit destination index as a binary string
        dest_bin = f"{action_dest:04b}"
        # Format the 128-bit packed payload as a binary string
        payload_bin = f"{payload_128b:0128b}"
        # Create the 3-bit zero padding for the bottom bits [2:0]
        padding_bin = "000"
        
        # Concatenate them into a perfect 135-bit string
        full_135b_string = dest_bin + payload_bin + padding_bin
        
        # Convert the binary string directly into a 34-character hex string for Vivado
        action_hex = f"{int(full_135b_string, 2):034x}"
        
        # Ping-Pong Hardware
        self._write_action(f"{valid_bit} {action_hex}")
        self._resume_sim()
        self._wait_for_obs()

        # Reward calculation
        reward = -1.0 
        
        # Hardware Corruption Penalty
        if self.hw_state["mismatches"] > 0:
            reward -= 1000.0
            return self._get_obs(), reward, True, False, {"deadlock": True, "mismatch": True}

        # Micro-Penalty for Wasting the Bus
        # If no slots were packed (agent chose NO_OP or target was empty) but work remains
        if slots_used == 0 and any(len(q) > 0 for q in self.cmds):
            self.stall_cycles += 1
            reward -= 2.0 
        else:
            self.stall_cycles = 0
        # give positive reward if bus is fully utilized
        if slots_used == BEATS_ON_BUS:
            reward += 0.1

        #  Proportional Deadlock Penalty
        if getattr(self, 'stall_cycles', 0) > 500:
            reward -= 1000.0
            return self._get_obs(), reward, True, False, {"deadlock": True}

        # Reward Shaping Phase (Throughput & Imbalance)
        # Calculate Row Spread directly from HW state
        out_counts = [self.hw_state["sub_decoders"][s]["out"] for s in range(8)]
        row_spread = max(out_counts) - min(out_counts)
        imbalance_penalty = 2.0 if row_spread > 2 else 0.0

        # Calculate Hoard Penalty directly from HW state
        hoard_penalty = 0.0
        for s in range(8):
            cmd_util = self.hw_state["sub_decoders"][s]["cmd"] / CMD_FIFO_SIZE
            if cmd_util > FIFO_HOARD_THRESH:
                hoard_penalty += 0.5
                
        # Apply shaping penalties
        reward -= (imbalance_penalty + hoard_penalty)

        # Check Termination
        terminated = False
        if sum(len(q) for q in self.cmds) == 0:
            # Check if hardware pipelines are empty
            hw_clear = True
            for s in range(8):
                if self.hw_state["sub_decoders"][s]["out"] > 0 or self.hw_state["sub_decoders"][s]["cmd"] > 0:
                    hw_clear = False
            
            if hw_clear and self.hw_state["bypass_in_cnt"] == 0:
                terminated = True

        info = {"makespan": self.cycles} if terminated else {}
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        obs = {
            "top_in_ready": np.array(self.hw_state["top_in_ready"], dtype=np.int64),
            "top_row_valid": np.array(self.hw_state["top_row_valid"], dtype=np.int64),
            "cmd_fifo_fill": np.zeros(8, dtype=np.float32),
            "lit_fifo_fill": np.zeros(8, dtype=np.float32),
            "output_fifo_fill": np.zeros(8, dtype=np.float32),
            "bypass_fifo_fill": np.array([self.hw_state["bypass_in_cnt"] / BYPASS_IN_SIZE], dtype=np.float32),
            "lookahead_types": np.full((NUM_DECODERS, LOOKAHEAD_WINDOW), -1, dtype=np.int32), 
            "global_progress": np.array([0.0], dtype=np.float32),
            "cycles_left": np.zeros(NUM_DECODERS, dtype=np.float32),
            "lookahead_cycles": np.full((NUM_DECODERS, LOOKAHEAD_WINDOW), -1, dtype=np.float32), 
            "lookahead_is_barrier": np.full((NUM_DECODERS, LOOKAHEAD_WINDOW), -1, dtype=np.int32), 
        }

        # Calculate remianing cycles
        MAX_CYCLES_LEFT = 16.0 
        for i in range(NUM_DECODERS):
            is_bypass = (i == 8)
            bus_queue = self.cmds[i]
            
            # Estimate immediate 'cycles_left' based on the command at the head of the queue
            if len(bus_queue) > 0:
                head_info = self._decode_token(bus_queue[0], is_bypass)
                obs["cycles_left"][i] = min(head_info["cycles"] / MAX_CYCLES_LEFT, 1.0)
            else:
                obs["cycles_left"][i] = 0.0

            # Populate Lookahead Buffers
            window_size = min(len(bus_queue), LOOKAHEAD_WINDOW)
            
            for j in range(window_size):
                raw_token = bus_queue[j]
                token_info = self._decode_token(raw_token, is_bypass) 
                
                obs["lookahead_cycles"][i, j] = min(token_info["cycles"] / MAX_CYCLES_LEFT, 1.0)
                obs["lookahead_is_barrier"][i, j] = float(token_info["is_barrier"])

        # Map Hardware state
        for s in range(8):
            sd_state = self.hw_state["sub_decoders"][s]
            obs["cmd_fifo_fill"][s] = sd_state["cmd"] / CMD_FIFO_SIZE
            obs["output_fifo_fill"][s] = sd_state["out"] / OUT_FIFO_SIZE
            
            # Map LIT fill to the maximum occupancy ratio across the 4 banks
            max_lit_ratio = max(
                sd_state["lit1"] / 24.0,
                sd_state["lit2"] / 24.0,
                sd_state["lit3"] / 18.0,
                sd_state["lit4"] / 12.0
            )
            obs["lit_fifo_fill"][s] = max_lit_ratio
            
        # Map Software State (Lookahead)
        for i in range(NUM_DECODERS):
            window_size = min(len(self.cmds[i]), LOOKAHEAD_WINDOW)
            for j in range(window_size):
                if i == 8:
                    obs["lookahead_types"][i, j] = CMD_TYPE # Bypass is raw
                else:
                    tok = decode_token_slots(self.cmds[i][j])
                    type_mapping = {"CMD": 0, "LIT": 1, "RLE": 2, "SEED": 3}
                    obs["lookahead_types"][i, j] = type_mapping.get(tok.category, -1)
                    
        total_remaining = sum(len(q) for q in self.cmds)
        obs["global_progress"][0] = total_remaining / max(1, self.initial_tokens)

        return obs

    def close(self):
        if self.sim_proc:
            try:
                self._write_action("0 0")
                self._resume_sim()
                self.sim_proc.stdin.write("quit\n")
                self.sim_proc.stdin.flush()
                self.sim_proc.wait(timeout=2)
            except Exception:
                pass
            self.sim_proc = None