"""
Hardware-in-the-Loop Environment for the T8 Decompressor.
Uses Tcl-Ping-Pong to step a live Vivado SystemVerilog simulation.
"""
import os
import subprocess
from pathlib import Path
from collections import deque

import gymnasium as gym
import numpy as np
from torch import signal

VIVADO_VERBOSE = False

# Hardware Constants
NUM_DECODERS = 9      # 0-7 are LC/RLE, 8 is Bypass
NO_OP_ACTION = 9      # Action index to yield the bus (Valid=0)
BEATS_ON_BUS = 8      # Max 16-bit slots per beat
MAX_CYCLES_LEFT = 16 
LOOKAHEAD_WINDOW = 64

# FIFO capacities used to normalize XSim-reported occupancy counts.
CMD_FIFO_SIZE = 16
LIT_FIFO_CAPACITIES = (24, 24, 18, 12)
OUT_FIFO_SIZE = 4
BYPASS_FIFO_SIZE = 8

DEADLOCK_STALL_CYCLES = 500
CMD_TYPE=0
LIT_TYPE=1
RLE_TYPE=2

def decode_token_slots(token: int) -> tuple[str, int]:
    """Return a token's category and the number of bus slots in its command group."""
    t = int(token)
    payload = t & 0xFFFF
    type_code = (t >> 16) & 0xF
    
    if type_code == LIT_TYPE:
        return "LIT", 1
        
    if type_code not in [CMD_TYPE, LIT_TYPE, RLE_TYPE]:
        return "SEED", 1

    is_rle = (type_code == RLE_TYPE) or (type_code == CMD_TYPE and ((payload >> 15) & 0x1 == 1))

    if is_rle:
        return "RLE", 2
    else: 
        lit_field = (payload >> 11) & 0xF
        return "CMD", 1 + lit_field

class DecoderEnvXSim(gym.Env):
    def __init__(self, data_dir: str, sim_dir: str, is_inference: bool = False):
        super().__init__()
        self.data_dir = Path(data_dir).resolve()
        self.sim_dir = Path(sim_dir).resolve()
        self.run_dir = self.sim_dir / "xsim_run" / "run"
        self.action_file = self.run_dir / "action.txt"
        
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
        
        # Observation Space combining live hardware state and input lookahead
        self.observation_space = gym.spaces.Dict({
            "top_in_ready": gym.spaces.Discrete(2),
            "top_row_valid": gym.spaces.Discrete(2),
            "sub_decoder_fifo_counts": gym.spaces.Box(
                low=0.0, high=1.0, shape=(8, 6), dtype=np.float32
            ),
            "cmd_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            "lit_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            "output_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32),
            "bypass_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "lookahead_types": gym.spaces.Box(
                low=0,
                high=1,
                shape=(NUM_DECODERS, LOOKAHEAD_WINDOW, 4),
                dtype=np.int32,
            ),
            "global_progress": gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "cycles_left": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
            "lookahead_cycles": gym.spaces.Box(low=-1.0, high=1.0, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.float32),
            "lookahead_is_barrier": gym.spaces.Box(low=-1.0, high=1.0, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.int32),
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
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        self.sim_proc = subprocess.Popen(
            ["xsim.bat", "tb_rl_interactive_snap", "-tclbatch", "quiet_run.tcl", "-R"],
            cwd=str(self.run_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=CREATE_NEW_PROCESS_GROUP
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
            if VIVADO_VERBOSE: print(f"[XSIM] {line.strip()}")

            if "err" in line:
                print(f"[XSIM ERROR] {line.strip()}")
            if not line:
                raise RuntimeError("Simulation crashed or closed unexpectedly.")

            if line.startswith("@HANDSHAKE_READY"):
                return line
                
            if line.startswith("@OBS"):
                self._parse_obs(line)
                if VIVADO_VERBOSE: print(f"[XSIM] Observation received: {line.strip()}")
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
        self.hw_state["bypass_fifo_count"] = int(parts[2])
        self.hw_state["mismatches"] = int(parts[3])

        self.hw_state["sub_decoders"] = []
        idx = 4
        for _ in range(8):
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

                # check the state of FIFO for sub-decoders
                if i < 8:
                    hw_sd = self.hw_state["sub_decoders"][i]
                
                    # Protect the CMD FIFO
                    if hw_sd["cmd"] > 14:
                        mask[i] = False
                        continue 
                    
                    # Protect ALL 4 LIT Banks based on their unbalanced physical depths (24, 24, 18, 12)
                    if (hw_sd["lit1"] > 24 or 
                        hw_sd["lit2"] > 24 or 
                        hw_sd["lit3"] > 18 or 
                        hw_sd["lit4"] > 12):
                        mask[i] = False
                        continue
                
        if not work_remaining or not mask[:NUM_DECODERS].any():
            mask[NO_OP_ACTION] = True
        return mask

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        restarted = self._restart_sim_if_needed()

        selected_idx = self.np_random.integers(0, len(self.dataset_pool))
        selected_stem_path, selected_dataset = self.dataset_pool[selected_idx]
        self.current_sample_name = str(selected_stem_path.relative_to(self.data_dir))

        self.cmds = [deque(list(queue)) for queue in selected_dataset]
        self.initial_tokens = sum(len(cmd) for cmd in self.cmds)
        self.cycles = 0
        self.stall_cycles = 0

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

    def step(self, action):
        self.cycles += 1
        
        valid_bit = 1
        payload_128b = 0
        slots_used = 0
        commands_packed = 0

        if int(action) == NO_OP_ACTION:
            valid_bit = 0
            action_dest = 0
        else:
            action_dest = int(action)
            
            # Packing loop
            while slots_used < BEATS_ON_BUS and self.cmds[action_dest]:
                if action_dest == 8:
                    needed = 1
                else:
                    raw_cmd = self.cmds[action_dest][0]
                    _, needed = decode_token_slots(raw_cmd)

                if slots_used + needed > BEATS_ON_BUS:
                    break
                if needed > len(self.cmds[action_dest]):
                    break

                # The group fits perfectly. Safely pop and pack it into the payload.
                for _ in range(needed):
                    token = self.cmds[action_dest].popleft()
                    clean_16b_token = token & 0xFFFF
                    payload_128b = payload_128b | (clean_16b_token << (slots_used * 16))
                    slots_used += 1

                commands_packed += 1

        dest_bin = f"{action_dest:04b}"
        
        if action_dest == 8 or commands_packed == 0:
            ncmd_bin = "000"
        else:
            ncmd_bin = f"{commands_packed - 1:03b}"
            
        payload_bin = f"{payload_128b:0128b}"
        full_135b_string = dest_bin + ncmd_bin + payload_bin
        action_hex = f"{int(full_135b_string, 2):034x}"
        self._write_action(f"{valid_bit} {action_hex}")
        self._resume_sim()
        self._wait_for_obs()

        #DEBUG PRINTS
        if VIVADO_VERBOSE:
            print(f"[Env] action_dest: {action_dest}, commands_packed: {commands_packed}, slots_used: {slots_used}")
            print(f"[Env] action: {action_hex}")
            decoder_rows = self.hw_state.get("sub_decoders", [])
            if 0 <= action_dest < len(decoder_rows):
                for i in range(8):
                    d = decoder_rows[i]
                    print(
                        f"[Env] current subdecoder {i}: "
                        f"cmd={d['cmd']}, lit1={d['lit1']}, lit2={d['lit2']}, "
                        f"lit3={d['lit3']}, lit4={d['lit4']}, out={d['out']}"
                    )
                    print()
            else:
                print(f"[Env] current subdecoder {action_dest}: unavailable")

        # Rewards calculation
        reward = -1.0 
        
        if self.hw_state["mismatches"] > 0:
            reward -= 1000.0
            print(f"[Env] Hardware Mismatch Detected! Total Mismatches: {self.hw_state['mismatches']}")
            print(f"[Env] action_hex: {action_hex}")
            return self._get_obs(), reward, True, False, {
                "deadlock": False,
                "mismatch": True,
                "makespan": self.cycles,
                "sample_name": self.current_sample_name,
            }

        if slots_used == 0 and any(len(q) > 0 for q in self.cmds):
            self.stall_cycles += 1
            reward -= 2.0 
        else:
            self.stall_cycles = 0

        # positive reward for fully utilizing the bus
        if slots_used == BEATS_ON_BUS:
            reward += 0.1

        # if stalled, exit early 
        if getattr(self, 'stall_cycles', 0) > DEADLOCK_STALL_CYCLES:
            remaining_cmds = [len(q) for q in self.cmds]
            print(f"[Env] Deadlock detected: stall_cycles={self.stall_cycles}, remaining_cmds={remaining_cmds}")
            reward -= 1000.0

            return self._get_obs(), reward, True, False, {
                "deadlock": True,
                "makespan": self.cycles,
                "sample_name": self.current_sample_name,
            }

        terminated = False
        if sum(len(q) for q in self.cmds) == 0:
            sub_decoders = self.hw_state.get("sub_decoders", [])
            hw_clear = (
                self.hw_state.get("top_row_valid", 0) == 0
                and len(sub_decoders) == 8
                and all(
                    decoder["cmd"] == 0
                    and decoder["lit1"] == 0
                    and decoder["lit2"] == 0
                    and decoder["lit3"] == 0
                    and decoder["lit4"] == 0
                    and decoder["out"] == 0
                    for decoder in sub_decoders
                )
            )
            if hw_clear:
                self.completed_images += 1
                terminated = True
                reward += 10.0

        info = {
            "makespan": self.cycles,
            "images_completed": self.completed_images,
            "sample_name": self.current_sample_name,
        } if terminated else {}
        return self._get_obs(), reward, terminated, False, info

    def _get_obs(self):
        obs = {
            "top_in_ready": np.array(self.hw_state["top_in_ready"], dtype=np.int64),
            "top_row_valid": np.array(self.hw_state["top_row_valid"], dtype=np.int64),
            "sub_decoder_fifo_counts": np.zeros((8, 6), dtype=np.float32),
            "cmd_fifo_fill": np.zeros(8, dtype=np.float32),
            "lit_fifo_fill": np.zeros(8, dtype=np.float32),
            "output_fifo_fill": np.zeros(8, dtype=np.float32),
            "bypass_fifo_fill": np.array(
                [self.hw_state.get("bypass_fifo_count", 0) / BYPASS_FIFO_SIZE],
                dtype=np.float32,
            ),
            "lookahead_types": np.zeros((NUM_DECODERS, LOOKAHEAD_WINDOW, 4), dtype=np.int32),
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

        for i, decoder in enumerate(self.hw_state.get("sub_decoders", [])[:8]):
            obs["sub_decoder_fifo_counts"][i] = [
                decoder["cmd"] / CMD_FIFO_SIZE,
                decoder["lit1"] / LIT_FIFO_CAPACITIES[0],
                decoder["lit2"] / LIT_FIFO_CAPACITIES[1],
                decoder["lit3"] / LIT_FIFO_CAPACITIES[2],
                decoder["lit4"] / LIT_FIFO_CAPACITIES[3],
                decoder["out"] / OUT_FIFO_SIZE,
            ]
            obs["cmd_fifo_fill"][i] = decoder["cmd"] / CMD_FIFO_SIZE
            obs["lit_fifo_fill"][i] = max(
                decoder["lit1"] / LIT_FIFO_CAPACITIES[0],
                decoder["lit2"] / LIT_FIFO_CAPACITIES[1],
                decoder["lit3"] / LIT_FIFO_CAPACITIES[2],
                decoder["lit4"] / LIT_FIFO_CAPACITIES[3],
            )
            obs["output_fifo_fill"][i] = decoder["out"] / OUT_FIFO_SIZE
            
        for i in range(NUM_DECODERS):
            window_size = min(len(self.cmds[i]), LOOKAHEAD_WINDOW)
            for j in range(window_size):
                if i == 8:
                    category = "CMD"
                else:
                    category, _ = decode_token_slots(self.cmds[i][j])
                type_index = {"CMD": 0, "LIT": 1, "RLE": 2, "SEED": 3}[category]
                obs["lookahead_types"][i, j, type_index] = 1
                    
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
                os.kill(self.sim_proc.pid, signal.CTRL_BREAK_EVENT)
                self.sim_proc.wait(timeout=2)
                if self.sim_proc.stdin and not self.sim_proc.stdin.closed:
                    self.sim_proc.stdin.close()
            except Exception:
                pass
            self.sim_proc = None
