"""
New version of a python environment that simulates the T8 decompressor

now implements:
- Action chunking
- Curriculum based learning
"""
from pathlib import Path
from collections import deque
from dataclasses import dataclass
import random

import gymnasium as gym
import numpy as np

LIT_FIFO_SIZE = 72 # 24/24/18/12
CMD_FIFO_SIZE = 16
BYPASS_FIFO_SIZE = 8
BEATS_ON_BUS = 8
NUM_DECODERS = 9
LOOKAHEAD_WINDOW = 64 # Deep lookahead for Action Masking / PPO planning
OUTPUT_ROW_BUFFER_SIZE = 4 # How many rows ahead a subdecoder can get before stalling
FIFO_HOARD_THRESH = 0.75

CMD_TYPE=0
LIT_TYPE=1
RLE_TYPE=2
MAX_CYCLES_LEFT = 16 # normalization constant for observation

@dataclass
class TokenInfo:
    """A cleanly parsed representation of a hardware token."""
    raw_val: int
    category: str        # "CMD", "LIT", "RLE", "SEED"
    req_lit_chunks: int  # 0 to 15
    is_barrier: bool     # True for 0x0780
    requires_seed: bool  # True for RLE period_class 7
    cycles: int          # Pre-calculated execution time

class SubDecoder():
    def __init__(self, is_bypass=False):
        # NOTE: 1 chunk = 16 bit / 2 bytes
        self.LIT_FIFO = deque(maxlen=LIT_FIFO_SIZE)
        
        # Bypass lane has a much smaller CMD FIFO (skid buffer) in the RTL
        fifo_depth = BYPASS_FIFO_SIZE if is_bypass else CMD_FIFO_SIZE
        # Bypass lane can only accept 2 rows per bus cycle (1 beat = 2 rows)
        self.bus_transfer_limit = 2 if is_bypass else BEATS_ON_BUS

        # 16 chunk CMD FIFO buffer (or smaller for bypass)
        self.CMD_FIFO = deque(maxlen=fifo_depth)
        
        # 4-row output FIFO
        self.OUTPUT_FIFO = deque(maxlen=OUTPUT_ROW_BUFFER_SIZE)
        
        # Execution State
        self.cycles_left = 0
        self.active_cycles = 0
        self.is_lit_starved = False
        self.is_bypass = is_bypass
        
    def decode_token(self, token: int) -> TokenInfo:
        if self.is_bypass:
            # Bypass rows take exactly 1 cycle and instantly output a row (barrier)
            return TokenInfo(int(token), "CMD", 0, True, False, 1)

        """Parses a raw integer ONCE and returns all hardware traits."""
        t = int(token)
        payload = t & 0xFFFF
        type_code = (t >> 16) & 0xF
        
        # 1. Identify LITs
        if type_code == LIT_TYPE:
            return TokenInfo(t, "LIT", 1, False, False, 0)
            
        # 2. Identify Raw Seeds (Missing standard type headers)
        if type_code not in [CMD_TYPE, LIT_TYPE, RLE_TYPE]:
            return TokenInfo(t, "SEED", 0, False, False, 0)

        # 3. Identify RLE vs CMD using Bit 15
        is_rle = (type_code == RLE_TYPE) or (type_code == CMD_TYPE and ((payload >> 15) & 0x1 == 1))

        # 4. Calculate the required cycles
        if is_rle:
            rle_length = (payload >> 11) & 0xF
            cycles = 2 if rle_length > 7 else 1
            period_class = (payload >> 8) & 0x7
            req_seed = (period_class == 7)
            return TokenInfo(t, "RLE", 0, False, req_seed, cycles)
            
        else: # Standard LC Command
            lit_field = (payload >> 11) & 0xF
            copy_len = (payload >> 7) & 0xF
            is_barrier = (lit_field == 0 and copy_len == 15)
            
            cycles = 1
            if lit_field >= 4:
                cycles = (lit_field // 4) + (1 if (lit_field % 4) else 0)
                
            return TokenInfo(t, "CMD", int(lit_field), is_barrier, False, cycles)
    
    def can_accept(self, token_info: TokenInfo) -> bool:
        """Evaluates if the appropriate FIFO has space for the decoded token."""
        if token_info.category == "LIT":
            return len(self.LIT_FIFO) < self.LIT_FIFO.maxlen
        else:
            return len(self.CMD_FIFO) < self.CMD_FIFO.maxlen
    
    def receive_token(self, token_info: TokenInfo):
        """Pushes the decoded token object directly into the hardware queues."""
        if token_info.category == "LIT":
            self.LIT_FIFO.append(token_info)
        else:
            self.CMD_FIFO.append(token_info)

    def tick(self):
        """Advances the clock cycle by 1 for the execution unit."""
        self.is_lit_starved = False
        
        # Process active task
        if self.cycles_left > 0:
            self.cycles_left -= 1
            self.active_cycles += 1
            return

        # Attempt to start new task
        if self.cycles_left == 0 and self.CMD_FIFO:
            next_tok = self.CMD_FIFO[0] 
            
            # Dependency Stalls
            if next_tok.requires_seed and len(self.CMD_FIFO) < 2:
                return # Stall waiting for bus to deliver seed
                
            if next_tok.category == "CMD" and len(self.LIT_FIFO) < next_tok.req_lit_chunks:
                self.is_lit_starved = True
                return # Stall waiting for bus to deliver literals
                
            if next_tok.is_barrier and len(self.OUTPUT_FIFO) == self.OUTPUT_FIFO.maxlen:
                return # Stall waiting for the output buffer to drain
                
            # Safe Execution
            executed_tok = self.CMD_FIFO.popleft()
            
            if executed_tok.is_barrier:
                self.OUTPUT_FIFO.append(1) # Push a completed row into the buffer
                self.cycles_left = executed_tok.cycles
                self.active_cycles += 1
                return

            if executed_tok.requires_seed:
                self.CMD_FIFO.popleft() # Swallow the seed silently
                
            for _ in range(executed_tok.req_lit_chunks):
                self.LIT_FIFO.popleft() # Swallow the literals silently
                
            self.cycles_left = executed_tok.cycles
            self.active_cycles += 1

class DecoderEnvV3(gym.Env):
    def __init__(self, data_dir: str, train_split_pct: float = 1.0, is_eval: bool = False, seed: int = 42, is_inference: bool = False):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f"data_dir must be a directory: {data_dir}")
            
        if is_inference:
            # Single dataset mode for deployment
            hex_dir = self.data_dir / "beats_hex"
            if not hex_dir.exists() or not hex_dir.is_dir():
                # Fallback if they pointed directly to the beats_hex folder
                if self.data_dir.name == "beats_hex":
                    hex_dir = self.data_dir
                else:
                    raise ValueError(f"Inference mode requires a 'beats_hex' folder in {self.data_dir}")
            
            cmds = self._load_hex_folder(hex_dir)
            self.dataset_pool = [cmds]
            print(f"Loaded single dataset for inference from: {self.data_dir}")
        else:
            # Discover all subfolders that contain a 'beats_hex' folder
            all_datasets = []
            for test_folder in sorted(self.data_dir.iterdir()):
                if test_folder.is_dir():
                    hex_dir = test_folder / "beats_hex"
                    if hex_dir.exists() and hex_dir.is_dir():
                        try:
                            cmds = self._load_hex_folder(hex_dir)
                            all_datasets.append(cmds)
                        except Exception as e:
                            print(f"Skipping {test_folder.name} due to error: {e}")
                            
            if not all_datasets:
                raise ValueError(f"No valid datasets found in {data_dir}")
                
            # Deterministically shuffle to create consistent train/eval splits
            rng = random.Random(seed)
            rng.shuffle(all_datasets)
            
            # Split datasets
            split_idx = max(1, int(len(all_datasets) * train_split_pct))
            
            if is_eval:
                self.dataset_pool = all_datasets[split_idx:]
                if not self.dataset_pool:
                    print("Warning: train_split_pct too high, using all datasets for eval.")
                    self.dataset_pool = all_datasets
            else:
                self.dataset_pool = all_datasets[:split_idx]
                
            print(f"Loaded {len(self.dataset_pool)} datasets for {'evaluation' if is_eval else 'training'}.")
        
        self.decoders = []
        # --- Dynamically calculate n_actual based on the datasets ---
        max_n_actual = 0
        for dataset in self.dataset_pool:
            n_min = 0
            for i, cmds in enumerate(dataset):
                temp_dec = SubDecoder(is_bypass=(i == 8))
                slots = 0
                for raw_cmd in cmds:
                    tok = temp_dec.decode_token(raw_cmd)
                    if tok.category == "RLE": slots += 2
                    elif tok.category == "CMD": slots += (1 + tok.req_lit_chunks)
                    else: slots += 1 # LIT or SEED
                n_min += math.ceil(slots / 8.0)
            
            n_actual = math.ceil(1.5 * n_min)
            if n_actual > max_n_actual:
                max_n_actual = n_actual
                
        self.n_actual = max_n_actual
        print(f"Calculated Max Action Chunk Size (n_actual) with 50% headroom: {self.n_actual}")

        # --- Curriculum State ---
        self.chunk_size = 1
        
        # Action space is now a MultiDiscrete array of length n_actual
        self.action_space = gym.spaces.MultiDiscrete([NUM_DECODERS] * self.n_actual)
        
        self.decoders = [SubDecoder(is_bypass=(i == 8)) for i in range(NUM_DECODERS)]
        self.num_cycles = 0
        self.stall_cycles = 0
        
        self.observation_space = gym.spaces.Dict({
            "cycles_left": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
            "cmd_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
            "lit_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
            "output_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
            "is_stalled_by_output": gym.spaces.MultiBinary(NUM_DECODERS),
            "is_done": gym.spaces.MultiBinary(NUM_DECODERS),
            "is_lit_starved": gym.spaces.MultiBinary(NUM_DECODERS),
            "lookahead_types": gym.spaces.Box(low=-1, high=3, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.int32),
            "lookahead_req_lits": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.float32),
            "lookahead_cycles": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.float32),
            "lookahead_is_barrier": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.float32),
        })

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

    def action_masks(self):
        """
        Dynamically calculates which actions (subdecoders) are legal to pick right now.
        Returns a boolean array of length NUM_DECODERS (1 = legal, 0 = illegal).
        """
        masks = np.zeros(NUM_DECODERS, dtype=np.int8)
        for i, sub_d in enumerate(self.decoders):
            # If there are no commands left to send on this bus lane, it's an illegal action
            if not self.cmds[i]:
                continue
                
            # Peek at the next token to be transferred
            raw_cmd = self.cmds[i][0]
            token_info = sub_d.decode_token(raw_cmd)
            
            # Action is only legal if the target FIFO actually has space for it
            if sub_d.can_accept(token_info):
                masks[i] = 1
                
        # Fallback: If ALL paths are physically blocked, we must return at least one '1' 
        # to prevent SB3 from crashing. The env will catch the stall and terminate anyway.
        if not masks.any():
            masks.fill(1)
            
        return masks
    
    def increase_curriculum(self):
        """Gradually scales up the number of decisions the agent makes per step"""
        if self.chunk_size < self.n_actual:
            self.chunk_size += 1

    def step_sub_decoders(self):
        for sub_d in self.decoders:
            sub_d.tick()

    def step(self, action_array):
        """Action is now an array of length n_actual. We execute up to chunk_size."""
        total_reward = 0
        terminated = False
        truncated = False
        info = {"tokens_pushed": 0}

        for step_idx in range(self.chunk_size):
            target_sub_d = action_array[step_idx]
            self.num_cycles += 1
            tokens_pushed = 0

            # --- Bus Transfer Phase ---
            if self.cmds[target_sub_d]:
                i = 0
                subdecoder = self.decoders[target_sub_d]
                while i < subdecoder.bus_transfer_limit and self.cmds[target_sub_d]:
                    raw_cmd = self.cmds[target_sub_d][0]
                    token_info = subdecoder.decode_token(raw_cmd)
                    
                    if subdecoder.can_accept(token_info):
                        self.cmds[target_sub_d].popleft()
                        subdecoder.receive_token(token_info)
                        i += 1
                        tokens_pushed += 1
                    else: 
                        break 

            # --- Micro-Penalty for Wasting the Bus ---
            if tokens_pushed == 0 and any(len(q) > 0 for q in self.cmds):
                self.stall_cycles += 1
                total_reward -= 2.0 
            else:
                self.stall_cycles = 0

            # --- Execution Phase ---
            self.step_sub_decoders()

            # --- Dynamic Output Buffer Synchronization Phase ---
            active_decoders = []
            for idx, sub_d in enumerate(self.decoders):
                is_permanently_finished = (
                    len(self.cmds[idx]) == 0 and 
                    len(sub_d.CMD_FIFO) == 0 and 
                    sub_d.cycles_left == 0 and
                    len(sub_d.OUTPUT_FIFO) == 0
                )
                if not is_permanently_finished:
                    active_decoders.append(sub_d)
            
            if active_decoders and all(len(sub_d.OUTPUT_FIFO) > 0 for sub_d in active_decoders):
                for sub_d in active_decoders:
                    sub_d.OUTPUT_FIFO.popleft()
                # Positive reward removed for PPO stability

            # --- Proportional Deadlock Penalty ---
            if self.stall_cycles > 500:
                terminated = True
                total_reward -= 1000.0 # Bounded to prevent exploding gradients
                info["deadlock"] = True
                break # Break out of chunk loop

            # --- Reward Shaping Phase (Penalties Only) ---
            row_counts = [len(sub_d.OUTPUT_FIFO) for sub_d in self.decoders]
            row_spread = max(row_counts) - min(row_counts)
            
            imbalance_penalty = 2.0 if row_spread > 2 else 0.0
                    
            hoard_penalty = 0.0
            for sub_d in self.decoders:
                if not sub_d.is_bypass: 
                    cmd_util = len(sub_d.CMD_FIFO) / CMD_FIFO_SIZE
                    if cmd_util > FIFO_HOARD_THRESH:
                        hoard_penalty += 0.5
                
            total_reward -= (1.0 + imbalance_penalty + hoard_penalty)
            info["tokens_pushed"] += tokens_pushed
            
            # --- Termination Phase ---
            if all(len(q) == 0 for q in self.cmds) and all(sub_d.cycles_left == 0 for sub_d in self.decoders) and all(len(sub_d.OUTPUT_FIFO) == 0 for sub_d in self.decoders):
                terminated = True
                info["makespan"] = self.num_cycles
                info["decoder_times"] = [sub_d.active_cycles for sub_d in self.decoders]
                break # Sequence is complete, stop executing chunk array

        return self._get_obs(), total_reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly select one dataset from the pool for this episode
        selected_idx = self.np_random.integers(0, len(self.dataset_pool))
        selected_dataset = self.dataset_pool[selected_idx]

        self.cmds = [deque(list(queue)) for queue in selected_dataset]
        self.decoders = [SubDecoder(is_bypass=(i == 8)) for i in range(NUM_DECODERS)]
        self.num_cycles = 0
        self.stall_cycles = 0
        self.initial_total_tokens = sum(len(cmd) for cmd in self.cmds)

        return self._get_obs(), {}

    def _get_obs(self):
        obs = {
            "cycles_left": np.zeros(NUM_DECODERS, dtype=np.float32),
            "cmd_fifo_fill": np.zeros(NUM_DECODERS, dtype=np.float32),
            "lit_fifo_fill": np.zeros(NUM_DECODERS, dtype=np.float32),
            "output_fifo_fill": np.zeros(NUM_DECODERS, dtype=np.float32),
            "is_stalled_by_output": np.zeros(NUM_DECODERS, dtype=np.int8),
            "is_done": np.zeros(NUM_DECODERS, dtype=np.int8),
            "is_lit_starved": np.zeros(NUM_DECODERS, dtype=np.int8),
            "lookahead_types": np.full((NUM_DECODERS, LOOKAHEAD_WINDOW), -1, dtype=np.int32), 
            "lookahead_cycles": np.zeros((NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.float32),
            "lookahead_req_lits": np.zeros((NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.float32),
            "lookahead_is_barrier": np.zeros((NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.float32)
        }
        
        type_mapping = {"CMD": 0, "LIT": 1, "RLE": 2, "SEED": 3}

        for i, sub_d in enumerate(self.decoders):
            obs["cycles_left"][i] = min(sub_d.cycles_left / MAX_CYCLES_LEFT, 1.0) 
            
            # Use dynamic maxlen for observation normalization
            obs["cmd_fifo_fill"][i] = len(sub_d.CMD_FIFO) / sub_d.CMD_FIFO.maxlen
            obs["lit_fifo_fill"][i] = len(sub_d.LIT_FIFO) / LIT_FIFO_SIZE
            obs["output_fifo_fill"][i] = len(sub_d.OUTPUT_FIFO) / sub_d.OUTPUT_FIFO.maxlen

            obs["is_stalled_by_output"][i] = int(len(sub_d.OUTPUT_FIFO) == sub_d.OUTPUT_FIFO.maxlen)
            obs["is_done"][i] = int(len(self.cmds[i]) == 0 and len(sub_d.CMD_FIFO) == 0 and sub_d.cycles_left == 0 and len(sub_d.OUTPUT_FIFO) == 0)
            obs["is_lit_starved"][i] = int(sub_d.is_lit_starved)

            # Populate Lookahead Buffer
            bus_queue = self.cmds[i]
            window_size = min(len(bus_queue), LOOKAHEAD_WINDOW)
            
            for j in range(window_size):
                raw_token = bus_queue[j]
                token_info = sub_d.decode_token(raw_token) 
                
                obs["lookahead_types"][i, j] = type_mapping.get(token_info.category, -1)
                obs["lookahead_cycles"][i, j] = min(token_info.cycles / MAX_CYCLES_LEFT, 1.0)
                obs["lookahead_req_lits"][i, j] = token_info.req_lit_chunks / 15.0
                obs["lookahead_is_barrier"][i, j] = float(token_info.is_barrier)

        total_remaining = sum(len(q) for q in self.cmds)
        obs["global_progress"] = np.array([total_remaining / max(1, self.initial_total_tokens)], dtype=np.float32)

        return obs