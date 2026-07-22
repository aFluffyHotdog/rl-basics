"""
New version of a python environment that simulates the T8 decompressor

now implements:
    - the TDM mode (135 bits gets sent to a subdecoder at a time)
    - actual stalling if a subdecoder is full
    - FIFO buffer behavior
    - cycle modelling based on CMD type and size
    - Clean pipeline architecture (Decode -> Route -> Execute)
    - 4-Row Output Buffer with Dynamic Backpressure
    - Throughput & Deadzone Imbalance Reward Shaping
"""
from pathlib import Path
from collections import deque
from dataclasses import dataclass
import random

import gymnasium as gym
import numpy as np

LIT_FIFO_SIZE = 72 # 24/24/18/12
CMD_FIFO_SIZE = 16
BEATS_ON_BUS = 8
NUM_DECODERS = 8
LOOKAHEAD_WINDOW = 8
OUTPUT_ROW_BUFFER_SIZE = 4 # How many rows ahead a subdecoder can get before stalling

CMD_TYPE=0
LIT_TYPE=1
RLE_TYPE=2
MAX_CYCLES_LEFT = 16 # normalization constant for observation
DEADLOCK_THRESHOLD = 500

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
    def __init__(self):
        # NOTE: 1 chunk = 16 bit
        # 24/24/18/12 * 4 chunk LIT FIFO buffers
        self.LIT_FIFO = deque(maxlen=LIT_FIFO_SIZE)
        # 16 chunk CMD FIFO buffer
        self.CMD_FIFO = deque(maxlen=CMD_FIFO_SIZE)
        # 4-row output FIFO
        self.OUTPUT_FIFO = deque(maxlen=OUTPUT_ROW_BUFFER_SIZE)
        
        # Execution State
        self.cycles_left = 0
        self.active_cycles = 0

    def debug_dump_buffers(self):
        cmd_types = [tok.category for tok in self.CMD_FIFO]
        lit_preview = [tok.raw_val & 0xFFFF for tok in list(self.LIT_FIFO)[:8]]
        return (
            f"CMD_FIFO_count={len(self.CMD_FIFO)}, CMD_FIFO_types={cmd_types}, "
            f"LIT_FIFO_count={len(self.LIT_FIFO)}, LIT_preview={lit_preview}, "
            f"OUTPUT_FIFO_count={len(self.OUTPUT_FIFO)}, "
            f"cycles_left={self.cycles_left}"
        )

    def debug_dump_state(self):
        state = {}
        for name, value in self.__dict__.items():
            if isinstance(value, deque):
                state[name] = {
                    "count": len(value),
                    "items": [t.raw_val for t in value],
                    "maxlen": value.maxlen,
                }
            else:
                state[name] = value
        return state

    def decode_token(self, token: int) -> TokenInfo:
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
        # The bus only cares if the input FIFOs are full. Output stalls happen downstream.
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


class DecoderEnvV2(gym.Env):
    def __init__(self, data_dir: str, train_split_pct: float = 0.8, is_eval: bool = False, seed: int = 42):
        self.data_dir = Path(data_dir)
        if not self.data_dir.is_dir():
            raise ValueError(f"data_dir must be a directory: {data_dir}")
            
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
            # Fallback if split leaves eval empty
            if not self.dataset_pool:
                print("Warning: train_split_pct too high, using all datasets for eval.")
                self.dataset_pool = all_datasets
        else:
            self.dataset_pool = all_datasets[:split_idx]
            
        print(f"Loaded {len(self.dataset_pool)} datasets for {'evaluation' if is_eval else 'training'}.")
        
        self.decoders = []
        for i in range(NUM_DECODERS):
            self.decoders.append(SubDecoder())
            
        self.num_cycles = 0
        self.stall_cycles = 0
        self.action_space = gym.spaces.Discrete(NUM_DECODERS)
        
        self.observation_space = gym.spaces.Dict({
            "cycles_left": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
            "cmd_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
            "lit_fifo_fill": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS,), dtype=np.float32),
            "is_stalled_by_output": gym.spaces.MultiBinary(NUM_DECODERS),
            "is_done": gym.spaces.MultiBinary(NUM_DECODERS),
            "is_lit_starved": gym.spaces.MultiBinary(NUM_DECODERS),
            "lookahead_types": gym.spaces.Box(low=-1, high=3, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.int32),
            "lookahead_cycles": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS, LOOKAHEAD_WINDOW), dtype=np.float32),
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
    
    def step_sub_decoders(self):
        for sub_d in self.decoders:
            sub_d.tick()

    def step(self, action):
        observation = None
        reward = 0
        terminated = False
        truncated = False
        info = {}

        target_sub_d = action
        self.num_cycles += 1
        tokens_pushed = 0

        # --- Bus Transfer Phase ---
        if self.cmds[target_sub_d]:
            i = 0
            while i < BEATS_ON_BUS and self.cmds[target_sub_d]:
                raw_cmd = self.cmds[target_sub_d][0]
                subdecoder = self.decoders[target_sub_d]
                
                # Decode once at the boundary
                token_info = subdecoder.decode_token(raw_cmd)
                
                if subdecoder.can_accept(token_info):
                    self.cmds[target_sub_d].popleft()
                    subdecoder.receive_token(token_info)
                    i += 1
                    tokens_pushed += 1
                else: 
                    # Hardware block (FIFO Full or Output Buffer Full). Yield the bus.
                    break 

        # Track stalls: if the bus pushed nothing but there is still work to do
        if tokens_pushed == 0 and any(len(q) > 0 for q in self.cmds):
            self.stall_cycles += 1
        else:
            self.stall_cycles = 0

        # --- Deadlock Detection & Penalty ---
        if self.stall_cycles > DEADLOCK_THRESHOLD:
            terminated = True
            reward -= 50000.0 
            info["deadlock"] = True
            observation = self._get_obs()
            return observation, reward, terminated, truncated, info

        # --- Execution Phase ---
        self.step_sub_decoders()

        # --- Dynamic Output Buffer Synchronization Phase ---
        # Hardware row buffers drain when all active subdecoders have pushed 
        # at least one completed row into their output FIFOs.
        active_decoders = []
        for idx, sub_d in enumerate(self.decoders):
            # A subdecoder is truly finished only if its output is also completely flushed
            is_permanently_finished = (
                len(self.cmds[idx]) == 0 and 
                len(sub_d.CMD_FIFO) == 0 and 
                sub_d.cycles_left == 0 and
                len(sub_d.OUTPUT_FIFO) == 0
            )
            if not is_permanently_finished:
                active_decoders.append(sub_d)
        
        # Flush 1 row from all active decoders if they all have at least 1 completed row ready
        if active_decoders and all(len(sub_d.OUTPUT_FIFO) > 0 for sub_d in active_decoders):
            for sub_d in active_decoders:
                sub_d.OUTPUT_FIFO.popleft()

        # --- Reward Shaping Phase ---
        # 1. Throughput calculation (+1.0 per token moved, -1.0 per cycle)
        # 2. Deadzone Imbalance penalty (Safe margin = 2 rows. Exceeding costs points.)
        row_counts = [len(sub_d.OUTPUT_FIFO) for sub_d in self.decoders]
        row_spread = max(row_counts) - min(row_counts)
        
        SAFE_MARGIN = 2
        imbalance_penalty = 0.0
        if row_spread > SAFE_MARGIN:
            excess_spread = row_spread - SAFE_MARGIN
            imbalance_penalty = (excess_spread ** 2) * 0.5
            
        reward = (tokens_pushed * 1.0) - imbalance_penalty - 1.0
        
        # --- Termination Phase ---
        if all(len(q) == 0 for q in self.cmds) and all(sub_d.cycles_left == 0 for sub_d in self.decoders) and all(len(sub_d.OUTPUT_FIFO) == 0 for sub_d in self.decoders):
            terminated = True
            
            # Pass final diagnostic metrics to the PPO Monitor callback
            info["makespan"] = self.num_cycles
            info["decoder_times"] = [sub_d.active_cycles for sub_d in self.decoders]

        observation = self._get_obs()
        return observation, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Randomly select one dataset from the pool for this episode
        selected_idx = self.np_random.integers(0, len(self.dataset_pool))
        selected_dataset = self.dataset_pool[selected_idx]

        self.cmds = [deque(list(queue)) for queue in selected_dataset]
        self.decoders = [SubDecoder() for _ in range(NUM_DECODERS)]
        self.num_cycles = 0
        self.stall_cycles = 0

        observation = self._get_obs()
        info = {}
        return observation, info

    def _get_obs(self):
        obs = {
            "cycles_left": np.zeros(NUM_DECODERS, dtype=np.float32),
            "cmd_fifo_fill": np.zeros(NUM_DECODERS, dtype=np.float32),
            "lit_fifo_fill": np.zeros(NUM_DECODERS, dtype=np.float32),
            "is_stalled_by_output": np.zeros(NUM_DECODERS, dtype=np.int8),
            "is_done": np.zeros(NUM_DECODERS, dtype=np.int8),
            "is_lit_starved": np.zeros(NUM_DECODERS, dtype=np.int8),
            "lookahead_types": np.full((NUM_DECODERS, 8), -1, dtype=np.int32), # Pad with -1
            "lookahead_cycles": np.zeros((NUM_DECODERS, 8), dtype=np.float32)
        }
        
        type_mapping = {"CMD": 0, "LIT": 1, "RLE": 2, "SEED": 3}

        for i, sub_d in enumerate(self.decoders):
            # 1. Fill basic status & normalize
            obs["cycles_left"][i] = min(sub_d.cycles_left / MAX_CYCLES_LEFT, 1.0) 
            obs["cmd_fifo_fill"][i] = len(sub_d.CMD_FIFO) / CMD_FIFO_SIZE
            obs["lit_fifo_fill"][i] = len(sub_d.LIT_FIFO) / LIT_FIFO_SIZE
            obs["is_stalled_by_output"][i] = int(len(sub_d.OUTPUT_FIFO) == sub_d.OUTPUT_FIFO.maxlen)
            obs["is_done"][i] = int(len(self.cmds[i]) == 0 and len(sub_d.CMD_FIFO) == 0 and sub_d.cycles_left == 0 and len(sub_d.OUTPUT_FIFO) == 0)

            # 2. Check Literal Starvation
            if len(sub_d.CMD_FIFO) > 0:
                head_tok = sub_d.CMD_FIFO[0]
                if head_tok.category == "CMD" and len(sub_d.LIT_FIFO) < head_tok.req_lit_chunks:
                    obs["is_lit_starved"][i] = 1

            # 3. Populate Lookahead Buffer
            bus_queue = self.cmds[i]
            window_size = min(len(bus_queue), 8)
            
            for j in range(window_size):
                raw_token = bus_queue[j]
                token_info = sub_d.decode_token(raw_token) # Peek and decode
                
                obs["lookahead_types"][i, j] = type_mapping.get(token_info.category, -1)
                obs["lookahead_cycles"][i, j] = min(token_info.cycles / MAX_CYCLES_LEFT, 1.0)
                
        return obs