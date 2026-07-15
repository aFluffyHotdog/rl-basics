"""
New version of a python environment that simulates the T8 decompressor

now implements:
    - the TDM mode (135 bits gets sent to a subdecoder at a time)
    - actual stalling if a subdecoder is full
    - FIFO buffer behavior
    - cycle modelling based on CMD type and size
    - Clean pipeline architecture (Decode -> Route -> Execute)
    - Global row barrier synchronization
"""
from pathlib import Path
from collections import deque
from dataclasses import dataclass

import gymnasium as gym

LIT_FIFO_SIZE = 72 # 24/24/18/12
CMD_FIFO_SIZE = 16
BEATS_ON_BUS = 8
NUM_DECODERS = 8
CMD_TYPE=0
LIT_TYPE=1
RLE_TYPE=2

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
        
        # Execution State
        self.cycles_left = 0
        self.active_cycles = 0
        self.is_waiting_at_barrier = False

    def debug_dump_buffers(self):
        cmd_types = [tok.category for tok in self.CMD_FIFO]
        lit_preview = [tok.raw_val & 0xFFFF for tok in list(self.LIT_FIFO)[:8]]
        return (
            f"CMD_FIFO_count={len(self.CMD_FIFO)}, CMD_FIFO_types={cmd_types}, "
            f"LIT_FIFO_count={len(self.LIT_FIFO)}, LIT_preview={lit_preview}, "
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
        if self.is_waiting_at_barrier:
            return False
            
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
            
        # Halt at barrier
        if self.is_waiting_at_barrier:
            return

        # Attempt to start new task
        if self.cycles_left == 0 and self.CMD_FIFO:
            next_tok = self.CMD_FIFO[0] 
            
            # Dependency Stalls
            if next_tok.requires_seed and len(self.CMD_FIFO) < 2:
                return # Stall waiting for bus to deliver seed
                
            if next_tok.category == "CMD" and len(self.LIT_FIFO) < next_tok.req_lit_chunks:
                return # Stall waiting for bus to deliver literals
                
            # Safe Execution
            executed_tok = self.CMD_FIFO.popleft()
            
            if executed_tok.is_barrier:
                self.is_waiting_at_barrier = True
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
    def __init__(self, cmd_path: str):
        self.cmds = self._load_hex_folder(cmd_path)
        
        self.decoders = []
        for i in range(NUM_DECODERS):
            self.decoders.append(SubDecoder())
            
        self.num_cycles = 0
        self.action_space = gym.spaces.Discrete(NUM_DECODERS)
        self.observation_space = {}

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

        # Bus Transfer Phase 
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
                else: 
                    # Hardware block (FIFO Full or Barrier). Yield the bus.
                    break 

        # --- Execution Phase ---
        self.step_sub_decoders()

        # --- Barrier Synchronization Phase ---
        all_at_barrier = True
        for idx, sub_d in enumerate(self.decoders):
            is_waiting = getattr(sub_d, 'is_waiting_at_barrier', False)
            
            # A subdecoder is only truly done if the main bitstream is ALSO empty
            is_permanently_finished = (
                len(self.cmds[idx]) == 0 and 
                len(sub_d.CMD_FIFO) == 0 and 
                sub_d.cycles_left == 0
            )
            
            if not (is_waiting or is_permanently_finished):
                all_at_barrier = False
                break
                
        if all_at_barrier:
            for sub_d in self.decoders:
                sub_d.is_waiting_at_barrier = False
        
        # --- Termination Phase ---
        if all(len(q) == 0 for q in self.cmds) and all(sub_d.cycles_left == 0 for sub_d in self.decoders):
            terminated = True

        return observation, reward, terminated, truncated, info