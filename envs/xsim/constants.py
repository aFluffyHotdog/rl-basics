"""Hardware, protocol and reward-shaping constants for the XSim decoder environment."""

VIVADO_VERBOSE = False

# Hardware Constants
NUM_DECODERS = 8      # 0-7 are LC/RLE
NO_OP_ACTION = 8      # Action index to yield the bus (Valid=0)
NUM_ACTIONS = NUM_DECODERS + 1
BEATS_ON_BUS = 8      # Max 16-bit slots per beat
MAX_CYCLES_LEFT = 16
LOOKAHEAD_WINDOW = 64

# FIFO capacities used to normalize XSim-reported occupancy counts.
CMD_FIFO_SIZE = 16
LIT_FIFO_CAPACITIES = (24, 24, 18, 12)
OUT_FIFO_SIZE = 4
# Combined depth of every per-lane FIFO (CMD + 4 literal banks + output row queue), used to
# normalize the aggregate occupancy feature in the observation.
TOTAL_FIFO_CAPACITY = CMD_FIFO_SIZE + sum(LIT_FIFO_CAPACITIES) + OUT_FIFO_SIZE
# Output rows per lane per frame (TB NUM_ROWS) and decoded bytes per row.
NUM_ROWS = 720
ROW_BYTES = 16

# Decode-work scale for the beat1/beat2 work features (a full 8-slot beat of ~2-tick groups).
BEAT_WORK_SCALE = BEATS_ON_BUS * 2

DEADLOCK_STALL_CYCLES = 500

# v20 testbench packer (blk_packer_v17), used by the round-robin teacher and the opt-in mask gate.
TB_INFLIGHT_GATE = 8                  # TB only feeds a lane with <= 8 CMDs in flight (committed - popped)
TB_RR_CHANNELS = NUM_DECODERS + 1     # TB cursor runs over the 8 lanes + the (disabled) bypass channel
# Observation v4: cycles since a lane last received a beat, divided by this and clipped to 1.
CYCLES_SINCE_FED_SCALE = 64
CMD_TYPE = 0
LIT_TYPE = 1
RLE_TYPE = 2

GAMMA = 1.0

# Potential-based reward shaping: Phi(s) = -(LB(s) + C_LAG * lag_excess(s)^2), in cycles
LAG_MARGIN = 2        # rows of output spread tolerated before the lag term kicks in
C_LAG = 4.0           # cycles of "virtual lost time" per squared excess row
FAIL_PENALTY = 2.0    # terminal penalty (in lb0 units) for mismatch / deadlock

# Reward v3 (reward_v3.py): r = LAMBDA_THROUGHPUT * throughput - LAMBDA_STDEV * sigma
#   throughput = complete rows produced over the last THROUGHPUT_WINDOW cycles (sum of the window),
#   sigma      = population stdev of decoded rows across the NUM_DECODERS lanes.
# Maximize throughput, minimize per-lane output spread. Both lambdas are placeholders to tune.
THROUGHPUT_WINDOW = 20          # cycles of history summed into the windowed throughput signal
THROUGHPUT_SCALE = THROUGHPUT_WINDOW * NUM_DECODERS   # max rows a window can close (1 row/lane/cycle)
LAMBDA_THROUGHPUT = 1.0         # reward per complete row produced in the window
LAMBDA_STDEV = 0.1              # penalty per row of output stdev across lanes
# Reward v4 (reward_v4.py): r = -cycles_elapsed / REWARD_SCALE + REWARD_ALPHA * rows_done_delta (per env step)
# With REWARD_ALPHA = 0 an episode's summed reward is exactly -makespan / REWARD_SCALE.
REWARD_SCALE = 1000             # fixed divisor that puts the per-step cycle cost near O(1e-3..1e-2)
REWARD_ALPHA = 0.0              # reward per complete row output since the previous env step
# Terminal failure penalty (in lb0 units) for mismatch / deadlock; still charged per cycle.
LAMBDA_T = 1
DEBUG_CHECKS = False   # assert LB / CMD-accounting invariants every step; switch off for long runs

# Per-subdecoder FIFO occupancy fields reported in each @OBS line, in order.
SUB_DECODER_FIELDS = ("cmd", "lit1", "lit2", "lit3", "lit4", "out")
LIT_FIELDS = ("lit1", "lit2", "lit3", "lit4")

# Original @OBS fields: top_in_ready | top_row_valid | mismatches | 8 x SUB_DECODER_FIELDS (= 51)
BASE_OBS_LEN = 3 + NUM_DECODERS * len(SUB_DECODER_FIELDS)
# Extended @OBS fields appended by tb_interactive.sv (after the original 51):
#   51..58 tail_busy[8] | 59..66 cmd_popped[8] | 67 rows_out | 68 rows_consumed | 69 lane_mismatches
#   70 proto_err | 71 tailbusy_hits | 72 cmd_overcommit | 73 ready_viol
EXT_OBS_LEN = 74
EXT_OBS_KEYS = ("tail_busy", "cmd_popped", "rows_out", "rows_consumed", "lane_mismatches",
                "proto_err", "tailbusy_hits", "cmd_overcommit", "ready_viol")
# Testbench counters that must stay at 0 now that the mask enforces the same gates.
# An increase means the mask / packer disagrees with the RTL -> the episode is failed.
TB_VIOLATION_KEYS = ("proto_err", "cmd_overcommit", "ready_viol", "tailbusy_hits")
# Per-lane golden mismatches are opt-in: the old testbench flagged free-run (barrier off) lane
# mismatches as a measurement artifact, so they are only logged unless you turn this on.
FAIL_ON_LANE_MISMATCH = False

NO_OP_REASONS = ("top_in_not_ready", "no_valid_subdecoder_actions", "chosen_while_actions_valid")
