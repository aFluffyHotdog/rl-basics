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

DEADLOCK_STALL_CYCLES = 500
CMD_TYPE = 0
LIT_TYPE = 1
RLE_TYPE = 2

GAMMA = 1.0

# Potential-based reward shaping: Phi(s) = -(LB(s) + C_LAG * lag_excess(s)^2), in cycles
LAG_MARGIN = 2        # rows of output spread tolerated before the lag term kicks in
C_LAG = 4.0           # cycles of "virtual lost time" per squared excess row
FAIL_PENALTY = 2.0    # terminal penalty (in lb0 units) for mismatch / deadlock
DEBUG_CHECKS = True   # assert LB / CMD-accounting invariants every step; switch off for long runs

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
