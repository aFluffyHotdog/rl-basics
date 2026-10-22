"""Compact per-lane observation (v2): NUM_DECODERS x 14 lane features + 5 globals (117 numbers).

Replaces the ~3,300-dim token-level lookahead of v1 with the quantities that actually set the
makespan: remaining decode work vs. remaining bus beats per lane, what is already buffered, and
what the next beats would carry.

Normalisation (every feature lies in [0, 1]):
  - Episode-scale cycle counts (work_left, work_gap, unsent_work, beats_left and the matching
    globals) are divided by lb0, the episode's initial makespan lower bound. lb0 >= every lane's
    total work and >= the total dense beats, so these never exceed 1, and since one beat takes one
    bus cycle, work and beats share a unit and can be compared directly. This also makes the
    features independent of image size.
  - Short-horizon cycle counts (runway, beat work) would sit near 0 at the lb0 scale, so they are
    divided by hardware bounds instead (RUNWAY_SCALE, BEAT_WORK_SCALE below).
  - Occupancies are divided by FIFO capacity, slots by BEATS_ON_BUS, counts by NUM_DECODERS;
    flags are already 0/1.
Values are clipped to [0, 1] as a guard against testbench counts outside the expected range.
"""
import gymnasium as gym
import numpy as np

from .constants import BEATS_ON_BUS, CMD_FIFO_SIZE, LIT_FIFO_CAPACITIES, NUM_DECODERS
from .flow_control import cmd_inflight, plan_beat
from .reward import LowerBoundTracker

LANE_FEATURES = (
    "work_left",     # undecoded work (groups not yet popped by the RTL) / lb0
    "work_gap",      # (max lane work_left - this lane's work_left) / lb0
    "unsent_work",   # work of groups not yet sent / lb0
    "beats_left",    # dense beats still needed for this lane / lb0
    "inflight",      # CMDs sent but not popped / CMD FIFO depth
    "cmd_fifo",      # CMD FIFO occupancy / depth
    "runway",        # buffered decode work (sent, not popped) / RUNWAY_SCALE
    "tail_busy",     # arbiter tail busy flag
    "eligible",      # action mask bit
    "beat1_slots",   # slots the beat sent now would carry (0 if not eligible) / BEATS_ON_BUS
    "beat1_work",    # decode work in that beat / BEAT_WORK_SCALE
    "beat2_slots",   # slots of the following dense beat / BEATS_ON_BUS
    "beat2_work",    # decode work in that beat / BEAT_WORK_SCALE
    "lit4_fill",     # literal bank 4 occupancy / capacity (the smallest bank)
)
GLOBAL_FEATURES = (
    "in_ready",      # arbiter input ready
    "bus_left",      # total dense beats left / lb0
    "work_left_max", # bottleneck lane's work_left / lb0
    "lb_left",       # current makespan lower bound / lb0
    "n_eligible",    # eligible lanes / NUM_DECODERS
)

# Largest group cost: a CMD with lit_field=15 takes ceil(15/4)=4 cycles + 1 start tick.
MAX_GROUP_TICKS = 5
# The mask keeps at most CMD_FIFO_SIZE groups in flight per lane.
RUNWAY_SCALE = CMD_FIFO_SIZE * MAX_GROUP_TICKS
# Densest work per beat: 8 one-slot CMDs (lit_field=0) at 2 ticks each.
BEAT_WORK_SCALE = BEATS_ON_BUS * 2

_F = {name: k for k, name in enumerate(LANE_FEATURES)}
_G = {name: k for k, name in enumerate(GLOBAL_FEATURES)}


def build_observation_space_v2() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "lanes": gym.spaces.Box(low=0.0, high=1.0, shape=(NUM_DECODERS, len(LANE_FEATURES)), dtype=np.float32),
        "globals": gym.spaces.Box(low=0.0, high=1.0, shape=(len(GLOBAL_FEATURES),), dtype=np.float32),
    })


def build_observation_v2(hw_state: dict, cmds: list, lb: LowerBoundTracker, lb0: int,
                         mask: np.ndarray) -> dict:
    """Call after lb.compute() so lb.consumed and lb.lb_parts reflect the current hw_state."""
    lanes = np.zeros((NUM_DECODERS, len(LANE_FEATURES)), dtype=np.float32)
    glob = np.zeros(len(GLOBAL_FEATURES), dtype=np.float32)
    if lb.n_sent is None or "sub_decoders" not in hw_state:
        return {"lanes": lanes, "globals": glob}

    sd = hw_state["sub_decoders"]
    tail_busy = hw_state.get("tail_busy")
    work_left = [lb.work_left(i) for i in range(NUM_DECODERS)]
    work_max = max(work_left)
    bus_left = 0

    for i in range(NUM_DECODERS):
        g = lb.n_sent[i]
        inflight = cmd_inflight(hw_state, lb.n_sent, i)
        groups1 = slots1 = work1 = 0
        if mask[i]:
            groups1, _, slots1 = plan_beat(cmds[i], sd[i], inflight)
            work1 = lb.work_between(i, g, g + groups1)
        slots2, work2 = lb.dense_beat(i, g + groups1)
        beats = lb.beats_left(i)
        bus_left += beats

        f = lanes[i]
        f[_F["work_left"]] = work_left[i] / lb0
        f[_F["work_gap"]] = (work_max - work_left[i]) / lb0
        f[_F["unsent_work"]] = lb.unsent_work(i) / lb0
        f[_F["beats_left"]] = beats / lb0
        f[_F["inflight"]] = inflight / CMD_FIFO_SIZE
        f[_F["cmd_fifo"]] = sd[i]["cmd"] / CMD_FIFO_SIZE
        f[_F["runway"]] = lb.runway(i) / RUNWAY_SCALE
        f[_F["tail_busy"]] = float(tail_busy[i]) if tail_busy is not None else 0.0
        f[_F["eligible"]] = float(mask[i])
        f[_F["beat1_slots"]] = slots1 / BEATS_ON_BUS
        f[_F["beat1_work"]] = work1 / BEAT_WORK_SCALE
        f[_F["beat2_slots"]] = slots2 / BEATS_ON_BUS
        f[_F["beat2_work"]] = work2 / BEAT_WORK_SCALE
        f[_F["lit4_fill"]] = sd[i]["lit4"] / LIT_FIFO_CAPACITIES[3]

    glob[_G["in_ready"]] = float(hw_state.get("top_in_ready", 0))
    glob[_G["bus_left"]] = bus_left / lb0
    glob[_G["work_left_max"]] = work_max / lb0
    glob[_G["lb_left"]] = max(lb.lb_parts) / lb0
    glob[_G["n_eligible"]] = float(np.count_nonzero(mask[:NUM_DECODERS])) / NUM_DECODERS

    np.clip(lanes, 0.0, 1.0, out=lanes)
    np.clip(glob, 0.0, 1.0, out=glob)
    return {"lanes": lanes, "globals": glob}
