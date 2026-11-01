"""Per-lane observation v3: v2 layout, but work is measured in output rows, not estimated cycles.

A row is ROW_BYTES of decoded output; each lane emits NUM_ROWS per frame. Rows still to send come
from the per-group row table in LowerBoundTracker (built from the CMD output-byte model), rows
already decoded from the lane's row queue (LowerBoundTracker.rows_decoded, free-run pop rule).
Row counts are divided by NUM_ROWS, beats by bus0 (the episode's initial dense beat total), the
short-horizon cycle estimates (runway, beat work) by the same hardware bounds as v2.
"""
import math
import gymnasium as gym
import numpy as np

from .constants import BEATS_ON_BUS, CMD_FIFO_SIZE, LIT_FIFO_CAPACITIES, NUM_DECODERS, NUM_ROWS
from .flow_control import cmd_inflight, plan_beat
from .observation_v2 import BEAT_WORK_SCALE, RUNWAY_SCALE
from .reward import LowerBoundTracker

LANE_FEATURES = (
    "work_left",     # rows whose CMDs are not all on the bus yet / NUM_ROWS
    "work_gap",      # (max lane work_left - this lane's work_left) / NUM_ROWS
    "unsent_work",   # rows sent to the lane but not yet decoded / NUM_ROWS
    "beats_left",    # dense beats still needed for this lane / bus0
    "inflight",      # CMDs sent but not popped / CMD FIFO depth
    "cmd_fifo",      # CMD FIFO occupancy / depth
    "runway",        # buffered decode work (sent, not popped) / RUNWAY_SCALE
    "tail_busy",     # arbiter tail busy flag
    "eligible",      # action mask bit
    "lane_outputs",  # exact rows decoded by this lane / NUM_ROWS
    "lag_score",     # (mean decoded rows - this lane's rows) / max(stdev, 1)
    "expected_utilization",  # slots the next beat would use / BEATS_ON_BUS
    "beat1_slots",   # slots the beat sent now would carry (0 if not eligible) / BEATS_ON_BUS
    "beat1_work",    # decode work in that beat / BEAT_WORK_SCALE
    "beat2_slots",   # slots of the following dense beat / BEATS_ON_BUS
    "beat2_work",    # decode work in that beat / BEAT_WORK_SCALE
    "lit4_fill",     # literal bank 4 occupancy / capacity (the smallest bank)
)
GLOBAL_FEATURES = (
    "in_ready",      # arbiter input ready
    "bus_left",      # total dense beats left / bus0
    "work_left_max", # bottleneck lane's work_left / NUM_ROWS
    "n_eligible",    # eligible lanes / NUM_DECODERS
    "mean_outputs",  # mean decoded rows across lanes / NUM_ROWS
    "stdev_outputs", # population stdev of decoded rows across lanes / NUM_ROWS
)

_F = {name: k for k, name in enumerate(LANE_FEATURES)}
_G = {name: k for k, name in enumerate(GLOBAL_FEATURES)}


def build_observation_space_v3() -> gym.spaces.Dict:
    lane_low = np.zeros((NUM_DECODERS, len(LANE_FEATURES)), dtype=np.float32)
    lane_high = np.ones((NUM_DECODERS, len(LANE_FEATURES)), dtype=np.float32)
    lane_low[:, _F["lag_score"]] = -math.sqrt(NUM_DECODERS - 1)
    lane_high[:, _F["lag_score"]] = math.sqrt(NUM_DECODERS - 1)
    return gym.spaces.Dict({
        "lanes": gym.spaces.Box(
            low=lane_low,
            high=lane_high,
            dtype=np.float32,
        ),
        "globals": gym.spaces.Box(low=0.0, high=1.0, shape=(len(GLOBAL_FEATURES),), dtype=np.float32),
    })


def build_observation_v3(hw_state: dict, cmds: list, lb: LowerBoundTracker, mask: np.ndarray) -> dict:
    """Call after lb.compute() so lb.rows_popped reflects the current hw_state."""
    lanes = np.zeros((NUM_DECODERS, len(LANE_FEATURES)), dtype=np.float32)
    glob = np.zeros(len(GLOBAL_FEATURES), dtype=np.float32)
    if lb.n_sent is None or "sub_decoders" not in hw_state:
        return {"lanes": lanes, "globals": glob}

    sd = hw_state["sub_decoders"]
    tail_busy = hw_state.get("tail_busy")
    bus0 = max(lb.bus0, 1)
    rows_unsent = [lb.rows_unsent(i) for i in range(NUM_DECODERS)]
    rows_max = max(rows_unsent)
    rows_outputs = [lb.rows_decoded(i, hw_state) for i in range(NUM_DECODERS)]
    mean_outputs = float(np.mean(rows_outputs))
    stdev_outputs = float(np.std(rows_outputs))
    lag_denominator = max(stdev_outputs, 1.0)
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
        f[_F["work_left"]] = rows_unsent[i] / NUM_ROWS
        f[_F["work_gap"]] = (rows_max - rows_unsent[i]) / NUM_ROWS
        f[_F["unsent_work"]] = lb.rows_inflight(i, hw_state) / NUM_ROWS
        f[_F["beats_left"]] = beats / bus0
        f[_F["inflight"]] = inflight / CMD_FIFO_SIZE
        f[_F["cmd_fifo"]] = sd[i]["cmd"] / CMD_FIFO_SIZE
        f[_F["runway"]] = lb.runway(i) / RUNWAY_SCALE
        f[_F["tail_busy"]] = float(tail_busy[i]) if tail_busy is not None else 0.0
        f[_F["eligible"]] = float(mask[i])
        f[_F["lane_outputs"]] = rows_outputs[i] / NUM_ROWS
        f[_F["lag_score"]] = (mean_outputs - rows_outputs[i]) / lag_denominator
        f[_F["expected_utilization"]] = slots1 / BEATS_ON_BUS
        f[_F["beat1_slots"]] = slots1 / BEATS_ON_BUS
        f[_F["beat1_work"]] = work1 / BEAT_WORK_SCALE
        f[_F["beat2_slots"]] = slots2 / BEATS_ON_BUS
        f[_F["beat2_work"]] = work2 / BEAT_WORK_SCALE
        f[_F["lit4_fill"]] = sd[i]["lit4"] / LIT_FIFO_CAPACITIES[3]

    glob[_G["in_ready"]] = float(hw_state.get("top_in_ready", 0))
    glob[_G["bus_left"]] = bus_left / bus0
    glob[_G["work_left_max"]] = rows_max / NUM_ROWS
    glob[_G["n_eligible"]] = float(np.count_nonzero(mask[:NUM_DECODERS])) / NUM_DECODERS
    glob[_G["mean_outputs"]] = mean_outputs / NUM_ROWS
    glob[_G["stdev_outputs"]] = stdev_outputs / NUM_ROWS

    lag_scores = lanes[:, _F["lag_score"]].copy()
    np.clip(lanes, 0.0, 1.0, out=lanes)
    lanes[:, _F["lag_score"]] = lag_scores
    np.clip(glob, 0.0, 1.0, out=glob)
    return {"lanes": lanes, "globals": glob}
