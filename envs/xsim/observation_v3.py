"""Per-lane observation: work measured in output rows, not estimated cycles.

A row is ROW_BYTES of decoded output; each lane emits NUM_ROWS per frame. Rows still to send come
from the per-group row table in LowerBoundTracker (built from the CMD output-byte model), rows
already decoded from the lane's row queue (LowerBoundTracker.rows_decoded, free-run pop rule).
Row counts are divided by NUM_ROWS, beats by bus0 (the episode's initial dense beat total), the
short-horizon beat-work estimate by the hardware bound BEAT_WORK_SCALE, throughput by
THROUGHPUT_SCALE.

This is the single observation layout for the env. The lane block carries a one-hot of the next
beat's head-token type (CMD / LIT / RLE) and the lane's aggregate FIFO fill; the globals carry a
windowed throughput signal (rows produced over the last THROUGHPUT_WINDOW cycles).
"""
import gymnasium as gym
import numpy as np

from .constants import (
    BEAT_WORK_SCALE,
    BEATS_ON_BUS,
    CMD_FIFO_SIZE,
    NUM_DECODERS,
    NUM_ROWS,
    THROUGHPUT_SCALE,
    TOTAL_FIFO_CAPACITY,
)
from .flow_control import cmd_inflight, plan_beat
from .reward import LowerBoundTracker
from .tokens import TYPE_INDEX, decode_token

# One-hot of the next beat's head-token type. SEED heads map to the all-zero vector.
BEAT_TYPES = ("CMD", "LIT", "RLE")

LANE_FEATURES = (
    "work_left",     # rows whose CMDs are not all on the bus yet / NUM_ROWS
    "beats_left",    # dense beats still needed for this lane / bus0
    "inflight",      # CMDs sent but not popped / CMD FIFO depth
    "cmd_fifo",      # CMD FIFO occupancy / depth
    "tail_busy",     # arbiter tail busy flag
    "eligible",      # action mask bit
    "lane_outputs",  # exact rows decoded by this lane / NUM_ROWS
    "expected_utilization",  # slots the next beat would use / BEATS_ON_BUS
    "beat1_type_cmd",  # one-hot: next beat's head token is a CMD
    "beat1_type_lit",  # one-hot: next beat's head token is a LIT
    "beat1_type_rle",  # one-hot: next beat's head token is an RLE
    "beat1_slots",   # slots the beat sent now would carry (0 if not eligible) / BEATS_ON_BUS
    "beat1_work",    # decode work in that beat / BEAT_WORK_SCALE
    "beat2_slots",   # slots of the following dense beat / BEATS_ON_BUS
    "beat2_work",    # decode work in that beat / BEAT_WORK_SCALE
    "fifo_fill",     # total occupancy of all this lane's FIFOs / TOTAL_FIFO_CAPACITY
)
GLOBAL_FEATURES = (
    "throughput",    # complete rows produced over the last THROUGHPUT_WINDOW cycles / THROUGHPUT_SCALE
    "bus_left",      # total dense beats left / bus0
    "n_eligible",    # eligible lanes / NUM_DECODERS
    "mean_outputs",  # mean decoded rows across lanes / NUM_ROWS
    "stdev_outputs", # population stdev of decoded rows across lanes / NUM_ROWS
)

_F = {name: k for k, name in enumerate(LANE_FEATURES)}
_G = {name: k for k, name in enumerate(GLOBAL_FEATURES)}


def build_observation_space_v3() -> gym.spaces.Dict:
    return gym.spaces.Dict({
        "lanes": gym.spaces.Box(
            low=np.zeros((NUM_DECODERS, len(LANE_FEATURES)), dtype=np.float32),
            high=np.ones((NUM_DECODERS, len(LANE_FEATURES)), dtype=np.float32),
            dtype=np.float32,
        ),
        "globals": gym.spaces.Box(low=0.0, high=1.0, shape=(len(GLOBAL_FEATURES),), dtype=np.float32),
    })


def _beat1_type_onehot(queue) -> tuple[float, float, float]:
    """One-hot (cmd, lit, rle) of the head token of the beat sent now; all-zero for SEED / empty."""
    if not queue:
        return 0.0, 0.0, 0.0
    category = decode_token(queue[0]).category
    idx = TYPE_INDEX.get(category)
    return tuple(1.0 if idx == TYPE_INDEX[t] else 0.0 for t in BEAT_TYPES)


def build_observation_v3(hw_state: dict, cmds: list, lb: LowerBoundTracker, mask: np.ndarray,
                         throughput: float = 0.0) -> dict:
    """Call after lb.compute() so lb.rows_popped reflects the current hw_state.

    throughput: complete rows produced over the last THROUGHPUT_WINDOW cycles (env-tracked window).
    """
    lanes = np.zeros((NUM_DECODERS, len(LANE_FEATURES)), dtype=np.float32)
    glob = np.zeros(len(GLOBAL_FEATURES), dtype=np.float32)
    if lb.n_sent is None or "sub_decoders" not in hw_state:
        return {"lanes": lanes, "globals": glob}

    sd = hw_state["sub_decoders"]
    tail_busy = hw_state.get("tail_busy")
    bus0 = max(lb.bus0, 1)
    rows_outputs = [lb.rows_decoded(i, hw_state) for i in range(NUM_DECODERS)]
    mean_outputs = float(np.mean(rows_outputs))
    stdev_outputs = float(np.std(rows_outputs))
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
        beat1_cmd, beat1_lit, beat1_rle = _beat1_type_onehot(cmds[i]) if mask[i] else (0.0, 0.0, 0.0)
        fifo_fill = sum(sd[i][field] for field in ("cmd", "lit1", "lit2", "lit3", "lit4", "out"))

        f = lanes[i]
        f[_F["work_left"]] = lb.rows_unsent(i) / NUM_ROWS
        f[_F["beats_left"]] = beats / bus0
        f[_F["inflight"]] = inflight / CMD_FIFO_SIZE
        f[_F["cmd_fifo"]] = sd[i]["cmd"] / CMD_FIFO_SIZE
        f[_F["tail_busy"]] = float(tail_busy[i]) if tail_busy is not None else 0.0
        f[_F["eligible"]] = float(mask[i])
        f[_F["lane_outputs"]] = rows_outputs[i] / NUM_ROWS
        f[_F["expected_utilization"]] = slots1 / BEATS_ON_BUS
        f[_F["beat1_type_cmd"]] = beat1_cmd
        f[_F["beat1_type_lit"]] = beat1_lit
        f[_F["beat1_type_rle"]] = beat1_rle
        f[_F["beat1_slots"]] = slots1 / BEATS_ON_BUS
        f[_F["beat1_work"]] = work1 / BEAT_WORK_SCALE
        f[_F["beat2_slots"]] = slots2 / BEATS_ON_BUS
        f[_F["beat2_work"]] = work2 / BEAT_WORK_SCALE
        f[_F["fifo_fill"]] = fifo_fill / TOTAL_FIFO_CAPACITY

    glob[_G["throughput"]] = throughput / THROUGHPUT_SCALE
    glob[_G["bus_left"]] = bus_left / bus0
    glob[_G["n_eligible"]] = float(np.count_nonzero(mask[:NUM_DECODERS])) / NUM_DECODERS
    glob[_G["mean_outputs"]] = mean_outputs / NUM_ROWS
    glob[_G["stdev_outputs"]] = stdev_outputs / NUM_ROWS

    np.clip(lanes, 0.0, 1.0, out=lanes)
    np.clip(glob, 0.0, 1.0, out=glob)
    return {"lanes": lanes, "globals": glob}
