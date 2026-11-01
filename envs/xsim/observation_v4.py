"""Per-lane observation v4: v3 plus the scheduling state of the testbench round-robin teacher.

The v20 testbench packer (deployment/testbench_but_python.py) is stateful: it scans lanes from a
cursor that moves past the lane it served and also steps on idle cycles. A policy can only imitate
it if that state is visible, so two per-lane features are appended to the v3 lane block:

  cycles_since_fed  cycles since the lane last received a beat / CYCLES_SINCE_FED_SCALE, clipped to 1
  rr_distance       scan distance from the env-tracked cursor, ((lane - cursor) mod 9) / 8;
                    the teacher serves the eligible lane with the smallest value

Both are tracked by the env from its own actions (DecoderEnvXSim._track_schedule), so they are
available at deployment without a teacher. The globals are unchanged from v3.
"""
import gymnasium as gym
import numpy as np

from .constants import CYCLES_SINCE_FED_SCALE, NUM_DECODERS, TB_RR_CHANNELS
from .observation_v3 import (
    GLOBAL_FEATURES,  # noqa: F401  (re-exported: v4 globals == v3 globals)
    LANE_FEATURES as LANE_FEATURES_V3,
    build_observation_space_v3,
    build_observation_v3,
)
from .reward import LowerBoundTracker

LANE_FEATURES = LANE_FEATURES_V3 + ("cycles_since_fed", "rr_distance")
N_EXTRA = len(LANE_FEATURES) - len(LANE_FEATURES_V3)


def build_observation_space_v4() -> gym.spaces.Dict:
    v3 = build_observation_space_v3()
    lanes = v3["lanes"]
    low = np.concatenate([lanes.low, np.zeros((NUM_DECODERS, N_EXTRA), dtype=np.float32)], axis=1)
    high = np.concatenate([lanes.high, np.ones((NUM_DECODERS, N_EXTRA), dtype=np.float32)], axis=1)
    return gym.spaces.Dict({
        "lanes": gym.spaces.Box(low=low, high=high, dtype=np.float32),
        "globals": v3["globals"],
    })


def schedule_features(cycles_since_fed, rr_cursor: int) -> np.ndarray:
    """(NUM_DECODERS, 2) block: normalized cycles_since_fed and round-robin scan distance."""
    extra = np.zeros((NUM_DECODERS, N_EXTRA), dtype=np.float32)
    extra[:, 0] = np.minimum(np.asarray(cycles_since_fed, dtype=np.float32) / CYCLES_SINCE_FED_SCALE, 1.0)
    extra[:, 1] = ((np.arange(NUM_DECODERS) - rr_cursor) % TB_RR_CHANNELS) / (TB_RR_CHANNELS - 1)
    return extra


def build_observation_v4(hw_state: dict, cmds: list, lb: LowerBoundTracker, mask: np.ndarray,
                         cycles_since_fed, rr_cursor: int) -> dict:
    """Call after lb.compute() so lb.rows_popped reflects the current hw_state (as for v3)."""
    obs = build_observation_v3(hw_state, cmds, lb, mask)
    obs["lanes"] = np.concatenate([obs["lanes"], schedule_features(cycles_since_fed, rr_cursor)], axis=1)
    return obs
