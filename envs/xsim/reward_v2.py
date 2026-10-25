"""Reward v2: no potential shaping. Per step,

    r_t = -LAMBDA_T - LAMBDA_B * sigma(t) + LAMBDA_A * z_a * (slots_a / BEATS_ON_BUS) * [a_t is not no-op]
    z_a = (mean(rows) - rows[a]) / max(sigma(t), 1)

where rows[i] is the number of rows lane i has output (LowerBoundTracker.rows_decoded) and sigma is
their population stdev over the NUM_DECODERS lanes. z_a is positive when the fed lane is behind the
mean, negative when it leads, so a full beat to the slowest lane earns the most.
"""
import numpy as np

from .constants import BEATS_ON_BUS, LAMBDA_A, LAMBDA_B, LAMBDA_T, NO_OP_ACTION, NUM_DECODERS


def reward_v2(rows, action: int, slots_used: int) -> float:
    """rows: rows output per lane (len NUM_DECODERS); action/slots_used as returned by _pack_beat."""
    rows = np.asarray(rows[:NUM_DECODERS], dtype=np.float64)
    sigma = float(rows.std())
    r = -LAMBDA_T - LAMBDA_B * sigma
    if action != NO_OP_ACTION:
        z = (rows.mean() - rows[action]) / max(sigma, 1.0)
        r += LAMBDA_A * z * (slots_used / BEATS_ON_BUS)
    return float(r)
