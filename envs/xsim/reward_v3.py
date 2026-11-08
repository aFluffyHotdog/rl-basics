"""Reward v3: throughput minus per-lane output spread. Per step,

    r_t = LAMBDA_THROUGHPUT * throughput(t) - LAMBDA_STDEV * sigma(t)

where
  * throughput(t) is the number of complete output rows produced across all lanes over the last
    THROUGHPUT_WINDOW testbench cycles (the env keeps a rolling window and passes its sum), and
  * sigma(t) is the population stdev of decoded rows across the NUM_DECODERS lanes
    (rows[i] = LowerBoundTracker.rows_decoded(i)).

The agent is rewarded for pushing rows out fast (throughput) and for keeping the lanes even
(low output stdev), so it both maximizes throughput and minimizes the per-lane row spread.
"""
import numpy as np

from .constants import LAMBDA_STDEV, LAMBDA_THROUGHPUT, NUM_DECODERS


def reward_v3(throughput: float, rows) -> float:
    """throughput: complete rows produced over the recent window. rows: decoded rows per lane."""
    rows = np.asarray(rows[:NUM_DECODERS], dtype=np.float64)
    sigma = float(rows.std())
    return float(LAMBDA_THROUGHPUT * throughput - LAMBDA_STDEV * sigma)
