"""Reward v4: per-step cycle cost (scaled by a fixed constant), optionally plus a bonus per complete row."""
from .constants import REWARD_SCALE


def reward_v4(cycles_elapsed: int, rows_done_delta: int = 0, alpha: float = 0.0) -> float:
    """cycles_elapsed: clock cycles since the previous env step (>= 1, includes skipped cycles).
    rows_done_delta: complete rows output since the previous env step (not a window).
    Summed over an episode with alpha = 0 this is -makespan / REWARD_SCALE."""
    return -float(cycles_elapsed) / REWARD_SCALE + alpha * float(rows_done_delta)
