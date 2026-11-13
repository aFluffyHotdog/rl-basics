"""Reward v5: cycle cost plus potential-based shaping on the dense beats still to be sent.

Let B_left(s) = sum over lanes of LowerBoundTracker.beats_left(i): the number of beats needed to
send everything that is left when each lane's remaining tokens are packed greedily into 8-slot
beats. With the potential Phi(s) = -B_left(s) the per-step reward is

    r_t = [ -cycles_elapsed + (B_left(s_t) - B_left(s_{t+1})) ] / REWARD_SCALE

A dense beat lowers B_left by exactly 1, a sparse one by less, so the agent is paid for packing.
Since GAMMA = 1 the shaping telescopes: an episode sums to (B_left(s_0) - B_left(s_end) - makespan)
/ REWARD_SCALE, so it does not change which schedules are optimal.
"""
from .constants import REWARD_SCALE


def reward_v5(cycles_elapsed: int, beats_left_prev: int, beats_left_now: int) -> float:
    """cycles_elapsed: clock cycles since the previous env step (>= 1, includes skipped cycles).
    beats_left_prev / beats_left_now: B_left before and after this step."""
    return (-float(cycles_elapsed) + float(beats_left_prev - beats_left_now)) / REWARD_SCALE
