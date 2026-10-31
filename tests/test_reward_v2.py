"""Run with: python -m pytest tests  (or python tests/test_reward_v2.py)."""
import math
import sys, types, signal
sys.modules.setdefault("torch", types.SimpleNamespace(signal=signal))   # sim_client imports signal from torch

from envs.xsim.constants import (
    BEATS_ON_BUS,
    LAMBDA_A,
    LAMBDA_B,
    LAMBDA_C,
    LAMBDA_D,
    LAMBDA_T,
    NO_OP_ACTION,
)
from envs.xsim.reward_v2 import reward_v2


def test_reward_v2():
    even = [10] * 8
    assert reward_v2(even, NO_OP_ACTION, 0) == -LAMBDA_T                 # no spread, no-op: time cost only
    assert reward_v2(even, 3, 8) == -LAMBDA_T + LAMBDA_C                # at the mean: z = 0
    rows = [10, 10, 10, 10, 10, 10, 10, 2]
    sigma = math.sqrt(sum((r - 9) ** 2 for r in rows) / 8)
    base = -LAMBDA_T - LAMBDA_B * sigma
    assert math.isclose(reward_v2(rows, NO_OP_ACTION, 0), base)
    behind = reward_v2(rows, 7, 8)                                       # full beat to the lagging lane
    leader = reward_v2(rows, 0, 8)
    assert behind > leader > base
    assert math.isclose(
        behind - base,
        LAMBDA_A * (9 - 2) / sigma + LAMBDA_C,
    )
    assert math.isclose(
        reward_v2(rows, 7, 4) - base,
        LAMBDA_A * (9 - 2) / sigma + LAMBDA_C / 2,
    )
    assert math.isclose(reward_v2([0, 1] * 4, 1, BEATS_ON_BUS) - reward_v2([0, 1] * 4, NO_OP_ACTION, 0),
                        -LAMBDA_A * 0.5 + LAMBDA_C)                     # sigma < 1 -> divisor clamps to 1


def test_reward_v2_penalizes_lost_slots_against_best_lane():
    rows = [10] * 8
    same_capacity = reward_v2(rows, 0, 4, best_slots=4)
    lower_capacity = reward_v2(rows, 0, 4, best_slots=7)

    assert math.isclose(same_capacity - lower_capacity, 3 * LAMBDA_D)


def test_reward_v2_does_not_penalize_no_op_or_negative_slot_difference():
    rows = [10] * 8

    assert reward_v2(rows, NO_OP_ACTION, 0, best_slots=8) == reward_v2(
        rows, NO_OP_ACTION, 0
    )
    assert reward_v2(rows, 0, 5, best_slots=3) == reward_v2(
        rows, 0, 5, best_slots=5
    )


if __name__ == "__main__":
    test_reward_v2(); print("ok")
