"""Run with: python -m pytest tests  (or python tests/test_reward_v2.py)."""
import math
import sys, types, signal
sys.modules.setdefault("torch", types.SimpleNamespace(signal=signal))   # sim_client imports signal from torch

from envs.xsim.constants import BEATS_ON_BUS, LAMBDA_A, LAMBDA_B, LAMBDA_T, NO_OP_ACTION
from envs.xsim.reward_v2 import reward_v2


def test_reward_v2():
    even = [10] * 8
    assert reward_v2(even, NO_OP_ACTION, 0) == -LAMBDA_T                 # no spread, no-op: time cost only
    assert reward_v2(even, 3, 8) == -LAMBDA_T                            # at the mean: z = 0
    rows = [10, 10, 10, 10, 10, 10, 10, 2]
    sigma = math.sqrt(sum((r - 9) ** 2 for r in rows) / 8)
    base = -LAMBDA_T - LAMBDA_B * sigma
    assert math.isclose(reward_v2(rows, NO_OP_ACTION, 0), base)
    behind = reward_v2(rows, 7, 8)                                       # full beat to the lagging lane
    leader = reward_v2(rows, 0, 8)
    assert behind > base > leader
    assert math.isclose(behind - base, LAMBDA_A * (9 - 2) / sigma)       # z_a * slots/8 with slots = 8
    assert math.isclose(reward_v2(rows, 7, 4) - base, (behind - base) / 2)   # half-full beat, half credit
    assert math.isclose(reward_v2([0, 1] * 4, 1, BEATS_ON_BUS) - reward_v2([0, 1] * 4, NO_OP_ACTION, 0),
                        -LAMBDA_A * 0.5)                                 # sigma < 1 -> divisor clamps to 1


if __name__ == "__main__":
    test_reward_v2(); print("ok")
