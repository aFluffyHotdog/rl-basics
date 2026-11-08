"""Run with: python -m pytest tests  (or python tests/test_reward_v3.py)."""
import math
import sys, types, signal
sys.modules.setdefault("torch", types.SimpleNamespace(signal=signal))   # sim_client imports signal from torch

from envs.xsim.constants import LAMBDA_STDEV, LAMBDA_THROUGHPUT
from envs.xsim.reward_v3 import reward_v3


def _stdev(rows):
    mean = sum(rows) / len(rows)
    return math.sqrt(sum((r - mean) ** 2 for r in rows) / len(rows))


def test_even_lanes_have_no_stdev_penalty():
    even = [10] * 8
    assert reward_v3(0.0, even) == 0.0                         # no throughput, no spread
    assert reward_v3(5.0, even) == LAMBDA_THROUGHPUT * 5.0     # throughput only, spread is zero


def test_higher_throughput_scores_higher():
    rows = [10] * 8
    assert reward_v3(12.0, rows) > reward_v3(4.0, rows)
    assert math.isclose(reward_v3(12.0, rows) - reward_v3(4.0, rows), LAMBDA_THROUGHPUT * 8.0)


def test_higher_spread_scores_lower():
    even = [10] * 8
    uneven = [10, 10, 10, 10, 10, 10, 10, 2]
    assert reward_v3(5.0, uneven) < reward_v3(5.0, even)
    assert math.isclose(reward_v3(5.0, even) - reward_v3(5.0, uneven), LAMBDA_STDEV * _stdev(uneven))


def test_formula_matches_definition():
    rows = [3, 7, 10, 10, 10, 10, 10, 10]
    throughput = 9.0
    expected = LAMBDA_THROUGHPUT * throughput - LAMBDA_STDEV * _stdev(rows)
    assert math.isclose(reward_v3(throughput, rows), expected)


if __name__ == "__main__":
    test_even_lanes_have_no_stdev_penalty()
    test_higher_throughput_scores_higher()
    test_higher_spread_scores_lower()
    test_formula_matches_definition()
    print("ok")
