import pytest

from envs.xsim.constants import REWARD_SCALE
from envs.xsim.reward_v5 import reward_v5


def test_dense_beat_pays_one_cycle_back():
    # one cycle elapsed, B_left drops by 1 (a dense beat): the two terms cancel
    assert reward_v5(1, 10, 9) == pytest.approx(0.0)


def test_idle_cycle_costs_one_cycle():
    assert reward_v5(1, 10, 10) == pytest.approx(-1 / REWARD_SCALE)


def test_sparse_beat_costs_more_than_dense():
    assert reward_v5(1, 10, 10) < reward_v5(1, 10, 9)


def test_skipped_cycles_are_charged():
    assert reward_v5(5, 10, 9) == pytest.approx(-4 / REWARD_SCALE)


def test_shaping_telescopes():
    steps = [(3, 8, 7), (1, 7, 7), (2, 7, 5), (1, 5, 0)]
    total = sum(reward_v5(c, a, b) for c, a, b in steps)
    assert total == pytest.approx((8 - 0 - sum(c for c, _, _ in steps)) / REWARD_SCALE)
