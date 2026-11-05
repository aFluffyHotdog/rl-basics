from pathlib import Path

import numpy as np
import pytest

from deployment.balanced_testbench import BalancedPacker
from deployment.testbench_but_python import split_groups
from envs.xsim.constants import NUM_DECODERS, TB_INFLIGHT_GATE
from envs.xsim.dataset import load_hex_folder

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data_9_lanes" / "test_11_lantern" / "beats_hex"
GROUPS = [split_groups(list(q), i) for i, q in enumerate(load_hex_folder(SAMPLE))]


def _packer(rows, **kw) -> BalancedPacker:
    """Packer with projected rows pinned to `rows`. ROW_REPEAT CMDs make real lanes skip row counts
    (lane 7 of the sample jumps from 2 to 130), so arbitrary values cannot be reached by sending CMDs."""
    packer = BalancedPacker(GROUPS, **kw)
    packer.projected_rows = lambda: list(rows)
    return packer


def _hw(packer, tail_busy=(), inflight=None, ready=1) -> dict:
    """Bus ready, nothing in flight unless asked: cmd_popped = n_sent - inflight."""
    inflight = inflight or {}
    return {
        "top_in_ready": ready,
        "tail_busy": [int(s in tail_busy) for s in range(NUM_DECODERS)],
        "cmd_popped": [packer.n_sent[s] - inflight.get(s, 0) for s in range(NUM_DECODERS)],
        "sub_decoders": [{"cmd": 0} for _ in range(NUM_DECODERS)],
    }


@pytest.mark.parametrize("tie_break", ["random", "rr"])
def test_feeds_the_lane_with_fewest_projected_rows(tie_break):
    packer = _packer([5, 5, 3, 5, 5, 4, 5, 5], tie_break=tie_break, seed=0)
    beat, _ = packer.choose(_hw(packer))
    assert beat.dest == 2


def test_blocked_lowest_lane_falls_back_to_next_lowest():
    packer = _packer([5, 5, 3, 5, 5, 4, 5, 5], seed=0)
    assert packer.choose(_hw(packer, tail_busy={2}))[0].dest == 5

    packer = _packer([5, 5, 3, 5, 5, 4, 5, 5], seed=0)
    beat, _ = packer.choose(_hw(packer, inflight={2: TB_INFLIGHT_GATE + 1}))
    assert beat.dest == 5
    packer = _packer([5, 5, 3, 5, 5, 4, 5, 5], seed=0)
    beat, _ = packer.choose(_hw(packer, inflight={2: TB_INFLIGHT_GATE}))
    assert beat.dest == 2   # exactly at the gate is still eligible, as in the TB


def test_idles_when_bus_not_ready_and_cursor_holds():
    packer = _packer([5] * NUM_DECODERS, tie_break="rr")
    assert packer.choose(_hw(packer, ready=0)) == (None, "top_in_not_ready")
    assert packer.rr == 0


def test_max_lead_holds_lanes_that_are_too_far_ahead():
    rows = [10, 10, 7, 10, 10, 10, 10, 10]
    packer = _packer(rows, max_lead=3, seed=0)
    assert packer.choose(_hw(packer, tail_busy={2})) == (None, "lead_cap")

    packer = _packer(rows, max_lead=4, seed=0)
    beat, _ = packer.choose(_hw(packer, tail_busy={2}))
    assert beat is not None and beat.dest != 2

    with pytest.raises(ValueError):
        BalancedPacker(GROUPS, max_lead=0)


def test_rr_tie_break_follows_the_cursor():
    packer = _packer([5] * NUM_DECODERS, tie_break="rr")
    packer.rr = 3
    assert packer.choose(_hw(packer))[0].dest == 3
    assert packer.rr == 4
    # all lanes still tie at the minimum (5); the scan from 4 skips the busy lane 4
    assert packer.choose(_hw(packer, tail_busy={4}))[0].dest == 5


def test_random_tie_break_picks_only_tied_lanes_and_is_seeded():
    def picks(seed):
        rows = [5, 4, 4, 9, 4, 6, 4, 7]
        return [_packer(rows, seed=1000 * seed + k).choose(_hw(_packer(rows)))[0].dest for k in range(200)]

    first = picks(1)
    assert set(first) == {1, 2, 4, 6}
    assert first == picks(1)


@pytest.mark.parametrize("max_lead", [None, 2])
def test_full_image_drains_and_lead_cap_bounds_the_spread(max_lead):
    """Random gates for a whole image: every CMD is sent (no lead-cap deadlock), and with a cap the
    projected spread over unfinished lanes never exceeds max_lead - 1 + the largest single-beat jump."""
    packer = BalancedPacker(GROUPS, max_lead=max_lead, seed=3)
    rng = np.random.default_rng(3)
    popped = [0] * NUM_DECODERS
    max_jump = worst = 0
    for _ in range(200_000):
        if packer.remaining() == 0:
            break
        popped = [min(packer.n_sent[i], popped[i] + int(rng.random() < 0.5)) for i in range(NUM_DECODERS)]
        hw = {
            "top_in_ready": int(rng.random() < 0.9),
            "tail_busy": [int(rng.random() < 0.3) for _ in range(NUM_DECODERS)],
            "cmd_popped": list(popped),
            "sub_decoders": [{"cmd": 0} for _ in range(NUM_DECODERS)],
        }
        before = packer.projected_rows()
        beat, _ = packer.choose(hw)
        after = packer.projected_rows()
        if beat is not None:
            max_jump = max(max_jump, after[beat.dest] - before[beat.dest])
        live = [after[s] for s in range(NUM_DECODERS) if packer.n_sent[s] < len(GROUPS[s])]
        if live:
            worst = max(worst, max(live) - min(live))
    assert packer.remaining() == 0
    assert packer.projected_rows() == [packer.rows_before[s][-1] for s in range(NUM_DECODERS)]
    if max_lead is not None:
        assert worst <= max_lead - 1 + max_jump
