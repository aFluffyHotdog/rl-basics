import numpy as np

from envs.xsim.constants import NO_OP_ACTION
from envs.xsim.flow_control import compute_action_mask, packable_slots_by_lane


def _token_queue(slots: int) -> list[int]:
    if slots == 1:
        return [0]
    return [(slots - 1) << 11, 1 << 16]


def _hw_state(tail_busy=None, top_in_ready=1):
    return {
        "top_in_ready": top_in_ready,
        "tail_busy": tail_busy,
        "sub_decoders": [{"cmd": 0} for _ in range(8)],
    }


def test_packable_slots_by_lane_matches_feasible_beat_capacity():
    cmds = [_token_queue(1), _token_queue(2)] + [[] for _ in range(6)]
    slots = packable_slots_by_lane(_hw_state(), cmds, [0] * 8)

    assert slots == [1, 2, 0, 0, 0, 0, 0, 0]
    assert compute_action_mask(_hw_state(), cmds, [0] * 8)[:8].tolist() == [
        True,
        True,
        False,
        False,
        False,
        False,
        False,
        False,
    ]


def test_inflight_gate_blocks_lanes_over_the_testbench_limit():
    cmds = [_token_queue(1), _token_queue(1)] + [[] for _ in range(6)]
    hw_state = _hw_state()
    hw_state["cmd_popped"] = [0] * 8
    n_sent = [8, 9] + [0] * 6   # lane 0: 8 in flight, lane 1: 9 in flight

    assert compute_action_mask(hw_state, cmds, n_sent)[:2].tolist() == [True, True]
    gated = compute_action_mask(hw_state, cmds, n_sent, inflight_gate=8)
    assert gated[:2].tolist() == [True, False]
    assert not gated[NO_OP_ACTION]


def test_packable_slots_excludes_busy_lanes_and_all_lanes_when_input_blocked():
    cmds = [_token_queue(1), _token_queue(2)] + [[] for _ in range(6)]
    busy_slots = packable_slots_by_lane(
        _hw_state(tail_busy=[False, True] + [False] * 6),
        cmds,
        [0] * 8,
    )
    blocked_slots = packable_slots_by_lane(
        _hw_state(top_in_ready=0),
        cmds,
        [0] * 8,
    )

    assert busy_slots == [1, 0, 0, 0, 0, 0, 0, 0]
    assert blocked_slots == [0] * 8
    assert compute_action_mask(_hw_state(top_in_ready=0), cmds, [0] * 8)[
        NO_OP_ACTION
    ]
