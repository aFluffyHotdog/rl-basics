from collections import deque
from pathlib import Path

import numpy as np

from deployment.testbench_but_python import TestbenchPacker as TBPacker, split_groups
from envs.xsim.constants import NO_OP_ACTION, NUM_ACTIONS, NUM_DECODERS, TB_INFLIGHT_GATE
from envs.xsim.dataset import load_hex_folder
from envs.xsim.flow_control import compute_action_mask, plan_beat, tb_rr_advance
from policies.round_robin import RoundRobinTeacher

SAMPLE = Path(__file__).resolve().parents[1] / "sample_data_9_lanes" / "test_11_lantern" / "beats_hex"


def _mask(*lanes, noop=False):
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    mask[list(lanes)] = True
    mask[NO_OP_ACTION] = noop or not lanes
    return mask


def test_first_pick_is_lane_0_and_pointer_wraps_through_the_bypass_slot():
    teacher = RoundRobinTeacher()
    assert teacher.act(_mask(*range(NUM_DECODERS))) == 0
    assert teacher.act(_mask(7)) == 7
    # cursor now sits on the (disabled) bypass slot 8; the scan wraps to lane 0
    assert teacher.rr == 8
    assert teacher.act(_mask(*range(NUM_DECODERS))) == 0


def test_skips_ineligible_lanes():
    teacher = RoundRobinTeacher()
    assert teacher.act(_mask(3, 5)) == 3
    assert teacher.act(_mask(1, 3, 5)) == 5
    assert teacher.act(_mask(1, 3)) == 1


def test_noop_only_when_only_noop_is_eligible_and_cursor_rule():
    teacher = RoundRobinTeacher()
    assert teacher.act(_mask(), in_ready=False) == NO_OP_ACTION
    assert teacher.rr == 0            # bus not ready: cursor holds
    assert teacher.act(_mask(), in_ready=True) == NO_OP_ACTION
    assert teacher.rr == 1            # ready but nothing eligible: cursor steps
    assert teacher.act(_mask(0, 2)) == 2


def test_deterministic_and_env_cursor_tracks_teacher():
    rng = np.random.default_rng(0)
    seq = [(rng.random(NUM_DECODERS) < 0.4, rng.random() < 0.8) for _ in range(500)]
    runs = []
    for _ in range(2):
        teacher, env_rr, actions = RoundRobinTeacher(), 0, []
        for lanes, ready in seq:
            mask = _mask(*np.flatnonzero(lanes & ready))
            action = teacher.act(mask, ready)
            env_rr = tb_rr_advance(env_rr, action, ready)   # what DecoderEnvXSim._track_schedule does
            assert env_rr == teacher.rr
            actions.append(action)
        runs.append(actions)
    assert runs[0] == runs[1]


def test_teacher_on_gated_env_mask_matches_testbench_packer():
    """Env mask (TB gate) + teacher + env packing == testbench_but_python's packer, beat for beat."""
    lanes = load_hex_folder(SAMPLE)
    packer = TBPacker([split_groups(list(q), i) for i, q in enumerate(lanes)])
    cmds = [deque(q) for q in lanes]
    teacher, n_sent = RoundRobinTeacher(), [0] * NUM_DECODERS
    popped = [0] * NUM_DECODERS
    rng = np.random.default_rng(1)
    for _ in range(3000):
        popped = [min(n_sent[i], popped[i] + int(rng.random() < 0.4)) for i in range(NUM_DECODERS)]
        hw = {
            "top_in_ready": int(rng.random() < 0.85),
            "tail_busy": [int(rng.random() < 0.3) for _ in range(NUM_DECODERS)],
            "cmd_popped": list(popped),
            "sub_decoders": [{"cmd": 0} for _ in range(NUM_DECODERS)],
        }
        mask = compute_action_mask(hw, cmds, n_sent, TB_INFLIGHT_GATE)
        action = teacher.act(mask, bool(hw["top_in_ready"]))
        beat, _ = packer.choose(hw)

        assert (beat is None) == (action == NO_OP_ACTION)
        assert teacher.rr == packer.rr
        if beat is not None:
            assert action == beat.dest
            groups, n_tokens, slots = plan_beat(cmds[action], hw["sub_decoders"][action],
                                                n_sent[action] - popped[action])
            assert (groups, slots) == (beat.ncmd, beat.slots)
            for _ in range(n_tokens):
                cmds[action].popleft()
            n_sent[action] += groups
        assert n_sent == packer.n_sent
