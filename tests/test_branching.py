"""Branching-experiment tests; no xsim. A tiny fake env stands in for DecoderEnvXSim."""
import sys
import types

import numpy as np
import pytest

try:   # envs.xsim imports torch via sim_client; the tests do not need the real thing
    import torch  # noqa: F401
except ImportError:
    stub = types.ModuleType("torch")
    stub.signal = types.SimpleNamespace(CTRL_BREAK_EVENT=0)
    sys.modules["torch"] = stub

from experiments.branching import analyze  # noqa: E402
from experiments.branching.replay import ReplayDivergence, run_episode  # noqa: E402
from policies.round_robin import RoundRobinTeacher  # noqa: E402

NO_OP = 8


def mask(*lanes):
    m = np.zeros(9, dtype=bool)
    m[list(lanes)] = True
    m[NO_OP] = not lanes
    return m


class FakeEnv:
    """Fixed per-step masks; terminates after len(masks) steps with makespan = 100 + sum(actions)."""

    def __init__(self, masks):
        self.masks, self.i, self.taken = masks, 0, []
        self.cycles, self.hw_state, self.unwrapped = 0, {"top_in_ready": 1}, self

    def reset(self, **_):
        self.i, self.taken, self.cycles = 0, [], 0
        return {"x": np.zeros(1)}, {}

    def action_masks(self):
        return self.masks[min(self.i, len(self.masks) - 1)].copy()

    def step(self, a):
        self.taken.append(a)
        self.i += 1
        self.cycles += 1
        done = self.i >= len(self.masks)
        return {"x": np.zeros(1)}, 0.0, done, False, ({"makespan": 100 + sum(self.taken)} if done else {})


def test_teacher_wraps_skips_and_noops():
    t = RoundRobinTeacher()          # cursor 0 == last_served 7
    assert t.act(mask(2, 6)) == 2
    assert t.act(mask(2, 6)) == 6
    assert t.act(mask(2, 6)) == 2    # wrapped past 7
    assert t.act(mask(0, 1)) == 0    # skipped ineligible lanes 3..7
    assert t.act(mask()) == NO_OP


def test_forced_action_sets_cursor():
    t = RoundRobinTeacher()
    t.update(5, True)                # forced lane 5 == last_served = 5
    assert t.act(mask(5, 6)) == 6
    t.update(7, True)
    assert t.act(mask(0, 7)) == 0


def test_run_episode_replay_and_force():
    env = FakeEnv([mask(0, 1), mask(0, 1), mask(2)])
    base = run_episode(env)
    assert base.actions == [0, 1, 2] and base.decision_steps == [0, 1]
    assert sorted(base.decision_obs) == [0, 1] and base.cycles == [0, 1, 2]
    forced = run_episode(env, base.actions[:1], forced=0)
    assert forced.actions[:2] == [0, 0] and forced.makespan == 102


def test_run_episode_raises_on_masked_prefix_action():
    env = FakeEnv([mask(0, 1), mask(0, 1)])
    with pytest.raises(ReplayDivergence) as e:
        run_episode(env, [0, 5])
    assert e.value.step == 1
    with pytest.raises(ReplayDivergence):
        run_episode(env, [0], forced=NO_OP)


CSV = """state_id,step,cycle_at_step,n_eligible,eligible_lanes,teacher_action,forced_action,makespan,delta_vs_baseline,wall_seconds,error
s1,10,5,2,0 1,0,0,100,0,1,
s1,10,5,2,0 1,0,1,95,-5,1,
s2,20,9,3,0 1 2,1,0,100,0,1,
s2,20,9,3,0 1 2,1,1,100,0,1,
s2,20,9,3,0 1 2,1,2,130,30,1,
s3,30,9,2,3 4,3,3,-1,,1,timeout
s3,30,9,2,3 4,3,4,100,0,1,
"""


def test_analysis_spread_and_regret(tmp_path):
    p = tmp_path / "branches.csv"
    p.write_text(CSV)
    states, failed = analyze.per_state(analyze.load_rows(p))
    by = {s["state_id"]: s for s in states}
    assert failed == 1
    assert (by["s1"]["spread"], by["s1"]["regret"]) == (5, 5)
    assert (by["s2"]["spread"], by["s2"]["regret"]) == (30, 0)
    assert by["s3"]["spread"] is None and by["s3"]["regret"] is None   # one valid branch only
    assert analyze.fractions([5, 30])[">= 20"] == 0.5
    (tmp_path / "baseline.json").write_text('{"makespan": 100}')
    analyze.main(["--dir", str(tmp_path)])
    assert "Verdict" in (tmp_path / "report.md").read_text()
    assert (tmp_path / "spread_hist.png").exists() and (tmp_path / "spread_by_band.png").exists()
