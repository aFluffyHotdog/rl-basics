from deployment import brute_force_find_order
from envs.xsim.constants import NO_OP_ACTION


class FakeEnv:
    """Minimal stand-in for DecoderEnvXSim: one beat per lane pick, one cycle per step unless
    `extra_cycles` says a step also ticked idle cycles (like the real env's skip-until-ready)."""

    def __init__(self, cmds, masked_steps=(), fail_at_step=None, extra_cycles=None):
        self.initial = [list(q) for q in cmds]
        self.masked_steps = set(masked_steps)   # step numbers at which every lane is masked
        self.fail_at_step = fail_at_step
        self.extra_cycles = extra_cycles or {}
        self.sim = type("FakeSim", (), {"write_action": lambda self, content: None})()
        self.lb = type("FakeLB", (), {})()
        self.reset()

    def reset(self):
        self.cmds = [list(q) for q in self.initial]
        self.lb.beats_left = lambda i: len(self.cmds[i]) if i < len(self.cmds) else 0
        self.actions = []
        self.cycles = 0
        self.sim.write_action("path/to/stem")   # reset hands the image stem over first

    def action_masks(self):
        open_ = len(self.actions) not in self.masked_steps
        return [open_ and bool(q) for q in self.cmds] + [True]

    def step(self, action):
        self.actions.append(action)
        for _ in range(1 + self.extra_cycles.get(len(self.actions), 0)):
            self.cycles += 1
            valid = 0 if action == NO_OP_ACTION else 1
            self.sim.write_action(f"{valid} {self.cycles:034x}")
        if self.fail_at_step == len(self.actions):   # e.g. RTL asserts: env ends the episode as a failure
            return {}, -1.0, True, False, {"violation": "rtl_stop"}
        if action != NO_OP_ACTION:
            self.cmds[action].pop()
        return {}, 0, not any(self.cmds), False, {}


def test_generate_schedules_yields_each_multiset_order_once():
    assert list(brute_force_find_order.generate_schedules([2, 1])) == [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
    ]


def test_beats_per_lane_comes_from_the_env_not_the_row_files():
    env = FakeEnv([[1, 1, 1], [1], [1, 1]] + [[]] * 5)
    assert brute_force_find_order._beats_per_lane(env) == [3, 1, 2, 0, 0, 0, 0, 0]


def test_matches_counts_requires_exact_picks_per_lane():
    counts = [2, 1, 0, 0, 0, 0, 0, 0]
    assert brute_force_find_order._matches_counts((0, 1, 0), counts)
    assert not brute_force_find_order._matches_counts((0, 1), counts)
    assert not brute_force_find_order._matches_counts((0, 1, 0, 0), counts)


def test_read_schedules_supports_comments_and_physical_line_numbers(tmp_path):
    schedule_path = tmp_path / "schedules.txt"
    schedule_path.write_text("# first\n0, 1 0 # second\n\n2\n", encoding="ascii")

    assert list(brute_force_find_order._read_schedules(schedule_path)) == [
        (2, (0, 1, 0)),
        (4, (2,)),
    ]


def test_run_schedule_waits_with_noop_while_lane_is_masked():
    env = FakeEnv([[1], [1]] + [[]] * 6, masked_steps={0})
    trace = brute_force_find_order._record_actions(env)

    status, result = brute_force_find_order._run_schedule(env, trace, [0, 1], cycle_limit=10)

    assert status == "ok"
    assert env.actions == [NO_OP_ACTION, 0, 1]
    assert result["cycles"] == 3
    assert [action["cycle"] for action in result["actions"]] == [1, 2, 3]


def test_run_schedule_records_every_cycle_including_skipped_ones():
    # The second step also ticks 2 idle cycles (skip-until-ready): 4 cycles from 2 steps.
    env = FakeEnv([[1], [1]] + [[]] * 6, extra_cycles={1: 2})
    trace = brute_force_find_order._record_actions(env)

    status, result = brute_force_find_order._run_schedule(env, trace, [0, 1], cycle_limit=10)

    assert status == "ok"
    assert result["cycles"] == 4
    assert len(result["actions"]) == 4   # the stem written at reset is not an action


def test_run_schedule_abandons_and_rejects_rtl_stop():
    env = FakeEnv([[1, 1], [1]] + [[]] * 6, fail_at_step=2)
    trace = brute_force_find_order._record_actions(env)

    assert brute_force_find_order._run_schedule(env, trace, [0, 1, 0], cycle_limit=50) == ("failed", None)
    assert env.actions == [0, 1]   # stopped immediately, no further steps


def test_run_schedule_prunes_at_the_cycle_limit():
    env = FakeEnv([[1, 1, 1]] + [[]] * 7)
    trace = brute_force_find_order._record_actions(env)

    assert brute_force_find_order._run_schedule(env, trace, [0, 0, 0], cycle_limit=2) == ("pruned", None)
    assert env.actions == [0, 0]
    # finishing exactly on the limit counts
    assert brute_force_find_order._run_schedule(env, trace, [0, 0, 0], cycle_limit=3)[0] == "ok"
