from deployment import brute_force_find_order
from envs.xsim.constants import NO_OP_ACTION


def test_generate_schedules_yields_each_multiset_order_once():
    assert list(brute_force_find_order.generate_schedules([2, 1])) == [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
    ]


def test_schedule_counts_come_from_per_lane_row_files(tmp_path):
    beats_dir = tmp_path / "sample" / "beats_hex"
    rows_dir = tmp_path / "sample" / "rows_hex"
    beats_dir.mkdir(parents=True)
    rows_dir.mkdir()
    for lane, count in enumerate((3, 1, 2, 0, 0, 0, 0, 0)):
        (rows_dir / f"sample_subdec{lane}_rows.hex").write_text("00\n" * count, encoding="ascii")

    assert brute_force_find_order._schedule_counts(beats_dir) == [3, 1, 2, 0, 0, 0, 0, 0]


def test_read_schedules_supports_comments_and_physical_line_numbers(tmp_path):
    schedule_path = tmp_path / "schedules.txt"
    schedule_path.write_text("# first\n0, 1 0 # second\n\n2\n", encoding="ascii")

    assert list(brute_force_find_order._read_schedules(schedule_path)) == [
        (2, (0, 1, 0)),
        (4, (2,)),
    ]


def test_run_schedule_retries_masked_lane_with_noop(tmp_path):
    class FakeEnv:
        def __init__(self):
            self.step_number = 0
            self.actions = []
            self.cmds = [[1], [1]]
            self.sim = type("FakeSim", (), {})()
            self.sim.action_file = tmp_path / "action.txt"

        def reset(self):
            self.step_number = 0

        def action_masks(self):
            return [self.step_number != 0, True, False]

        def step(self, action):
            self.actions.append(action)
            self.sim.action_file.write_text(f"1 {action:034x}\n", encoding="ascii")
            self.step_number += 1
            if action != NO_OP_ACTION:
                self.cmds[action].clear()
            return {}, 0, self.step_number == 3, False, {}

    env = FakeEnv()
    result = brute_force_find_order._run_schedule(env, [0, 1], max_cycles=10)

    assert env.actions == [NO_OP_ACTION, 0, 1]
    assert result["cycles"] == 3
    assert [action["cycle"] for action in result["actions"]] == [1, 2, 3]