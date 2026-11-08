from pathlib import Path

from deployment import infer_best_lane_xsim
from envs.xsim.constants import NO_OP_ACTION


def test_select_best_lane_uses_capacity_and_lowest_index_tie_break():
    assert infer_best_lane_xsim._select_best_lane(
        [4, 7, 7, 2, 0, 0, 0, 0],
        [True, True, True, True, False, False, False, False, False],
    ) == 1


def test_select_best_lane_yields_when_no_lane_is_eligible():
    assert infer_best_lane_xsim._select_best_lane(
        [0] * 8,
        [False] * 8 + [True],
    ) == NO_OP_ACTION


def test_run_inference_picks_most_packable_eligible_lane(tmp_path, monkeypatch):
    sample_dir = tmp_path / "sample"
    beats_dir = sample_dir / "beats_hex"
    rows_dir = sample_dir / "rows_hex"
    beats_dir.mkdir(parents=True)
    rows_dir.mkdir()
    (rows_dir / "sample_subdec0_rows.hex").write_text("0\n", encoding="ascii")
    output_path = tmp_path / "out" / "generated.hex"

    class FakeEnv:
        def __init__(self, data_dir, sim_dir, is_inference):
            self.step_number = 0
            self.actions = []
            self.hw_state = {"step_number": 0}
            self.cmds = []
            self.lb = type("FakeLowerBound", (), {"n_sent": []})()
            self.sim = type("FakeSim", (), {})()
            self.sim.action_file = Path(sim_dir) / "action.txt"
            self.sim.action_file.parent.mkdir(parents=True, exist_ok=True)

        def reset(self):
            return {}, {}

        def action_masks(self):
            if self.step_number == 0:
                return [True, True, False, False, False, False, False, False, False]
            if self.step_number == 1:
                return [True, True, False, False, False, False, False, False, False]
            return [False, False, False, False, False, False, False, False, True]

        def step(self, action):
            self.actions.append(action)
            self.sim.action_file.write_text(f"1 {action:034x}\n", encoding="ascii")
            self.step_number += 1
            self.hw_state["step_number"] = self.step_number
            return {}, 0.0, self.step_number == 3, False, {"makespan": 3}

        def close(self):
            pass

    monkeypatch.setattr(
        infer_best_lane_xsim,
        "packable_slots_by_lane",
        lambda hw_state, cmds, n_sent: (
            [3, 7, 0, 0, 0, 0, 0, 0]
            if hw_state["step_number"] == 0
            else [5, 5, 0, 0, 0, 0, 0, 0]
            if hw_state["step_number"] == 1
            else [0] * 8
        ),
    )
    created_envs = []

    class RecordingEnv(FakeEnv):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created_envs.append(self)

    monkeypatch.setattr(infer_best_lane_xsim, "DecoderEnvXSim", RecordingEnv)

    info = infer_best_lane_xsim.run_inference(
        image_dir=sample_dir,
        sim_dir=tmp_path / "sim",
        output_path=output_path,
    )

    assert created_envs[0].actions == [1, 0, NO_OP_ACTION]
    assert info["makespan"] == 3
    assert len(output_path.read_text(encoding="ascii").splitlines()) == 3
