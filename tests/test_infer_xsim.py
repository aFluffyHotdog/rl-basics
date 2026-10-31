from pathlib import Path

import pytest

from deployment import infer_xsim


class FakeEnv:
    def __init__(self, data_dir, sim_dir, is_inference, obs_version):
        self.sim = type("FakeSim", (), {})()
        self.sim.action_file = Path(sim_dir) / "action.txt"
        self.sim.action_file.parent.mkdir(parents=True, exist_ok=True)
        self.steps = 0
        self.closed = False

    def reset(self):
        return {"obs": 0}, {}

    def action_masks(self):
        return [True, False]

    def step(self, action):
        self.steps += 1
        self.sim.action_file.write_text(f"1 {action:034x}\n", encoding="ascii")
        return {"obs": self.steps}, 0.0, self.steps == 2, False, {"makespan": 2}

    def close(self):
        self.closed = True


class FakeModel:
    def __init__(self):
        self.calls = 0

    def predict(self, obs, deterministic, action_masks):
        assert deterministic is True
        assert action_masks.tolist() == [True, False]
        self.calls += 1
        return self.calls, None


def _make_sample(root: Path) -> Path:
    sample_dir = root / "sample"
    beats_dir = sample_dir / "beats_hex"
    rows_dir = sample_dir / "rows_hex"
    beats_dir.mkdir(parents=True)
    rows_dir.mkdir()
    (beats_dir / "sample_subdec0_beats.hex").write_text("0\n", encoding="ascii")
    (rows_dir / "sample_subdec0_rows.hex").write_text("0\n", encoding="ascii")
    return sample_dir


def test_run_inference_exports_dispatched_records(tmp_path, monkeypatch):
    sample_dir = _make_sample(tmp_path)
    model_path = tmp_path / "model.zip"
    model_path.touch()
    output_path = tmp_path / "out" / "generated.hex"
    fake_model = FakeModel()

    monkeypatch.setattr(infer_xsim, "DecoderEnvXSim", FakeEnv)
    monkeypatch.setattr(infer_xsim.MaskablePPO, "load", lambda path: fake_model)

    info = infer_xsim.run_inference(
        model_path=model_path,
        image_dir=sample_dir,
        sim_dir=tmp_path / "sim",
        output_path=output_path,
    )

    assert output_path.read_text(encoding="ascii").splitlines() == [
        f"{1:034x}",
        f"{2:034x}",
    ]
    assert info["makespan"] == 2
    assert fake_model.calls == 2


def test_run_inference_requires_preprocessed_rows_hex(tmp_path):
    sample_dir = tmp_path / "sample"
    (sample_dir / "beats_hex").mkdir(parents=True)
    model_path = tmp_path / "model.zip"
    model_path.touch()

    with pytest.raises(ValueError, match="row hex"):
        infer_xsim.run_inference(
            model_path=model_path,
            image_dir=sample_dir,
            sim_dir=tmp_path / "sim",
            output_path=tmp_path / "out.hex",
        )
