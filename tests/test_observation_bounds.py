"""Every observation feature must sit inside the observation space (no running statistics needed).

Run with: python -m pytest tests/test_observation_bounds.py
The one-episode test needs the Vivado xsim simulator: set XSIM_DIR (simulation dir) to enable it.
"""
import os
from pathlib import Path

import numpy as np
import pytest

from envs.xsim.constants import NUM_ACTIONS, NUM_DECODERS, REWARD_SCALE, TB_INFLIGHT_GATE
from envs.xsim.observation_v3 import build_observation_space_v3, build_observation_v3

SAMPLE_POOL = Path(__file__).resolve().parents[1] / "sample_data_9_lanes"


def assert_in_space(obs: dict, space) -> None:
    for key, box in space.spaces.items():
        value = obs[key]
        assert value.shape == box.shape and value.dtype == box.dtype, (key, value.shape, value.dtype)
        assert np.isfinite(value).all(), key
        assert (value >= box.low).all() and (value <= box.high).all(), (
            f"{key} outside [{box.low.min()}, {box.high.max()}]: min={value.min()} max={value.max()}")


def test_uninitialised_observation_is_in_space():
    # Before the first @OBS line the builder returns zeros.
    class _LB:
        n_sent = None
    obs = build_observation_v3({}, [], _LB(), np.zeros(NUM_ACTIONS, dtype=bool))
    assert_in_space(obs, build_observation_space_v3())


@pytest.mark.skipif(not os.environ.get("XSIM_DIR"), reason="needs the xsim simulator (set XSIM_DIR)")
def test_round_robin_episode_observations_and_reward():
    from envs.xsim.env import DecoderEnvXSim
    from policies.round_robin import RoundRobinTeacher, resolve_data_dir

    path, is_inference = resolve_data_dir(str(SAMPLE_POOL))
    env = DecoderEnvXSim(path, os.environ["XSIM_DIR"], is_inference=is_inference, inflight_gate=TB_INFLIGHT_GATE)
    space, teacher = build_observation_space_v3(), RoundRobinTeacher()
    try:
        obs, _ = env.reset(options={"sample_idx": 0})
        assert_in_space(obs, space)
        total, info = 0.0, {}
        while True:
            in_ready = bool(env.hw_state.get("top_in_ready", 0))
            obs, reward, terminated, truncated, info = env.step(teacher.act(env.action_masks(), in_ready))
            assert_in_space(obs, space)
            total += reward
            if terminated or truncated:
                break
        assert "makespan" in info, info
        assert total == pytest.approx(-info["makespan"] / REWARD_SCALE, abs=1e-6)
    finally:
        env.close()
