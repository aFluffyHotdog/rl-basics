import numpy as np
import pytest
import torch as th
from stable_baselines3.common.logger import Logger
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.xsim.constants import NUM_ACTIONS
from envs.xsim.observation_v3 import build_observation_space_v3
from training.bc_common import (
    SpaceOnlyEnv,
    ent_coef_at,
    kl_beta_at,
    masked_kl,
    parse_ent_schedule,
    policy_logits,
)
from training.kl_ppo import KLMaskablePPO
from training.train_xsim import build_policy_kwargs


def _mask(batch=32, seed=0):
    g = th.Generator().manual_seed(seed)
    mask = th.rand(batch, NUM_ACTIONS, generator=g) < 0.5
    mask[:, 0] = True
    return mask


def test_kl_of_a_distribution_with_itself_is_zero():
    logits = th.randn(32, NUM_ACTIONS)
    assert th.allclose(masked_kl(logits, logits, _mask()), th.zeros(32), atol=1e-6)


def test_kl_is_finite_with_masked_actions_and_infinite_logits():
    mask = _mask()
    p = th.randn(32, NUM_ACTIONS, requires_grad=True)
    q = th.randn(32, NUM_ACTIONS).masked_fill(~mask, float("-inf"))
    kl = masked_kl(p, q, mask)
    assert th.isfinite(kl).all() and (kl >= -1e-6).all()
    kl.mean().backward()
    assert th.isfinite(p.grad).all()
    assert th.all(p.grad[~mask] == 0)


def test_kl_ignores_masked_logits():
    mask = _mask()
    p, q = th.randn(32, NUM_ACTIONS), th.randn(32, NUM_ACTIONS)
    q_changed = q.masked_fill(~mask, 123.0)
    assert th.allclose(masked_kl(p, q, mask), masked_kl(p, q_changed, mask))


def test_beta_schedule_reaches_zero_at_decay_episodes():
    assert kl_beta_at(0.1, 0, 500) == pytest.approx(0.1)
    assert kl_beta_at(0.1, 250, 500) == pytest.approx(0.05)
    assert kl_beta_at(0.1, 500, 500) == 0.0
    assert kl_beta_at(0.1, 900, 500) == 0.0


def test_entropy_schedule():
    schedule = parse_ent_schedule("0:0,100:0.001")
    assert ent_coef_at(schedule, 0) == 0.0
    assert ent_coef_at(schedule, 99) == 0.0
    assert ent_coef_at(schedule, 100) == 0.001


def test_kl_ppo_trains_and_logs_kl(tmp_path):
    th.manual_seed(0)
    venv = DummyVecEnv([lambda: SpaceOnlyEnv(build_observation_space_v3())])
    model = KLMaskablePPO("MultiInputPolicy", venv, n_steps=64, batch_size=32, n_epochs=2,
                          policy_kwargs=build_policy_kwargs(), seed=0)
    clone_path = tmp_path / "clone.pt"
    th.save(model.policy.state_dict(), clone_path)
    model.set_kl_prior(str(clone_path), beta=0.1, decay_episodes=10)

    assert not any(p.requires_grad for p in model.kl_clone.parameters())
    records = {}

    class CaptureLogger(Logger):
        def record(self, key, value, exclude=None):
            records[key] = value
            super().record(key, value, exclude)

    model.set_logger(CaptureLogger(folder=None, output_formats=[]))
    model.learn(64)   # one rollout + train
    assert records["bc/beta"] == pytest.approx(0.1)
    assert np.isfinite(records["bc/kl_to_clone"]) and records["bc/kl_to_clone"] >= 0

    model.bc_episodes = 10
    assert model.kl_beta() == 0.0
    obs = {k: th.as_tensor(np.stack([v] * 4)) for k, v in venv.envs[0].observation_space.sample().items()}
    assert th.isfinite(policy_logits(model.kl_clone, obs)).all()
    model.save(tmp_path / "m")   # the frozen clone must not be pickled into the zip
    assert KLMaskablePPO.load(tmp_path / "m").kl_clone is None
