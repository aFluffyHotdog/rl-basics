import numpy as np
import pytest
import torch as th

from envs.xsim.constants import NO_OP_ACTION, NUM_ACTIONS, NUM_DECODERS
from envs.xsim.observation_v4 import LANE_FEATURES
from training.bc_common import OBS_PREFIX, load_teacher_data, split_by_episode
from training.bc_pretrain import bc_epoch, build_bc_model, match_rate

RR = LANE_FEATURES.index("rr_distance")


def _synthetic(n_episodes=2, steps=64, seed=0) -> dict:
    """Teacher-like data: action = eligible lane with the smallest rr_distance."""
    rng = np.random.default_rng(seed)
    n = n_episodes * steps
    lanes = rng.random((n, NUM_DECODERS, len(LANE_FEATURES))).astype(np.float32)
    globals_ = rng.random((n, 6)).astype(np.float32)
    masks = np.zeros((n, NUM_ACTIONS), dtype=bool)
    actions = np.zeros(n, dtype=np.int64)
    for k in range(n):
        cursor = rng.integers(0, NUM_DECODERS + 1)
        lanes[k, :, RR] = ((np.arange(NUM_DECODERS) - cursor) % (NUM_DECODERS + 1)) / NUM_DECODERS
        eligible = rng.random(NUM_DECODERS) < 0.5
        eligible[rng.integers(0, NUM_DECODERS)] = True
        masks[k, :NUM_DECODERS] = eligible
        lane_ids = np.flatnonzero(eligible)
        actions[k] = lane_ids[np.argmin(lanes[k, lane_ids, RR])]
    return {
        "obs": {"lanes": lanes, "globals": globals_},
        "masks": masks,
        "actions": actions,
        "episode_id": np.repeat(np.arange(n_episodes), steps),
        "value_targets": None,
    }


def test_masked_actions_get_zero_probability():
    th.manual_seed(0)
    policy = build_bc_model("v4").policy
    data = _synthetic()
    obs = {k: th.as_tensor(v) for k, v in data["obs"].items()}
    masks = th.as_tensor(data["masks"])
    probs = policy.get_distribution(obs, action_masks=masks).distribution.probs
    assert th.all(probs[~masks] == 0)
    log_prob = policy.get_distribution(obs, action_masks=masks).log_prob(th.as_tensor(data["actions"]))
    assert th.isfinite(log_prob).all()


def test_two_episode_dataset_overfits_within_20_epochs():
    th.manual_seed(0)
    policy = build_bc_model("v4").policy
    data = _synthetic()
    idx = np.arange(len(data["actions"]))
    optimizer = th.optim.Adam(policy.parameters(), lr=1e-3)
    rng = np.random.default_rng(0)
    for _ in range(20):
        bc_epoch(policy, optimizer, data, idx, batch_size=16, rng=rng, fit_value=False)
        if match_rate(policy, data, idx) == 1.0:
            break
    assert match_rate(policy, data, idx) == 1.0


def test_loader_rejects_masked_teacher_actions(tmp_path):
    data = _synthetic(steps=4)
    data["actions"][0] = NO_OP_ACTION       # masked out in every synthetic step
    path = tmp_path / "bad.npz"
    np.savez(path, **{OBS_PREFIX + k: v for k, v in data["obs"].items()},
             masks=data["masks"], actions=data["actions"], episode_id=data["episode_id"])
    with pytest.raises(AssertionError, match="masked out"):
        load_teacher_data(path)


def test_holdout_is_by_episode():
    episode_id = np.repeat(np.arange(10), 5)
    train, val = split_by_episode(episode_id, 0.1)
    assert len(val) == 5 and len(set(episode_id[val])) == 1
    assert not set(episode_id[train]) & set(episode_id[val])
