"""Shared pieces of the behavior-cloning warm start: masked logits / KL, schedules, the teacher dataset.

Masks are applied explicitly everywhere: MaskableCategorical only pushes masked logits to -1e8, so
nothing here relies on what the distribution does with them.
"""
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import BaseModel

from envs.xsim.constants import NUM_ACTIONS

OBS_PREFIX = "obs__"   # npz key prefix for the per-key stacked observations


# ---------------------------------------------------------------------- policy outputs
def policy_logits(policy, obs) -> th.Tensor:
    """Unmasked action logits of a MaskableActorCriticPolicy (same path as get_distribution)."""
    features = BaseModel.extract_features(policy, obs, policy.pi_features_extractor)
    return policy.action_net(policy.mlp_extractor.forward_actor(features))


def masked_log_softmax(logits: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """log pi over eligible actions; masked entries are a finite very negative number, never -inf."""
    return th.log_softmax(logits.masked_fill(~mask, th.finfo(logits.dtype).min), dim=-1)


def masked_argmax(logits: th.Tensor, mask: th.Tensor) -> th.Tensor:
    return logits.masked_fill(~mask, th.finfo(logits.dtype).min).argmax(dim=-1)


def masked_kl(logits_p: th.Tensor, logits_q: th.Tensor, mask: th.Tensor) -> th.Tensor:
    """Per-sample KL(p || q) = sum over eligible a of p(a) * (log p(a) - log q(a)).

    Both distributions share the mask; masked actions are dropped from the sum (th.where), so a
    masked term never reaches the result and -inf - -inf cannot occur.
    """
    log_p = masked_log_softmax(logits_p, mask)
    log_q = masked_log_softmax(logits_q, mask)
    terms = th.where(mask, log_p.exp() * (log_p - log_q), th.zeros_like(log_p))
    return terms.sum(dim=-1)


# ---------------------------------------------------------------------- schedules
def kl_beta_at(beta0: float, episodes: int, decay_episodes: int) -> float:
    """Linear decay of the KL weight from beta0 to 0 at decay_episodes."""
    if decay_episodes <= 0:
        return 0.0
    return beta0 * max(0.0, 1.0 - episodes / decay_episodes)


def parse_ent_schedule(text: str) -> list[tuple[int, float]]:
    """'0:0,100:0.001' -> [(0, 0.0), (100, 0.001)] (episode thresholds, sorted)."""
    points = []
    for part in text.split(","):
        episode, value = part.split(":")
        points.append((int(episode), float(value)))
    points.sort()
    if not points or points[0][0] != 0:
        raise ValueError(f"entropy schedule must start at episode 0: {text!r}")
    return points


def ent_coef_at(schedule: list[tuple[int, float]], episodes: int) -> float:
    """Value of the last schedule point whose episode threshold has been reached."""
    value = schedule[0][1]
    for threshold, v in schedule:
        if episodes >= threshold:
            value = v
    return value


# ---------------------------------------------------------------------- data
class SpaceOnlyEnv(gym.Env):
    """Carries the observation / action spaces so a MaskablePPO can be built without xsim."""

    def __init__(self, observation_space: gym.spaces.Dict):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return self.observation_space.sample(), 0.0, True, False, {}

    def action_masks(self) -> np.ndarray:
        return np.ones(NUM_ACTIONS, dtype=bool)


def load_teacher_data(path) -> dict:
    """Load a collect_teacher.py dataset; asserts every teacher action is allowed by its mask."""
    with np.load(path, allow_pickle=False) as npz:
        data = {key: npz[key] for key in npz.files}
    data["obs"] = {key[len(OBS_PREFIX):]: data.pop(key) for key in list(data) if key.startswith(OBS_PREFIX)}
    actions, masks = data["actions"], data["masks"]
    assert masks.dtype == bool and masks.shape == (len(actions), NUM_ACTIONS), masks.shape
    assert masks[np.arange(len(actions)), actions].all(), (
        "a teacher action is masked out -> log_prob would be -inf (was the data collected with the TB mask gate?)")
    return data


def split_by_episode(episode_id: np.ndarray, holdout: float, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Step indices (train, held_out), holding out whole episodes."""
    episodes = np.unique(episode_id)
    if len(episodes) < 2:
        raise ValueError("need at least 2 episodes to hold one out")
    rng = np.random.default_rng(seed)
    rng.shuffle(episodes)
    n_val = max(1, int(round(holdout * len(episodes))))
    is_val = np.isin(episode_id, episodes[:n_val])
    return np.flatnonzero(~is_val), np.flatnonzero(is_val)


def batch_to_tensors(data: dict, idx: np.ndarray, device) -> tuple[dict, th.Tensor, th.Tensor, th.Tensor | None]:
    """(obs, masks, actions, returns) tensors for the given step indices."""
    obs = {key: th.as_tensor(value[idx], dtype=th.float32, device=device) for key, value in data["obs"].items()}
    masks = th.as_tensor(data["masks"][idx], device=device)
    actions = th.as_tensor(data["actions"][idx], dtype=th.int64, device=device)
    returns = data.get("value_targets")
    returns = None if returns is None else th.as_tensor(returns[idx], dtype=th.float32, device=device)
    return obs, masks, actions, returns
