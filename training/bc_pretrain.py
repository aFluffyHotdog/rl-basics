"""Behavior-clone the round-robin teacher into a MaskablePPO policy (masked cross-entropy).

The model is built by train_xsim.build_model (same policy_kwargs / net_arch / features extractor as
PPO), so its weights load straight into train_xsim.py --init-from. Whole episodes are held out;
training stops at >= 99% held-out teacher match or --max-epochs. A plateau below 95% means the
observation does not carry the teacher's state, and the script exits non-zero.

Outputs models/bc_<timestamp>.zip and the frozen policy state dict models/bc_<timestamp>_clone.pt.
"""
import argparse
import copy
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch as th
from torch.nn import functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from stable_baselines3.common.vec_env import DummyVecEnv

from envs.xsim.observation_v3 import build_observation_space_v3
from envs.xsim.observation_v4 import build_observation_space_v4
from training.bc_common import (
    SpaceOnlyEnv,
    batch_to_tensors,
    load_teacher_data,
    masked_argmax,
    policy_logits,
    split_by_episode,
)
from training.train_xsim import build_model

OBS_SPACES = {"v3": build_observation_space_v3, "v4": build_observation_space_v4}
TARGET_MATCH = 0.99
PLATEAU_MATCH = 0.95


def build_bc_model(obs_version: str):
    """A training-identical MaskablePPO on an env that only carries the spaces (no xsim)."""
    space = OBS_SPACES[obs_version]()
    return build_model(DummyVecEnv([lambda: SpaceOnlyEnv(space)]), obs_version)


def bc_epoch(policy, optimizer, data: dict, idx: np.ndarray, batch_size: int, rng,
             fit_value: bool = True, vf_coef: float = 0.5) -> float:
    """One pass of masked cross-entropy (+ value MSE) over idx; returns the mean loss."""
    policy.set_training_mode(True)
    losses = []
    perm = rng.permutation(idx)
    for start in range(0, len(perm), batch_size):
        obs, masks, actions, targets = batch_to_tensors(data, perm[start:start + batch_size], policy.device)
        dist = policy.get_distribution(obs, action_masks=masks)
        loss = -dist.log_prob(actions).mean()
        if fit_value and targets is not None:
            loss = loss + vf_coef * F.mse_loss(policy.predict_values(obs).flatten(), targets)
        optimizer.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(policy.parameters(), 0.5)
        optimizer.step()
        losses.append(loss.item())
    return float(np.mean(losses))


def match_rate(policy, data: dict, idx: np.ndarray, batch_size: int = 4096) -> float:
    """Fraction of steps where the policy's masked argmax equals the teacher's action."""
    policy.set_training_mode(False)
    hits = 0
    with th.no_grad():
        for start in range(0, len(idx), batch_size):
            obs, masks, actions, _ = batch_to_tensors(data, idx[start:start + batch_size], policy.device)
            hits += int((masked_argmax(policy_logits(policy, obs), masks) == actions).sum())
    return hits / max(len(idx), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Behavior-clone the round-robin teacher.")
    parser.add_argument("--data", required=True, help="teacher_<timestamp>.npz from collect_teacher.py")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--holdout", type=float, default=0.1, help="Fraction of episodes held out")
    parser.add_argument("--fit-value", action=argparse.BooleanOptionalAction, default=True,
                        help="Also fit the value head to the (VecNormalize-scaled) return-to-go")
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--patience", type=int, default=10,
                        help="Epochs without held-out improvement that count as a plateau")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="models")
    args = parser.parse_args()

    th.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = load_teacher_data(args.data)
    data["value_targets"] = data.get("returns_norm", data["returns"]) if args.fit_value else None
    obs_version = str(data["obs_version"])
    train_idx, val_idx = split_by_episode(data["episode_id"], args.holdout, args.seed)
    print(f"{len(train_idx)} train / {len(val_idx)} held-out steps, obs {obs_version}")

    model = build_bc_model(obs_version)
    policy = model.policy
    optimizer = th.optim.Adam(policy.parameters(), lr=args.lr)
    best, best_state, since_best = -1.0, None, 0
    for epoch in range(1, args.max_epochs + 1):
        loss = bc_epoch(policy, optimizer, data, train_idx, args.batch_size, rng, args.fit_value, args.vf_coef)
        val = match_rate(policy, data, val_idx)
        print(f"epoch {epoch:3d}: loss {loss:.4f}  held-out teacher match {100 * val:.2f}%")
        if val > best + 1e-4:
            best, best_state, since_best = val, copy.deepcopy(policy.state_dict()), 0
        else:
            since_best += 1
        if best >= TARGET_MATCH or (since_best >= args.patience and best < PLATEAU_MATCH):
            break

    if best < PLATEAU_MATCH:
        print(f"\n>>> BC plateaued at {100 * best:.2f}% held-out match (< {100 * PLATEAU_MATCH:.0f}%). The observation "
              f"probably lacks the teacher's state (cycles_since_fed_i / rr_distance, obs v4) -- not saving.")
        sys.exit(1)

    policy.load_state_dict(best_state)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"bc_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model.save(str(stem))
    th.save(policy.state_dict(), f"{stem}_clone.pt")
    print(f"\n>>> BC held-out teacher match {100 * best:.2f}%"
          f"{'' if best >= TARGET_MATCH else f' (below the {100 * TARGET_MATCH:.0f}% target)'}")
    print(f"Saved {stem}.zip and {stem.name}_clone.pt")


if __name__ == "__main__":
    main()
