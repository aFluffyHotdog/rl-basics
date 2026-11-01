"""MaskablePPO with a KL penalty towards a frozen behavior-cloned policy.

loss = PPO loss + beta * KL(pi_theta || pi_clone), per minibatch on the rollout's observations and
action masks (bc_common.masked_kl: eligible actions only). beta decays linearly to 0 over
kl_decay_episodes, counted from Monitor episodes by bc_callbacks.EpisodeScheduleCallback, which
sets `bc_episodes` on the model.

train() is MaskablePPO.train() from sb3-contrib 2.9 with the KL term added; keep it in step with the
installed version if sb3-contrib is upgraded.
"""
import copy

import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance
from torch.nn import functional as F

from training.bc_common import kl_beta_at, masked_kl, policy_logits


class KLMaskablePPO(MaskablePPO):
    kl_clone = None
    kl_beta0 = 0.0
    kl_decay_episodes = 0
    bc_episodes = 0

    def set_kl_prior(self, clone_path: str, beta: float, decay_episodes: int):
        """Load the frozen BC policy (state dict from bc_pretrain.py) into a copy of this policy."""
        clone = copy.deepcopy(self.policy)
        clone.load_state_dict(th.load(clone_path, map_location=self.device))
        clone.to(self.device).eval()
        for param in clone.parameters():
            param.requires_grad_(False)
        self.kl_clone = clone
        self.kl_beta0 = beta
        self.kl_decay_episodes = decay_episodes

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["kl_clone"]

    def kl_beta(self) -> float:
        if self.kl_clone is None:
            return 0.0
        return kl_beta_at(self.kl_beta0, self.bc_episodes, self.kl_decay_episodes)

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        beta = self.kl_beta()
        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        kl_to_clone = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                # --- KL(pi_theta || pi_clone) over eligible actions (the rollout's masks) ---
                if self.kl_clone is not None:
                    masks = rollout_data.action_masks.reshape(len(actions), -1).bool()
                    logits = policy_logits(self.policy, rollout_data.observations)
                    with th.no_grad():
                        clone_logits = policy_logits(self.kl_clone, rollout_data.observations)
                    kl = masked_kl(logits, clone_logits, masks).mean()
                    kl_to_clone.append(kl.item())
                    if beta > 0:
                        loss = loss + beta * kl

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
        if kl_to_clone:
            self.logger.record("bc/kl_to_clone", np.mean(kl_to_clone))
        self.logger.record("bc/beta", beta)
