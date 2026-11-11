"""Callbacks for the behavior-cloning warm start of train_xsim.py (see training/BC_WARMSTART.md)."""
import numpy as np
import torch as th
from stable_baselines3.common.callbacks import BaseCallback

from envs.xsim.constants import NUM_DECODERS
from envs.xsim.observation_v3 import GLOBAL_FEATURES
from policies.round_robin import RoundRobinTeacher, run_teacher_episode
from training.bc_common import ent_coef_at, masked_argmax, parse_ent_schedule, policy_logits

IN_READY = GLOBAL_FEATURES.index("in_ready")   # same index in the v3 and v4 globals


def _count_episodes(infos) -> int:
    return sum("episode" in info for info in infos)   # Monitor adds "episode" at episode end


class EpisodeScheduleCallback(BaseCallback):
    """Counts finished episodes (model.bc_episodes, read by KLMaskablePPO for its beta decay) and
    applies an episode-based entropy schedule ('0:0,100:0.001') to model.ent_coef."""

    def __init__(self, ent_schedule: str | None = None):
        super().__init__()
        self.schedule = parse_ent_schedule(ent_schedule) if ent_schedule else None
        self.episodes = 0

    def _apply(self):
        self.model.bc_episodes = self.episodes
        if self.schedule:
            self.model.ent_coef = ent_coef_at(self.schedule, self.episodes)

    def _on_training_start(self):
        self._apply()

    def _on_step(self) -> bool:
        finished = _count_episodes(self.locals.get("infos", []))
        if finished:
            self.episodes += finished
            self._apply()
        return True

    def _on_rollout_end(self):
        self.logger.record("bc/episodes", self.episodes)
        self.logger.record("bc/ent_coef", float(self.model.ent_coef))


class TeacherMatchCallback(BaseCallback):
    """bc/teacher_match_rate: fraction of rollout steps (with a lane eligible) where the policy's
    greedy action equals the round-robin teacher's pick for the same mask.

    The teacher's cursor follows the actions the agent actually took, so it asks "what would the
    testbench do from here", not "what would it have done from the start". Needs obs v3 / v4.
    """

    def _on_training_start(self):
        self.teachers = [RoundRobinTeacher() for _ in range(self.training_env.num_envs)]
        self.matches = self.total = 0

    def _on_rollout_start(self):
        # Resync with the env's own cursor (an evaluation between rollouts resets the env).
        for teacher, rr in zip(self.teachers, self.training_env.get_attr("rr_cursor")):
            teacher.rr = int(rr)

    def _on_step(self) -> bool:
        obs = self.locals["obs_tensor"]
        masks = np.asarray(self.locals["action_masks"], dtype=bool).reshape(len(self.teachers), -1)
        actions = np.asarray(self.locals["actions"]).reshape(-1)
        with th.no_grad():
            greedy = masked_argmax(policy_logits(self.model.policy, obs),
                                   th.as_tensor(masks, device=self.model.device)).cpu().numpy()
        in_ready = obs["globals"][:, IN_READY].cpu().numpy() > 0.5
        for e, teacher in enumerate(self.teachers):
            if masks[e, :NUM_DECODERS].any():   # forced no-op steps say nothing about the policy
                self.matches += int(greedy[e] == teacher.peek(masks[e]))
                self.total += 1
            teacher.update(actions[e], in_ready[e])
            if self.locals["dones"][e]:
                teacher.reset()
        return True

    def _on_rollout_end(self):
        if self.total:
            self.logger.record("bc/teacher_match_rate", self.matches / self.total)
        self.matches = self.total = 0


class MakespanEvalCallback(BaseCallback):
    """Every `eval_every` training episodes (and once before training), run `n_episodes`
    deterministic episodes on fixed samples (sample_idx 0..n-1) and the teacher on the same samples
    (cached). Logs eval/makespan and eval/makespan_vs_teacher; saves `best_path`.zip () when the mean makespan improves. Selection uses raw makespan (cycles), never reward.

    Evaluation drives the bare env directly (no Monitor, so training stats are untouched) between
    rollouts, then resets the training env; the interrupted training episode is dropped.
    """

    def __init__(self, eval_every: int, n_episodes: int = 2, best_path: str = "models/best"):
        super().__init__()
        self.eval_every = eval_every
        self.n_episodes = n_episodes
        self.best_path = best_path
        self.best = np.inf
        self.episodes = self.last_eval = 0
        self.teacher_makespan = {}

    def _on_training_start(self):
        self._evaluate()

    def _on_step(self) -> bool:
        self.episodes += _count_episodes(self.locals.get("infos", []))
        return True

    def _on_rollout_end(self):
        if self.episodes - self.last_eval >= self.eval_every:
            self.last_eval = self.episodes
            self._evaluate()

    def _policy_episode(self, base, sample_idx: int) -> dict:
        obs, _ = base.reset(options={"sample_idx": sample_idx})
        while True:
            action, _ = self.model.predict(obs, deterministic=True, action_masks=base.action_masks())
            obs, _, terminated, truncated, info = base.step(int(action))
            if terminated or truncated:
                return info

    def _evaluate(self):
        venv = self.model.get_env()
        base = venv.envs[0].unwrapped
        makespans, ratios = [], []
        for k in range(self.n_episodes):
            info = self._policy_episode(base, k)
            name = info.get("sample_name")
            if name not in self.teacher_makespan:
                self.teacher_makespan[name] = run_teacher_episode(base, options={"sample_idx": k}).get("makespan")
            makespan, teacher = info.get("makespan"), self.teacher_makespan[name]
            makespans.append(np.inf if makespan is None else makespan)
            if makespan is not None and teacher:
                ratios.append(makespan / teacher)
            print(f"[eval] episode {self.episodes}: {name} makespan={makespan} teacher={teacher}")

        mean = float(np.mean(makespans))
        self.logger.record("eval/makespan", mean)
        if ratios:
            self.logger.record("eval/makespan_vs_teacher", float(np.mean(ratios)))
        if mean < self.best:
            self.best = mean
            self.model.save(self.best_path)
            print(f"[eval] new best makespan {mean:.1f} -> {self.best_path}.zip")

        # Evaluation moved the env on; restart the training episode from a clean reset.
        self.model._last_obs = venv.reset()
        self.model._last_episode_starts = np.ones((venv.num_envs,), dtype=bool)
