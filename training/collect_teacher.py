"""Roll the round-robin teacher through the PPO training env and save a behavior-cloning dataset.

The env is the same wrapped stack train_xsim.py builds (ActionMasker -> Monitor -> DummyVecEnv ->
VecNormalize) with the testbench mask gate on, so masks and observations match what PPO will see.

Saved to <out-dir>/teacher_<timestamp>.npz:
  obs__<key>    observations stacked per Dict key       masks        bool [N, 9]
  actions       int64 teacher actions                   episode_id   int64
  returns       undiscounted return-to-go of the raw env reward
  returns_norm  same, of the reward scaled by the final VecNormalize statistics (what PPO's value
                head is trained on)
  makespans / sample_names   one per episode;  inflight_gate   collection setting
plus the VecNormalize statistics in teacher_<timestamp>_vecnormalize.pkl (pass it to bc_pretrain.py
and to train_xsim.py --vecnorm-stats). Steps where no-op is the only eligible action are skipped.
"""
import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sb3_contrib.common.maskable.utils import get_action_masks

from envs.xsim.constants import NUM_DECODERS, TB_INFLIGHT_GATE
from policies.round_robin import RoundRobinTeacher, resolve_data_dir
from training.bc_common import OBS_PREFIX
from training.train_xsim import make_vec_env


def return_to_go(rewards) -> np.ndarray:
    return np.cumsum(np.asarray(rewards, dtype=np.float64)[::-1])[::-1]


def collect(data_dir, sim_dir, episodes: int, out_dir="data", seed: int = 0) -> Path:
    path, is_inference = resolve_data_dir(data_dir)
    venv = make_vec_env(path, sim_dir, inflight_gate=TB_INFLIGHT_GATE, is_inference=is_inference)
    venv.seed(seed)
    teacher = RoundRobinTeacher()

    obs_buf, masks, actions, episode_id = {}, [], [], []
    raw_by_episode, kept_by_episode = [], []   # per episode: every raw reward / kept step indices
    makespans, sample_names = [], []
    ep_raw, ep_kept = [], []
    try:
        obs = venv.reset()
        while len(makespans) < episodes:
            mask = get_action_masks(venv)[0]
            in_ready = bool(venv.get_attr("hw_state")[0].get("top_in_ready", 0))
            action = teacher.act(mask, in_ready)
            if mask[:NUM_DECODERS].any():   # forced no-op steps carry no information
                for key, value in obs.items():
                    obs_buf.setdefault(key, []).append(value[0].copy())
                masks.append(mask.copy())
                actions.append(action)
                episode_id.append(len(makespans))
                ep_kept.append(len(ep_raw))

            obs, _, dones, infos = venv.step(np.array([action]))
            ep_raw.append(float(venv.get_original_reward()[0]))
            if dones[0]:
                info = infos[0]
                if "makespan" not in info:
                    raise RuntimeError(f"teacher episode {len(makespans)} failed: {info}")
                makespans.append(int(info["makespan"]))
                sample_names.append(str(info.get("sample_name", "")))
                raw_by_episode.append(ep_raw)
                kept_by_episode.append(ep_kept)
                print(f"episode {len(makespans) - 1}: {sample_names[-1]} makespan={makespans[-1]} "
                      f"kept {len(ep_kept)}/{len(ep_raw)} steps")
                ep_raw, ep_kept = [], []
                teacher.reset()

        returns, returns_norm = [], []
        for raw, kept in zip(raw_by_episode, kept_by_episode):
            returns.append(return_to_go(raw)[kept])
            returns_norm.append(return_to_go(venv.normalize_reward(np.asarray(raw)))[kept])

        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = out_dir / f"teacher_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        np.savez_compressed(
            f"{stem}.npz",
            **{OBS_PREFIX + key: np.stack(values) for key, values in obs_buf.items()},
            masks=np.asarray(masks, dtype=bool),
            actions=np.asarray(actions, dtype=np.int64),
            episode_id=np.asarray(episode_id, dtype=np.int64),
            returns=np.concatenate(returns).astype(np.float32),
            returns_norm=np.concatenate(returns_norm).astype(np.float32),
            makespans=np.asarray(makespans, dtype=np.int64),
            sample_names=np.asarray(sample_names),
            inflight_gate=np.asarray(TB_INFLIGHT_GATE),
        )
        venv.save(f"{stem}_vecnormalize.pkl")
    finally:
        venv.close()

    print(f"\n>>> TEACHER MAKESPAN: mean = {np.mean(makespans):.1f} cycles over {len(makespans)} episodes")
    print(f"Saved {len(actions)} steps to {stem}.npz (+ {stem.name}_vecnormalize.pkl)")
    return Path(f"{stem}.npz")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect a behavior-cloning dataset from the round-robin teacher.")
    parser.add_argument("--data-dir", required=True,
                        help="Sample folder, its beats_hex/ folder, or a pool of sample folders.")
    parser.add_argument("--sim-dir", required=True, help="Path to the XSim simulation directory.")
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--seed", type=int, default=0, help="Sample selection seed for the dataset pool.")
    args = parser.parse_args()
    collect(args.data_dir, args.sim_dir, args.episodes, args.out_dir, args.seed)


if __name__ == "__main__":
    main()
