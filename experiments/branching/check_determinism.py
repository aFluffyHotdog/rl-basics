"""Is the env deterministic under the round-robin teacher?  PASS/FAIL, with a noise estimate on FAIL.

  python -m experiments.branching.check_determinism --data-dir <sample> --sim-dir <sim>
"""
import argparse
import statistics
import sys

import numpy as np

from experiments.branching.replay import close_env, make_env, run_episode


def same(a, b) -> list:
    """Names of the quantities that differ between two EpisodeResults."""
    bad = []
    if a.makespan != b.makespan:
        bad.append(f"makespan {a.makespan} vs {b.makespan}")
    if a.actions != b.actions:
        bad.append("actions")
    if len(a.masks) != len(b.masks) or not all(np.array_equal(x, y) for x, y in zip(a.masks, b.masks)):
        bad.append("masks")
    if a.cycles != b.cycles:
        bad.append("per-step cycles")
    return bad


def check(env, n_runs=3) -> tuple[bool, list, list]:
    runs = [run_episode(env, record_obs=False) for _ in range(n_runs)]
    problems = []
    for i in range(1, n_runs):
        problems += [f"run {i} vs run 0: {p}" for p in same(runs[0], runs[i])]
    half = len(runs[0].actions) // 2
    replay = run_episode(env, runs[0].actions[:half], record_obs=False)
    problems += [f"half-prefix replay vs run 0: {p}" for p in same(runs[0], replay)]
    return not problems, problems, runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--noise-runs", type=int, default=5)
    args = ap.parse_args(argv)
    env = make_env(args.data_dir, args.sim_dir)
    try:
        ok, problems, runs = check(env)
        print(f"baseline makespans: {[r.makespan for r in runs]}  steps: {len(runs[0].actions)}")
        if ok:
            print("DETERMINISM: PASS")
            return 0
        print("DETERMINISM: FAIL")
        for p in problems:
            print("  -", p)
        print(f"Measuring noise over {args.noise_runs} more baseline episodes; NOT continuing the experiment.")
        spans = [run_episode(env, record_obs=False).makespan for _ in range(args.noise_runs)]
        print(f"noise makespans: {spans}  std = {statistics.pstdev(spans):.3f} cycles")
        return 1
    finally:
        close_env(env)


if __name__ == "__main__":
    sys.exit(main())
