"""Round-robin teacher: the v20 testbench packer's lane choice (deployment/testbench_but_python.py).

The testbench keeps a cursor over 9 channels (8 lanes + the disabled bypass slot) and serves the
first eligible lane at or after it; the cursor then moves past the served lane, steps by one on an
idle cycle with the bus ready, and holds while top_in_ready is low (tb_rr_pick / tb_rr_advance in
envs/xsim/flow_control.py, shared with the testbench script).

Eligibility is the env's action mask. Run the env with inflight_gate=TB_INFLIGHT_GATE so the mask
is exactly the testbench's eligibility (tail not busy, <= 8 CMDs in flight); the teacher then only
picks no-op when the mask allows nothing else.

  python -m policies.round_robin --data-dir <sample dir | beats_hex dir | pool dir> --sim-dir <sim> --episodes 3
"""
import argparse
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.xsim.constants import NO_OP_ACTION, NUM_DECODERS, TB_INFLIGHT_GATE
from envs.xsim.flow_control import tb_rr_advance, tb_rr_pick


class RoundRobinTeacher:
    def __init__(self):
        self.reset()

    def reset(self):
        self.rr = 0   # testbench pk_rr: the first pick is the first eligible lane from 0

    def peek(self, mask) -> int:
        """Action the teacher would take for this mask, without moving the cursor."""
        if not np.any(mask[:NUM_DECODERS]):
            return NO_OP_ACTION
        return tb_rr_pick(self.rr, mask[:NUM_DECODERS])

    def update(self, action: int, in_ready: bool):
        """Move the cursor as the testbench would after `action` was taken (by anyone)."""
        self.rr = tb_rr_advance(self.rr, int(action), bool(in_ready))

    def act(self, mask, in_ready: bool = True) -> int:
        action = self.peek(mask)
        self.update(action, in_ready)
        return action


def resolve_data_dir(data_dir) -> tuple[str, bool]:
    """(data_dir, is_inference) for DecoderEnvXSim: a sample folder or a beats_hex folder is one
    image (inference mode); anything else is a pool of sample folders."""
    path = Path(data_dir).resolve()
    if (path / "beats_hex").is_dir():
        return str(path / "beats_hex"), True
    if any(path.glob("*.hex")):
        return str(path), True
    return str(path), False


def make_teacher_env(data_dir, sim_dir, obs_version: str = "v4"):
    """Bare DecoderEnvXSim with the testbench mask gate."""
    from envs.xsim.env import DecoderEnvXSim

    path, is_inference = resolve_data_dir(data_dir)
    return DecoderEnvXSim(data_dir=path, sim_dir=sim_dir, is_inference=is_inference,
                          obs_version=obs_version, inflight_gate=TB_INFLIGHT_GATE)


def run_teacher_episode(env, teacher: RoundRobinTeacher | None = None, options: dict | None = None) -> dict:
    """One episode on a DecoderEnvXSim (or a gym wrapper of it) -> its final info dict."""
    teacher = teacher or RoundRobinTeacher()
    base = env.unwrapped
    teacher.reset()
    env.reset(options=options)
    while True:
        in_ready = bool(base.hw_state.get("top_in_ready", 0))
        _, _, terminated, truncated, info = env.step(teacher.act(base.action_masks(), in_ready))
        if terminated or truncated:
            return info


def evaluate_teacher(env, n_episodes: int) -> list[int]:
    """Teacher makespans (cycles) for n_episodes; a failed episode raises."""
    teacher = RoundRobinTeacher()
    makespans = []
    for ep in range(n_episodes):
        info = run_teacher_episode(env, teacher)
        if "makespan" not in info:
            raise RuntimeError(f"teacher episode {ep} failed on {info.get('sample_name')}: {info}")
        makespans.append(int(info["makespan"]))
        print(f"episode {ep}: sample={info.get('sample_name')} makespan={info['makespan']} cycles")
    return makespans


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the testbench round-robin teacher in XSim.")
    parser.add_argument("--data-dir", required=True,
                        help="Sample folder, its beats_hex/ folder, or a pool of sample folders.")
    parser.add_argument("--sim-dir", required=True, help="Path to the XSim simulation directory.")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--obs-version", default="v4", choices=("v3", "v4"))
    parser.add_argument("--expected", type=float, default=None,
                        help="SV testbench makespan; exit non-zero if the mean differs by more than 1%%.")
    args = parser.parse_args()

    env = make_teacher_env(args.data_dir, args.sim_dir, args.obs_version)
    try:
        makespans = evaluate_teacher(env, args.episodes)
    finally:
        env.close()

    mean = float(np.mean(makespans))
    print(f"\n>>> TEACHER MAKESPAN: mean = {mean:.1f} cycles over {len(makespans)} episodes {makespans}")
    if args.expected is not None:
        diff = abs(mean - args.expected) / args.expected
        print(f">>> vs SV testbench {args.expected:.0f}: {100 * diff:.2f}% off "
              f"({'OK' if diff <= 0.01 else 'MISMATCH: env and testbench disagree'})")
        if diff > 0.01:
            sys.exit(1)


if __name__ == "__main__":
    main()
