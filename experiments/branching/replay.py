"""Replay an episode prefix, optionally force one action, and let a continuation policy finish.

Measurement harness only: it drives the env through its public API and never edits it.
"""
import copy
import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from policies.round_robin import RoundRobinTeacher  # noqa: E402

NUM_LANES = 8
NO_OP = 8


class ReplayDivergence(RuntimeError):
    """A replayed action was masked: the env no longer follows the recorded trajectory."""

    def __init__(self, step: int, action: int):
        super().__init__(f"replay diverged at step {step}: action {action} is masked")
        self.step, self.action = step, action


@dataclass
class EpisodeResult:
    makespan: int = -1
    actions: list = field(default_factory=list)
    decision_steps: list = field(default_factory=list)   # steps with >= 2 eligible lanes
    masks: list = field(default_factory=list)            # action mask at each step
    cycles: list = field(default_factory=list)           # env cycle counter at each step (before acting)
    in_ready: list = field(default_factory=list)         # top_in_ready at each step (teacher cursor input)
    decision_obs: dict = field(default_factory=dict)     # step -> observation the policy received
    decision_hw: dict = field(default_factory=dict)      # step -> deep copy of env hw_state
    info: dict = field(default_factory=dict)


class MWKRContinuation:
    """Feed the eligible lane with the most rows still unsent (env's lb.rows_unsent); ties -> lowest."""

    def __init__(self, env):
        self.env = env

    def reset(self):
        pass

    def update(self, action, in_ready):
        pass

    def act(self, mask, in_ready=True):
        lanes = np.flatnonzero(np.asarray(mask)[:NUM_LANES])
        if len(lanes) == 0:
            return NO_OP
        lb = self.env.unwrapped.lb
        return int(max(lanes, key=lambda i: (lb.rows_unsent(int(i)), -int(i))))


def make_continuation(name: str, env):
    if name == "mwkr":
        return MWKRContinuation(env)
    return RoundRobinTeacher()


def eligible_lanes(mask) -> list:
    return [int(i) for i in np.flatnonzero(np.asarray(mask)[:NUM_LANES])]


def run_episode(env, prefix_actions=(), forced=None, continuation=None, record_obs=True) -> EpisodeResult:
    """Reset, replay prefix_actions, apply `forced` at the next step, then follow `continuation`.

    The continuation (default: round-robin teacher) is told about every action taken, replayed and
    forced ones included, so its state is what it would be had it chosen them. Closing the env is the
    caller's job (see close_env), so one env can serve many episodes.
    """
    cont = continuation if continuation is not None else RoundRobinTeacher()
    cont.reset()
    base = env.unwrapped
    res = EpisodeResult()
    obs, _ = env.reset()
    prefix = list(prefix_actions)
    step = 0
    while True:
        mask = np.asarray(base.action_masks())
        in_ready = bool(base.hw_state.get("top_in_ready", 0))
        elig = eligible_lanes(mask)
        res.masks.append(mask.copy())
        res.cycles.append(int(base.cycles))
        res.in_ready.append(in_ready)
        if len(elig) >= 2:
            res.decision_steps.append(step)
            if record_obs:
                res.decision_obs[step] = copy.deepcopy(obs)
                res.decision_hw[step] = copy.deepcopy(base.hw_state)

        if step < len(prefix):
            action = int(prefix[step])
            if not mask[action]:
                raise ReplayDivergence(step, action)
            cont.update(action, in_ready)
        elif step == len(prefix) and forced is not None:
            action = int(forced)
            if not mask[action]:
                raise ReplayDivergence(step, action)
            cont.update(action, in_ready)
        else:
            action = int(cont.act(mask, in_ready))

        res.actions.append(action)
        obs, _, terminated, truncated, info = env.step(action)
        step += 1
        if terminated or truncated:
            res.info = info
            if "makespan" not in info:
                raise RuntimeError(f"episode failed without a makespan: {info}")
            res.makespan = int(info["makespan"])
            return res


def kill_xsim(env) -> bool:
    """Make sure the env's xsim process (and its children) is gone. Returns True if it is dead.

    XSimClient.close() ends with os.kill(pid, signal.CTRL_BREAK_EVENT) where `signal` is imported
    from torch, so that fallback always raises (swallowed) and a surviving xsim.bat child is never
    killed. Not fixed in envs/ by design; this kills the whole tree instead.
    """
    proc = getattr(getattr(env.unwrapped, "sim", None), "sim_proc", None)
    if proc is None:
        return True
    pid = proc.pid
    if proc.poll() is None or os.name == "nt":
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], capture_output=True, timeout=10)
            else:
                import signal as os_signal
                os.killpg(os.getpgid(pid), os_signal.SIGKILL)
        except (OSError, subprocess.SubprocessError):
            pass
    try:
        proc.wait(timeout=5)
    except Exception:
        pass
    return proc.poll() is not None


def close_env(env) -> None:
    """env.close() then force-kill whatever is left; never raises."""
    try:
        sim = env.unwrapped.sim
        proc = sim.sim_proc
        env.close()
        # close() nulls sim_proc; restore the handle so kill_xsim can still reach the process tree.
        sim.sim_proc = proc
    except Exception as exc:  # noqa: BLE001
        print(f"[replay] env.close() raised {exc!r}", file=sys.stderr)
    try:
        if not kill_xsim(env):
            print("[replay] WARNING: xsim process still alive after kill", file=sys.stderr)
        env.unwrapped.sim.sim_proc = None
    except Exception as exc:  # noqa: BLE001
        print(f"[replay] kill_xsim raised {exc!r}", file=sys.stderr)


def make_env(data_dir, sim_dir):
    """Teacher env (testbench mask gate). Imported lazily so tests need no xsim/torch."""
    from policies.round_robin import make_teacher_env
    return make_teacher_env(data_dir, sim_dir)
