"""Decision-branching experiment: force each eligible lane at sampled decision points, finish with a
continuation policy, record the makespan.

  python -m experiments.branching.run_branching --data-dir <sample> --sim-dir <sim> --n-states 50 --workers 4
"""
import argparse
import csv
import json
import multiprocessing as mp
import multiprocessing.util  # noqa: F401  (mp.util.Finalize)
import shutil
import sys
import threading
import time
from pathlib import Path

import numpy as np

from experiments.branching.replay import (
    close_env, eligible_lanes, kill_xsim, make_continuation, make_env, run_episode,
)

COLUMNS = ["state_id", "step", "cycle_at_step", "n_eligible", "eligible_lanes", "teacher_action",
           "forced_action", "makespan", "delta_vs_baseline", "wall_seconds", "error"]
N_BANDS = 5

_W = {}   # per-worker-process globals: env, config


# ----------------------------------------------------------------------------- selection
def state_id(step: int) -> str:
    return f"s{step:05d}"


def select_states(decision_steps, n_states, seed, explicit=None) -> list:
    """Stratified sample: split decision steps into N_BANDS equal bands by index, sample evenly."""
    if explicit:
        bad = [s for s in explicit if s not in set(decision_steps)]
        if bad:
            raise SystemExit(f"--states {bad} are not decision steps of the baseline episode")
        return sorted(explicit)
    rng = np.random.default_rng(seed)
    bands = np.array_split(np.asarray(decision_steps), N_BANDS)
    chosen = []
    for b, band in enumerate(bands):
        k = n_states // N_BANDS + (1 if b < n_states % N_BANDS else 0)
        chosen += rng.choice(band, size=min(k, len(band)), replace=False).tolist()
    return sorted(int(s) for s in chosen)


# ----------------------------------------------------------------------------- worker
def _init_worker(id_queue, data_dir, sim_dirs_root, base_sim_dir, continuation, timeout):
    wid = id_queue.get()
    sim_dir = Path(sim_dirs_root) / f"w{wid}"
    if not sim_dir.exists():   # own run dir per worker: the env runs xsim in <sim_dir>/xsim_run/run
        shutil.copytree(base_sim_dir, sim_dir)
    mp.util.Finalize(None, _close_worker_env, exitpriority=10)   # runs on normal worker exit
    _W.update(env=None, data_dir=data_dir, sim_dir=str(sim_dir), continuation=continuation, timeout=timeout)


def _close_worker_env():
    if _W.get("env") is not None:
        close_env(_W["env"])
        _W["env"] = None


def _branch(task):
    """Run one branch -> CSV row dict. Never raises; a timeout kills xsim and yields makespan -1."""
    meta, prefix, forced, baseline_makespan = task
    t0 = time.time()
    row = dict(meta, forced_action=forced, makespan=-1, delta_vs_baseline="", error="")
    timed_out = threading.Event()
    timer = None
    try:
        if _W["env"] is None:
            _W["env"] = make_env(_W["data_dir"], _W["sim_dir"])
        env = _W["env"]

        def on_timeout():
            timed_out.set()
            kill_xsim(env)   # blocked readline() sees EOF -> RuntimeError below
        timer = threading.Timer(_W["timeout"], on_timeout)
        timer.start()
        res = run_episode(env, prefix, forced=forced, continuation=make_continuation(_W["continuation"], env),
                          record_obs=False)
        row["makespan"] = res.makespan
        row["delta_vs_baseline"] = res.makespan - baseline_makespan
        if (_W["continuation"] == "rr" and forced == meta["teacher_action"]
                and res.makespan != baseline_makespan):
            row["error"] = f"teacher branch {res.makespan} != baseline {baseline_makespan}"
    except Exception as exc:  # noqa: BLE001
        row["makespan"] = -1
        row["error"] = f"timeout after {_W['timeout']}s" if timed_out.is_set() else f"{type(exc).__name__}: {exc}"
        if _W["env"] is not None:
            close_env(_W["env"])
        _W["env"] = None   # rebuild on the next branch
    finally:
        if timer:
            timer.cancel()
    row["wall_seconds"] = round(time.time() - t0, 1)
    return row


# ----------------------------------------------------------------------------- baseline
def run_baseline(args, out: Path, want_steps=None):
    env = make_env(args.data_dir, args.sim_dir)
    try:
        res = run_episode(env, record_obs=True)
    finally:
        close_env(env)
    data = {"makespan": res.makespan, "actions": res.actions, "decision_steps": res.decision_steps,
            "masks": [[int(v) for v in m] for m in res.masks], "cycles": res.cycles, "in_ready": res.in_ready}
    (out / "baseline.json").write_text(json.dumps(data))
    return data, res


def save_state_data(out: Path, steps, res):
    arrays = {}
    for s in steps:
        for key, val in res.decision_obs[s].items():
            arrays[f"{state_id(s)}__{key}"] = np.asarray(val)
    np.savez(out / "obs.npz", **arrays)
    with open(out / "hw_states.jsonl", "w") as f:
        for s in steps:
            f.write(json.dumps({"state_id": state_id(s), "step": s, "hw_state": res.decision_hw[s]},
                               default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


# ----------------------------------------------------------------------------- main
def read_done(csv_path: Path) -> set:
    if not csv_path.exists():
        return set()
    with open(csv_path, newline="") as f:
        return {(r["state_id"], int(r["forced_action"])) for r in csv.DictReader(f) if not r["error"]}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--sim-dir", required=True)
    ap.add_argument("--out-dir", default=None, help="default results/branching (results/branching_mwkr for mwkr)")
    ap.add_argument("--n-states", type=int, default=50)
    ap.add_argument("--states", type=int, nargs="+", default=None, help="explicit decision-step indices")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=180.0)
    ap.add_argument("--continuation", choices=["rr", "mwkr"], default="rr")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args(argv)

    out = Path(args.out_dir or ("results/branching" if args.continuation == "rr" else "results/branching_mwkr"))
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "branches.csv"

    # Step 1: baseline (always the round-robin teacher: it defines the prefixes)
    base_path = out / "baseline.json"
    if args.resume and base_path.exists() and (out / "obs.npz").exists():
        base = json.loads(base_path.read_text())
        res = None
        print(f"[resume] baseline makespan {base['makespan']}")
    else:
        print("[baseline] running teacher episode ...")
        base, res = run_baseline(args, out)
        print(f"[baseline] makespan {base['makespan']}, {len(base['actions'])} steps, "
              f"{len(base['decision_steps'])} decision points")

    # Step 2: states
    steps = select_states(base["decision_steps"], args.n_states, args.seed, args.states)
    if res is not None:
        save_state_data(out, steps, res)
    print(f"[states] {len(steps)} decision points: {steps}")

    # Step 3: branches
    done = read_done(csv_path) if args.resume else set()
    tasks = []
    for s in steps:
        elig = eligible_lanes(base["masks"][s])
        meta = dict(state_id=state_id(s), step=s, cycle_at_step=base["cycles"][s], n_eligible=len(elig),
                    eligible_lanes=" ".join(map(str, elig)), teacher_action=base["actions"][s])
        for a in elig:
            if (meta["state_id"], a) not in done:
                tasks.append((meta, base["actions"][:s], a, base["makespan"]))
    print(f"[branches] {len(tasks)} to run ({len(done)} already done)")
    if not tasks:
        return 0

    new_file = not csv_path.exists() or not args.resume
    errors, n_done, t_start = 0, 0, time.time()
    ctx = mp.get_context("spawn")
    ids = ctx.Queue()
    for i in range(args.workers):
        ids.put(i)
    pool = ctx.Pool(args.workers, _init_worker, (ids, args.data_dir, str(out / "workers"), args.sim_dir,
                                                  args.continuation, args.timeout))
    try:
        with open(csv_path, "w" if new_file else "a", newline="") as f:
            w = csv.DictWriter(f, COLUMNS)
            if new_file:
                w.writeheader()
            for row in pool.imap_unordered(_branch, tasks):
                w.writerow(row)
                f.flush()
                n_done += 1
                if row["error"]:
                    errors += 1
                    print(f"[ERROR] {row['state_id']} lane {row['forced_action']}: {row['error']}")
                if n_done == min(4, len(tasks)) and len(tasks) > 4:
                    per = (time.time() - t_start) / n_done
                    print(f"[eta] first {n_done} branches took {time.time() - t_start:.0f}s; "
                          f"~{per * (len(tasks) - n_done) / 60:.0f} min left")
        pool.close()   # graceful: each worker closes its env and kills its xsim tree on exit
        pool.join()
    except BaseException:
        pool.terminate()   # workers cannot clean up when terminated
        print("[WARNING] interrupted: check for stray xsim processes (taskkill /IM xsim.exe /F)", file=sys.stderr)
        raise
    print(f"[done] {n_done} branches, {errors} with errors -> {csv_path}")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
