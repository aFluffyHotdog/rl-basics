"""Single-lane XSim test: feed one lane of one image alone and check every cycle against the testbench.

Why: feeding a lane alone (experiments/lower_bound/isolated_lane_bound.py) ends in tailbusy_hits
on the lane being fed. This test tells the two possible causes apart:

  1. Python sends a beat while it can already see tail_busy[lane] = 1. The mask blocks this, but
     env.step() -> _pack_beat() -> plan_beat() does not check tail_busy (or the in-flight gate),
     so a caller that bypasses the mask sends it anyway.          -> anomaly "sent_to_busy_tail"
  2. Python and the testbench disagree about the state: the testbench counts a hit on a beat
     Python thought was legal, or counts one on a cycle Python sent nothing, or one resume
     produced more or fewer than one @OBS line (the testbench then re-reads action.txt and drives
     the same beat again), or the RTL popped more CMDs than Python sent.
                                    -> anomalies "hit_unseen_by_python", "hit_without_send",
                                       "obs_lines_not_1", "popped_more_than_sent"

Each step the lane is sent its densest beat when allowed, else a no-op:
  --mode mask    allowed = the env's action mask bit for the lane (tail_busy, in-flight gate, FIFO)
  --mode bypass  allowed = top_in_ready only (what isolated_lane_bound.py does on this branch)
  --mode both    run mask, then bypass, for each lane

The lane is done when its queue is empty, the RTL has popped every CMD sent to it, and all its
FIFOs read 0 (its last row may still be in the executor for a few cycles).

    python deployment/xsim_single_lane_test.py --sim-dir <T8_Decomp_xsim>/sim --sample test_11 --lanes 3
    python deployment/xsim_single_lane_test.py --sim-dir <sim> --sample test_11 --lanes 3 --mode both \
        --trace results/single_lane/trace.csv

Exit status 0 if every run finished with no anomaly and no testbench counter increase, else 1.
"""
import argparse
import csv
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.xsim.constants import (  # noqa: E402
    NO_OP_ACTION, NUM_DECODERS, SUB_DECODER_FIELDS, TB_INFLIGHT_GATE, TB_VIOLATION_KEYS,
)
from envs.xsim.flow_control import cmd_inflight  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "sample_data_9_lanes"
CHECK_KEYS = TB_VIOLATION_KEYS + ("lane_mismatches",)
ANOMALIES = ("sent_to_busy_tail", "hit_unseen_by_python", "hit_without_send",
             "obs_lines_not_1", "popped_more_than_sent")
MODES = ("mask", "bypass")
TRACE_COLUMNS = ("mode", "lane", "cycle", "action", "sent", "groups", "in_ready", "tail_busy_seen",
                 "mask_bit", "inflight_before", "obs_lines", "cmd_popped", "n_sent", "queue_left",
                 "new_counts", "anomalies")


def count_obs_lines(env) -> list:
    """Wrap env.sim.wait_for_obs to count @OBS lines per resume. Returns a 1-item list holding the
    count of the latest call (reset it to 0 before a step)."""
    counter = [0]
    original = env.sim.wait_for_obs

    def counting_wait_for_obs(on_obs):
        def counted(line):
            counter[0] += 1
            on_obs(line)
        return original(counted)

    env.sim.wait_for_obs = counting_wait_for_obs
    return counter


def lane_drained(env, lane: int) -> bool:
    """Queue empty, every CMD sent has been popped, and every FIFO of the lane reads 0."""
    if env.cmds[lane]:
        return False
    popped = env.hw_state.get("cmd_popped")
    if popped is not None and popped[lane] != env.lb.n_sent[lane]:
        return False
    sub = env.hw_state["sub_decoders"][lane]
    return all(sub[f] == 0 for f in SUB_DECODER_FIELDS)


def classify(sent: bool, tail_busy_seen: bool, new_counts: dict, obs_lines: int,
             popped: int | None, n_sent: int) -> list:
    """Anomalies for one step. new_counts = testbench counter increases during this step."""
    found = []
    hits = new_counts.get("tailbusy_hits", 0)
    if sent and tail_busy_seen:
        found.append("sent_to_busy_tail")
    if hits and sent and not tail_busy_seen:
        found.append("hit_unseen_by_python")
    if any(new_counts.get(k, 0) for k in TB_VIOLATION_KEYS) and not sent:
        found.append("hit_without_send")
    if obs_lines != 1:
        found.append("obs_lines_not_1")
    if popped is not None and popped > n_sent:
        found.append("popped_more_than_sent")
    return found


def run_single_lane(env, sample_idx: int, lane: int, mode: str, max_cycles: int,
                    obs_counter: list) -> dict:
    """Feed `lane` alone for one episode. Returns a summary dict with a per-step trace."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    env.reset(options={"sample_idx": sample_idx})
    trace = []
    status = "timeout"
    total_counts = {k: 0 for k in CHECK_KEYS}
    while env.cycles < max_cycles:
        if lane_drained(env, lane):
            status = "ok"
            break
        hw = env.hw_state
        in_ready = bool(hw.get("top_in_ready", 0))
        tail_busy = hw.get("tail_busy")
        tail_busy_seen = bool(tail_busy[lane]) if tail_busy is not None else False
        mask_bit = bool(env.action_masks()[lane])
        allowed = mask_bit if mode == "mask" else in_ready
        action = lane if allowed and env.cmds[lane] else NO_OP_ACTION
        inflight = cmd_inflight(hw, env.lb.n_sent, lane)
        n_sent_before = env.lb.n_sent[lane]
        before = {k: env._delta(k) for k in CHECK_KEYS}

        obs_counter[0] = 0
        _, _, terminated, truncated, info = env.step(action)

        groups = env.lb.n_sent[lane] - n_sent_before
        sent = groups > 0
        new_counts = {k: env._delta(k) - before[k] for k in CHECK_KEYS if env._delta(k) > before[k]}
        for k, v in new_counts.items():
            total_counts[k] += v
        popped = env.hw_state.get("cmd_popped")
        popped_lane = popped[lane] if popped is not None else None
        anomalies = classify(sent, tail_busy_seen, new_counts, obs_counter[0], popped_lane,
                             env.lb.n_sent[lane])
        trace.append({
            "lane": lane, "cycle": env.cycles, "action": action, "sent": int(sent), "groups": groups,
            "in_ready": int(in_ready), "tail_busy_seen": int(tail_busy_seen), "mask_bit": int(mask_bit),
            "inflight_before": inflight, "obs_lines": obs_counter[0], "cmd_popped": popped_lane,
            "n_sent": env.lb.n_sent[lane], "queue_left": len(env.cmds[lane]),
            "new_counts": new_counts, "anomalies": anomalies,
        })
        if info.get("deadlock"):
            status = "deadlock"
            break
        if terminated or truncated:
            status = "ok" if lane_drained(env, lane) else "env_stop"
            break

    anomaly_steps = {a: [t for t in trace if a in t["anomalies"]] for a in ANOMALIES}
    return {
        "lane": lane, "mode": mode, "status": status, "cycles": env.cycles,
        "beats_sent": sum(t["sent"] for t in trace),
        "counter_increases": {k: v for k, v in total_counts.items() if v},
        "anomaly_counts": {a: len(s) for a, s in anomaly_steps.items() if s},
        "first_anomalies": {a: s[:3] for a, s in anomaly_steps.items() if s},
        "trace": trace,
    }


def verdict(result: dict) -> str:
    counts = result["anomaly_counts"]
    sync = [a for a in ANOMALIES[1:] if a in counts]
    if sync:
        return ("Python and the testbench disagree (" + ", ".join(sync) + "): the testbench saw a "
                "beat or a state Python did not. Check the obs_lines column and the raw xsim output "
                "(VIVADO_VERBOSE=True) around the first one.")
    if "sent_to_busy_tail" in counts:
        return ("Every flagged beat was sent while Python could already see tail_busy=1, so the "
                "hits come from bypassing the mask: plan_beat() does not check tail_busy.")
    if result["counter_increases"]:
        return "Testbench counters rose without a classified anomaly; see counter_increases."
    if result["status"] != "ok":
        return f"No anomaly, but the run ended with status {result['status']}."
    return "Clean: no anomaly and no testbench counter increase."


def passed(result: dict) -> bool:
    return result["status"] == "ok" and not result["anomaly_counts"] and not result["counter_increases"]


def print_result(result: dict):
    print(f"\n=== lane {result['lane']}  mode={result['mode']}  status={result['status']}  "
          f"cycles={result['cycles']}  beats_sent={result['beats_sent']}")
    print(f"    testbench counter increases: {result['counter_increases'] or 'none'}")
    print(f"    anomalies: {result['anomaly_counts'] or 'none'}")
    for name, steps in result["first_anomalies"].items():
        for t in steps:
            print(f"      {name} @cycle {t['cycle']}: action={t['action']} sent={t['sent']} "
                  f"tail_busy_seen={t['tail_busy_seen']} mask_bit={t['mask_bit']} "
                  f"inflight_before={t['inflight_before']} obs_lines={t['obs_lines']} "
                  f"popped={t['cmd_popped']} n_sent={t['n_sent']} new={t['new_counts']}")
    print(f"    verdict: {verdict(result)}")


def write_trace(results: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=TRACE_COLUMNS)
        w.writeheader()
        for r in results:
            for t in r["trace"]:
                row = {**t, "mode": r["mode"],
                       "new_counts": ";".join(f"{k}+{v}" for k, v in t["new_counts"].items()),
                       "anomalies": ";".join(t["anomalies"])}
                w.writerow(row)
    print(f"\nTrace saved to {path}")


def find_sample(env, sample: str | None, is_inference: bool) -> tuple[int, str]:
    names = [stem.parent.parent.name if is_inference else stem.relative_to(env.data_dir).parts[0]
             for stem, _ in env.dataset_pool]
    if sample is None:
        return 0, names[0]
    matches = [i for i, n in enumerate(names) if sample in n]
    if not matches:
        raise SystemExit(f"no image matches --sample {sample!r}; available: {names}")
    return matches[0], names[matches[0]]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--sim-dir", required=True, help="xsim simulation directory")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="pool of sample folders (default sample_data_9_lanes), one sample folder, or a beats_hex folder")
    ap.add_argument("--sample", default=None, help="first image whose folder name contains this (default: first image)")
    ap.add_argument("--lanes", type=int, nargs="+", default=list(range(NUM_DECODERS)))
    ap.add_argument("--mode", choices=MODES + ("both",), default="mask")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gate", type=int, default=TB_INFLIGHT_GATE,
                   help=f"in-flight CMD gate used by the mask (default {TB_INFLIGHT_GATE}, the testbench rule)")
    g.add_argument("--no-gate", action="store_true", help="mask uses the CMD FIFO depth only")
    ap.add_argument("--max-cycles", type=int, default=20_000)
    ap.add_argument("--trace", type=Path, default=None, help="write the per-step trace of every run to this CSV")
    args = ap.parse_args(argv)
    if any(not 0 <= lane < NUM_DECODERS for lane in args.lanes):
        ap.error(f"--lanes must be in 0..{NUM_DECODERS - 1}")

    from envs.xsim.env import DecoderEnvXSim
    from policies.round_robin import resolve_data_dir

    path, is_inference = resolve_data_dir(Path(args.data_dir).resolve())
    gate = None if args.no_gate else args.gate
    modes = MODES if args.mode == "both" else (args.mode,)
    env = DecoderEnvXSim(data_dir=path, sim_dir=args.sim_dir, is_inference=is_inference, inflight_gate=gate)
    obs_counter = count_obs_lines(env)
    results = []
    try:
        sample_idx, name = find_sample(env, args.sample, is_inference)
        print(f"image={name}  lanes={args.lanes}  modes={list(modes)}  gate={gate if gate is not None else 'fifo-only'}")
        for lane in args.lanes:
            for mode in modes:
                result = run_single_lane(env, sample_idx, lane, mode, args.max_cycles, obs_counter)
                print_result(result)
                results.append(result)
    finally:
        env.close()

    if args.trace:
        write_trace(results, args.trace)
    failed = [f"lane {r['lane']} {r['mode']}" for r in results if not passed(r)]
    print(f"\n{len(results) - len(failed)}/{len(results)} runs clean" + (f"; not clean: {failed}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
