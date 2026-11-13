"""Measured makespan lower bound per image: max(bus0, longest isolated lane).

For every image and every lane, the lane is fed ALONE on an idle bus, as fast as flow control
allows, and timed until it has output all NUM_ROWS rows. No schedule can finish that lane sooner
when it has to share the bus, so the longest isolated lane is a per-image floor. bus0 (the dense
beats every lane needs, summed) is the bus floor: one beat per cycle. The bound is the larger one.

    bound = max(bus0, max over lanes of isolated_cycles(lane))

"As fast as flow control allows" = every cycle the lane's action-mask bit is set, send it the
densest beat the env packer builds (the same packer and mask the training env uses). With
--gate 8 (default, TB_INFLIGHT_GATE) that is the v20 testbench feed rule. Assumption: greedy
dense feeding is the fastest a lane can drain.

Both backends run the same code through DecoderEnvXSim:

    python -m experiments.lower_bound.isolated_lane_bound                          # verilator
    python -m experiments.lower_bound.isolated_lane_bound --backend xsim --sim-dir <sim>
    python -m experiments.lower_bound.isolated_lane_bound --samples test_14 test_16 --no-rr
    python -m experiments.lower_bound.isolated_lane_bound --data-dir <T8_Decomp_xsim>/sim/decomp_test/4_comp_data

--data-dir takes a pool of sample folders (sample_data_9_lanes, the default), one sample folder,
one beats_hex folder, or a flat generator output folder (<image>_subdec<s>_{beats,rows}.hex side
by side), which is copied into a temporary pool first.

Writes results/lower_bound/<data dir>_<backend>_gate<g>.json (per-lane detail) and .csv (one row
per image).
"""
import argparse
import csv
import glob
import json
import shutil
import sys
import tempfile
from pathlib import Path

from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from envs.xsim.constants import (  # noqa: E402
    NO_OP_ACTION, NUM_DECODERS, NUM_ROWS, TB_INFLIGHT_GATE, TB_VIOLATION_KEYS,
)
from envs.xsim.env import DecoderEnvXSim  # noqa: E402
from envs.xsim.protocol import IDLE_ACTION, encode_beat  # noqa: E402
from policies.round_robin import RoundRobinTeacher, resolve_data_dir, run_teacher_episode  # noqa: E402

DEFAULT_DATA_DIR = REPO_ROOT / "sample_data_9_lanes"
DEFAULT_OUT_DIR = REPO_ROOT / "results" / "lower_bound"
CHECK_KEYS = TB_VIOLATION_KEYS + ("lane_mismatches",)


def flat_images(data_dir: Path) -> list:
    """Image prefixes in a flat generator output folder (<image>_subdec<s>_beats.hex / _rows.hex /
    _meta.txt side by side, e.g. T8_Decomp_xsim/sim/decomp_test/4_comp_data). Empty if not flat."""
    # rsplit: an image name can itself contain "_subdec" (test_40_subdec_cols_subdec0_beats.hex)
    # Flat = each lane's beats file has its rows file beside it. A beats_hex folder is not flat:
    # its rows live in ../rows_hex (it may still hold the bypass lane's subdec8_rows.hex).
    beats = list(data_dir.glob("*_subdec*_beats.hex"))
    return sorted({p.name.rsplit("_subdec", 1)[0] for p in beats
                   if p.with_name(p.name.replace("_beats.hex", "_rows.hex")).exists()})


def stage_flat_dir(data_dir: Path, images: list, staging: Path) -> Path:
    """Copy a flat folder into the pool layout the env reads (<image>/beats_hex, <image>/rows_hex).

    Copies, not symlinks, so the xsim backend on Windows can read them too."""
    for image in images:
        for kind in ("beats", "rows"):
            dst = staging / image / f"{kind}_hex"
            dst.mkdir(parents=True)
            for src in data_dir.glob(f"{glob.escape(image)}_subdec[0-9]*_{kind}.hex"):
                shutil.copy2(src, dst / src.name)
    return staging


def image_names(env: DecoderEnvXSim, is_inference: bool) -> list:
    """Sample folder name per dataset_pool entry ('test_11_lantern' from .../test_11_lantern/beats_hex/<stem>)."""
    return [stem.parent.parent.name if is_inference else stem.relative_to(env.data_dir).parts[0]
            for stem, _ in env.dataset_pool]


def isolated_lane_cycles(env: DecoderEnvXSim, sample_idx: int, lane: int, max_cycles: int) -> dict:
    """Feed only `lane` of image `sample_idx` and count cycles until it has output all its rows.

    Drives the env one clock at a time (env._tick) instead of env.step(): step() would count the
    idle cycles spent waiting for the gate as a stall and end the episode as a deadlock, and it
    only ends an episode once every lane is drained.
    Returns {"status": "ok" | "rtl_stop" | "timeout", "cycles": int, "violations": {...}}.
    """
    env.reset(options={"sample_idx": sample_idx})   # also skips the leading not-ready cycles
    try:
        # lb.rows_popped = rows that have left the lane's row queue (free-run wrapper, exact per cycle)
        while env.lb.rows_popped[lane] < NUM_ROWS:
            if env.cycles >= max_cycles:
                return {"status": "timeout", "cycles": env.cycles, "violations": {}}
            in_ready = bool(env.hw_state.get("top_in_ready", 0))
            if env.action_masks()[lane]:
                valid, ncmd, payload, slots = env._pack_beat(lane)   # pops tokens, records n_sent
                env._mask_cache = None
                env._tick(f"{valid} {encode_beat(lane, ncmd, payload)}",
                          lane if valid else NO_OP_ACTION, in_ready, slots)
            else:
                env._tick(IDLE_ACTION, NO_OP_ACTION, in_ready, 0)
    except env._rtl_stop:   # verilator only: an RTL assertion fired (e.g. LIT bank overflow)
        return {"status": "rtl_stop", "cycles": env.cycles, "violations": {}}
    violations = {k: env._delta(k) for k in CHECK_KEYS if env._delta(k) > 0}
    return {"status": "ok", "cycles": env.cycles, "violations": violations}


def rr_makespan(env: DecoderEnvXSim, sample_idx: int) -> int | None:
    """Makespan of the round-robin teacher (policies/round_robin.py, all lanes shared), for the
    gap column. It follows the SV testbench packer's 9-channel cursor; it is not guaranteed to equal
    v20_tb-verilator_cycles.csv, which came from a different model of the cursor."""
    info = run_teacher_episode(env, RoundRobinTeacher(), options={"sample_idx": sample_idx})
    return info.get("makespan")


def measure(env: DecoderEnvXSim, samples: dict, max_cycles: int) -> list:
    """samples: {sample_idx: image name}. Returns one result dict per image (tb_* left None)."""
    results = []
    bar = tqdm(total=len(samples) * NUM_DECODERS, unit="lane", dynamic_ncols=True)
    try:
        for idx, name in samples.items():
            lanes = []
            for lane in range(NUM_DECODERS):
                r = isolated_lane_cycles(env, idx, lane, max_cycles)
                lanes.append(r)
                bar.set_postfix_str(f"{name} lane {lane}: {r['cycles']} {r['status']}")
                bar.update()
            bus0, work0 = env.lb.bus0, env.lb.work0_max    # static, from the queues built at reset
            ok = all(r["status"] == "ok" for r in lanes)
            longest = max(range(NUM_DECODERS), key=lambda i: lanes[i]["cycles"])
            iso_max = lanes[longest]["cycles"]
            bound = max(bus0, iso_max) if ok else None
            results.append({
                "image": name,
                "sample_idx": idx,
                "isolated_cycles": [r["cycles"] for r in lanes],
                "isolated_status": [r["status"] for r in lanes],
                "violations": {i: r["violations"] for i, r in enumerate(lanes) if r["violations"]},
                "longest_lane": longest if ok else None,
                "longest_isolated": iso_max if ok else None,
                "bus0": bus0,
                "static_lb0": max(bus0, work0),   # old group_ticks estimate, for reference only
                "bound": bound,
                "binding_term": None if bound is None else ("bus0" if bus0 > iso_max else "lane"),
                "rr_makespan": None,
                "rr_gap": None,
            })
    finally:
        bar.close()
    return results


def print_report(results: list, with_rr: bool):
    print("\nIsolated cycles per lane (* = longest; ! = RTL stop / timeout / testbench check fired)")
    for r in results:
        cells = []
        for i, (c, s) in enumerate(zip(r["isolated_cycles"], r["isolated_status"])):
            mark = "*" if i == r["longest_lane"] else ("!" if s != "ok" or i in r["violations"] else " ")
            cells.append(f"{mark}{c:5d}" if s == "ok" else f"!{s}@{c}")
        print(f"  {r['image']:22s} " + " ".join(cells))
        for i, v in r["violations"].items():
            print(f"    lane {i} testbench checks: {v}")

    header = f"\n{'image':22s} {'lane':>4s} {'isolated':>8s} {'bus0':>6s} {'bound':>6s} {'binds':>5s} {'static':>6s}"
    if with_rr:
        header += f" {'RR':>6s} {'RR-bound':>9s}"
    print(header)
    for r in results:
        if r["bound"] is None:
            print(f"{r['image']:22s}  n/a: a lane did not finish ({r['isolated_status']})")
            continue
        line = (f"{r['image']:22s} {r['longest_lane']:4d} {r['longest_isolated']:8d} {r['bus0']:6d} "
                f"{r['bound']:6d} {r['binding_term']:>5s} {r['static_lb0']:6d}")
        if with_rr and r["rr_makespan"] is not None:
            line += f" {r['rr_makespan']:6d} {r['rr_gap']:+5d} ({r['rr_gap'] / r['bound']:5.1%})"
        print(line)


def save(results: list, out_dir: Path, stem: str, meta: dict):
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path, csv_path = out_dir / f"{stem}.json", out_dir / f"{stem}.csv"
    json_path.write_text(json.dumps({"meta": meta, "images": results}, indent=2))
    cols = ["image", "longest_lane", "longest_isolated", "bus0", "bound", "binding_term",
            "static_lb0", "rr_makespan", "rr_gap"]
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(cols + [f"lane{i}" for i in range(NUM_DECODERS)])
        for r in results:
            w.writerow([r[c] for c in cols] + r["isolated_cycles"])
    print(f"\nSaved {json_path}\n      {csv_path}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                    help="pool of sample folders (default sample_data_9_lanes), one sample folder, a "
                         "beats_hex folder, or a flat folder of <image>_subdec<s>_{beats,rows}.hex files")
    ap.add_argument("--backend", choices=("verilator", "xsim"), default="verilator")
    ap.add_argument("--sim-dir", default=None, help="xsim simulation directory (xsim backend only)")
    ap.add_argument("--samples", nargs="*", default=None,
                    help="only images whose folder name contains one of these strings")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--gate", type=int, default=TB_INFLIGHT_GATE,
                   help=f"max CMDs in flight before a lane may get a beat (default {TB_INFLIGHT_GATE}, the testbench rule)")
    g.add_argument("--no-gate", action="store_true",
                   help="CMD-FIFO depth only (the RTL can assert a LIT bank overflow; reported as rtl_stop)")
    ap.add_argument("--no-rr", action="store_true", help="skip the round-robin teacher makespan / gap column")
    ap.add_argument("--max-cycles", type=int, default=50_000, help="give up on a lane after this many cycles")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args(argv)
    if args.backend == "xsim" and not args.sim_dir:
        ap.error("--backend xsim needs --sim-dir")

    gate = None if args.no_gate else args.gate
    if args.no_gate and not args.no_rr:
        print("note: the round-robin makespan is always measured with the testbench gate "
              f"({TB_INFLIGHT_GATE}), not with --no-gate")
    data_dir = Path(args.data_dir).resolve()
    if not data_dir.is_dir():
        ap.error(f"--data-dir {data_dir} is not a directory")
    with tempfile.TemporaryDirectory(prefix="lower_bound_pool_") as staging:
        flat = flat_images(data_dir)
        if flat:
            print(f"flat data dir: {len(flat)} images, staged into {staging}")
            path, is_inference = str(stage_flat_dir(data_dir, flat, Path(staging))), False
        else:
            path, is_inference = resolve_data_dir(data_dir)
        results, samples, names = run(ap, args, gate, path, is_inference)

    print_report(results, with_rr=not args.no_rr)
    dir_name = data_dir.parent.name if data_dir.name == "beats_hex" else data_dir.name
    stem = f"{dir_name}_{args.backend}_gate{gate if gate is not None else 'none'}"
    if len(samples) < len(names):   # a subset must not overwrite the full-pool file
        stem += "_" + "_".join(samples.values())
    save(results, Path(args.out_dir), stem, {
        "backend": args.backend, "gate": gate, "data_dir": str(data_dir), "num_rows": NUM_ROWS,
        "definition": "bound = max(bus0, max_lane isolated_cycles); isolated = lane fed alone, "
                      "densest beat whenever its mask bit is set, until all rows have left the lane",
    })


def run(ap, args, gate, path: str, is_inference: bool):
    """Build the env(s) on `path` and measure -> (results, {sample_idx: name}, all names)."""
    env = DecoderEnvXSim(data_dir=path, sim_dir=args.sim_dir, is_inference=is_inference,
                         inflight_gate=gate, backend=args.backend, log_rtl_stop=False)
    rr_env = None
    try:
        names = image_names(env, is_inference)
        samples = {i: n for i, n in enumerate(names)
                   if not args.samples or any(s in n for s in args.samples)}
        if not samples:
            ap.error(f"no image matches --samples {args.samples}; available: {names}")
        print(f"backend={args.backend} gate={gate if gate is not None else 'fifo-only'} "
              f"images={list(samples.values())}")

        results = measure(env, samples, args.max_cycles)
        if not args.no_rr:   # the teacher always uses the testbench gate: own env if --no-gate
            rr_env = env if gate == TB_INFLIGHT_GATE else DecoderEnvXSim(
                data_dir=path, sim_dir=args.sim_dir, is_inference=is_inference,
                inflight_gate=TB_INFLIGHT_GATE, backend=args.backend, log_rtl_stop=False)
            for r in tqdm(results, desc="round-robin", unit="image", dynamic_ncols=True):
                r["rr_makespan"] = rr_makespan(rr_env, r["sample_idx"])
                if r["bound"] is not None and r["rr_makespan"] is not None:
                    r["rr_gap"] = r["rr_makespan"] - r["bound"]
    finally:
        env.close()
        if rr_env is not None and rr_env is not env:
            rr_env.close()
    return results, samples, names


if __name__ == "__main__":
    main()
