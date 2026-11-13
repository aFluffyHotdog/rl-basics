"""Brute-force lane-selection schedules against one prepared image (Verilator or XSim)."""
import argparse
import contextlib
import ctypes
import json
import math
import os
import signal
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

from tqdm import tqdm

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.xsim.constants import NO_OP_ACTION, NUM_DECODERS, TB_INFLIGHT_GATE
from envs.xsim.env import DecoderEnvXSim


def _resolve_beats_dir(image_dir: Path) -> Path:
    image_dir = image_dir.resolve()
    if not image_dir.is_dir():
        raise ValueError(f"Image sample directory does not exist: {image_dir}")
    beats_dir = image_dir if image_dir.name == "beats_hex" else image_dir / "beats_hex"
    rows_dir = beats_dir.parent / "rows_hex"
    if not beats_dir.is_dir():
        raise ValueError(f"Expected a beats_hex folder at {beats_dir}")
    if not rows_dir.is_dir() or not any(rows_dir.glob("*.hex")):
        raise ValueError(f"Expected pre-generated row hex files in {rows_dir}")
    return beats_dir


def _beats_per_lane(env) -> list[int]:
    """Number of beats each lane needs: a schedule is one lane ID per beat.

    A pick sends one beat, packed greedily (whole CMD groups, order preserved, up to 8 slots).
    The env runs with the testbench in-flight gate (TB_INFLIGHT_GATE), so a lane is only offered
    with <= 8 CMDs in flight and a beat (<= 8 groups) never hits the 16-deep CMD FIFO limit: every
    beat is dense and the per-lane beat count is fixed, whatever the order. That count is
    lb.beats_left(i) at reset."""
    env.reset()
    counts = [env.lb.beats_left(i) for i in range(NUM_DECODERS)]
    if not any(counts):
        raise ValueError("The sample contains no commands to schedule")
    return counts


def _matches_counts(schedule: Sequence[int], counts: Sequence[int]) -> bool:
    """True if the schedule picks every lane exactly as many times as it needs beats."""
    picks = [0] * NUM_DECODERS
    for lane in schedule:
        picks[lane] += 1
    return picks == list(counts)


def _save_best(best: dict, output_path: Path) -> None:
    """Write the best result atomically, so a kill in the middle of the write cannot corrupt it."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_name(output_path.name + ".tmp")
    tmp_path.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp_path, output_path)


def _describe_best(best: dict, output_path: Path) -> str:
    schedule = best["schedule"]
    shown = " ".join(map(str, schedule[:40])) + (" ..." if len(schedule) > 40 else "")
    return (f"Best result: {best['cycles']} cycles (schedule line {best['schedule_line']}, "
            f"{len(schedule)} lane picks)\n  schedule: {shown}\n  saved to: {output_path}")


def _flush_c_stdio() -> None:
    """Flush C-level stdout. Verilated code prints with printf; when stdout is a pipe or file that
    text sits in a buffer and would otherwise appear after the redirect below has been undone."""
    try:
        ctypes.CDLL(None).fflush(None)
    except (OSError, AttributeError, TypeError):   # no libc handle (e.g. Windows): nothing to flush
        pass


@contextlib.contextmanager
def _silence_stdout():
    """Send everything written to file descriptor 1 to /dev/null.

    Verilator prints RTL assertions ("%Error: ... Assertion failed") from C++ straight to stdout,
    which sys.stdout redirection cannot catch. The progress bar writes to stderr and is unaffected."""
    sys.stdout.flush()
    saved = os.dup(1)
    devnull = os.open(os.devnull, os.O_WRONLY)
    try:
        os.dup2(devnull, 1)
        yield
    finally:
        sys.stdout.flush()
        _flush_c_stdio()   # while fd 1 still points at /dev/null
        os.dup2(saved, 1)
        os.close(saved)
        os.close(devnull)


class _Interrupted(Exception):
    """Raised from SIGINT / SIGTERM / SIGHUP so the best result can be printed and saved first."""

    def __init__(self, signum: int):
        super().__init__(signal.Signals(signum).name)
        self.signum = signum


def _raise_interrupted(signum, _frame):
    raise _Interrupted(signum)


def _format_count_scientific(counts: Sequence[int]) -> str:
    """Number of distinct schedules, N! / prod(n_i!), as an approximation like "3.142e+1234".

    The exact value has thousands of digits, more than Python allows in int -> str, so work with
    its base-10 logarithm (via lgamma) instead of ever building the integer. Small counts are
    printed exactly."""
    total = sum(counts)
    log10_count = (math.lgamma(total + 1) - sum(math.lgamma(n + 1) for n in counts)) / math.log(10)
    if log10_count < 15:   # exact, and cheap enough to compute
        return str(math.factorial(total) // math.prod(math.factorial(n) for n in counts))
    exponent = math.floor(log10_count)
    mantissa = 10 ** (log10_count - exponent)
    return f"~{mantissa:.3f}e+{exponent}"


def generate_schedules(counts: Sequence[int]) -> Iterator[tuple[int, ...]]:
    """Yield every distinct ordering of lane IDs with the requested multiplicities."""
    remaining = list(counts)
    schedule: list[int] = []
    next_lane = [0]
    total = sum(counts)
    while next_lane:
        lane = next_lane[-1]
        if lane == len(remaining):
            next_lane.pop()
            if schedule:
                remaining[schedule.pop()] += 1
            continue
        next_lane[-1] += 1
        if remaining[lane] == 0:
            continue
        remaining[lane] -= 1
        schedule.append(lane)
        if len(schedule) == total:
            yield tuple(schedule)
            remaining[schedule.pop()] += 1
        else:
            next_lane.append(0)


def _read_schedules(path: Path) -> Iterator[tuple[int, tuple[int, ...]]]:
    """Read one schedule per line; IDs are 0-based and may be comma- or whitespace-separated."""
    for line_number, raw_line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        try:
            schedule = tuple(int(part) for part in line.replace(",", " ").split())
        except ValueError as exc:
            raise ValueError(f"{path}:{line_number}: expected lane IDs separated by spaces or commas") from exc
        if any(lane < 0 or lane >= NUM_DECODERS for lane in schedule):
            raise ValueError(f"{path}:{line_number}: lane IDs must be between 0 and {NUM_DECODERS - 1}")
        yield line_number, schedule


def _is_action(content: str) -> bool:
    """True for a per-cycle action ("<valid> <34-digit hex>"), False for e.g. the image stem."""
    parts = content.split()
    if len(parts) != 2 or parts[0] not in ("0", "1") or len(parts[1]) != 34:
        return False
    try:
        int(parts[1], 16)
    except ValueError:
        return False
    return True


def _record_actions(env) -> list[tuple[int, str]]:
    """Record every per-cycle (valid bit, action hex) the env hands to the simulator, in order.

    One env.step can advance several cycles (it ticks idle beats until top_in is ready again), and
    reset() ticks some too, so record each write_action call rather than one action per step. The
    valid bit is kept because idle cycles carry a non-zero hex (dest = no-op) with valid = 0. The
    xsim client writes them to action.txt, the Verilator client keeps them in memory only, so
    wrap write_action instead of reading a file. The returned list is cleared by _run_schedule."""
    trace: list[tuple[int, str]] = []
    write_action = env.sim.write_action

    def recording_write_action(content: str):
        if _is_action(content):
            valid, action_hex = content.split()
            trace.append((int(valid), action_hex))
        return write_action(content)

    env.sim.write_action = recording_write_action
    return trace


def _run_schedule(env, trace: list[tuple[int, str]], schedule: Sequence[int], cycle_limit: int) -> tuple[str, dict | None]:
    """Run one lane schedule -> (status, result). result is set only when status is "ok".

    status:
      "ok"      the image drained cleanly within cycle_limit cycles
      "failed"  the env reported a failure (RTL $stop / assertion, protocol violation, golden
                mismatch, deadlock) or the schedule ran out with commands left; abandoned at once
      "pruned"  cycle_limit was reached first (it cannot beat --max-cycles / the best so far)
    Lanes are picked in schedule order; while the next lane is masked (busy, gated, bus not
    ready) idle cycles are spent until it opens up."""
    trace.clear()
    env.reset()
    terminated = truncated = False
    info = {}

    def step(action: int) -> None:
        nonlocal terminated, truncated, info
        _, _, terminated, truncated, info = env.step(action)

    def stopped() -> bool:
        return terminated or truncated or env.cycles >= cycle_limit

    for lane in schedule:
        if stopped():
            break
        if not env.cmds[lane]:
            continue
        while not env.action_masks()[lane] and not stopped():
            step(NO_OP_ACTION)
        if stopped():
            break
        step(lane)

    while not stopped():
        if any(env.cmds):
            return "failed", None
        step(NO_OP_ACTION)

    # A failure terminal also sets terminated=True, so it must be rejected explicitly or its cycle
    # count would look valid.
    if info.get("deadlock") or info.get("violation") or info.get("mismatch") or truncated:
        return "failed", None
    if not terminated or env.cycles > cycle_limit:
        return "pruned", None
    actions = [{"cycle": cycle, "valid": valid, "action_hex": action_hex}
               for cycle, (valid, action_hex) in enumerate(trace, start=1)]
    return "ok", {"cycles": env.cycles, "actions": actions}


def run_brute_force(
    image_dir: Path,
    sim_dir: Path | None,
    output_path: Path,
    schedule_file: Path | None = None,
    start_line: int = 1,
    max_cycles: int = 50000,
    max_schedules: int | None = None,
    backend: str = "verilator",
) -> dict | None:
    if backend == "xsim" and sim_dir is None:
        raise ValueError("backend 'xsim' needs sim_dir")
    if start_line < 1:
        raise ValueError("start_line must be at least 1")
    if max_cycles <= 0:
        raise ValueError("max_cycles must be greater than zero")
    if max_schedules is not None and max_schedules <= 0:
        raise ValueError("max_schedules must be greater than zero")

    beats_dir = _resolve_beats_dir(image_dir)
    env = DecoderEnvXSim(
        data_dir=str(beats_dir),
        sim_dir=str(sim_dir) if sim_dir else None,
        is_inference=True,
        inflight_gate=TB_INFLIGHT_GATE,   # testbench rule; also makes every beat dense (see _beats_per_lane)
        backend=backend,
        log_rtl_stop=False,   # failed schedules are skipped silently; the bar counts them
    )
    try:
        return _search(env, output_path, schedule_file, start_line, max_cycles, max_schedules)
    finally:
        try:
            env.close()
        except _Interrupted:
            pass


def _search(env, output_path: Path, schedule_file: Path | None, start_line: int,
            max_cycles: int, max_schedules: int | None) -> dict | None:
    with _silence_stdout():
        counts = _beats_per_lane(env)
    print(f"Beats per lane: {counts} (total {sum(counts)})")
    if schedule_file is None:
        schedule_source = (
            (line, schedule)
            for line, schedule in enumerate(generate_schedules(counts), start=1)
        )
        print(f"Distinct schedules: {_format_count_scientific(counts)}")
    else:
        schedule_file = schedule_file.resolve()
        if not schedule_file.is_file():
            raise ValueError(f"Schedule file does not exist: {schedule_file}")
        schedule_source = _read_schedules(schedule_file)
        total = sum(1 for line, _ in _read_schedules(schedule_file) if line >= start_line)

    if schedule_file is None:
        total = None   # the generated space is astronomically large: show a counter, not a percentage
    if max_schedules is not None:
        total = max_schedules if total is None else min(total, max_schedules)

    output_path = output_path.resolve()
    best = None
    if output_path.exists():
        try:
            best = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Could not load previous best result from {output_path}") from exc
        if not isinstance(best.get("cycles"), int) or not _matches_counts(best.get("schedule", []), counts):
            print(f"Ignoring the previous result in {output_path}: its schedule does not match this "
                  f"image's beats per lane (another image or an older run). It is replaced by the "
                  f"first schedule that completes.")
            best = None
        else:
            print(f"Previous best: {best['cycles']} cycles (only faster schedules are recorded)")

    trace = _record_actions(env)
    tried = failed = pruned = invalid = 0
    interrupted: _Interrupted | None = None
    # Ctrl-C, `kill` (SIGTERM) and a closed terminal (SIGHUP) all end up in the finally block below,
    # which prints and saves the best schedule. SIGKILL cannot be caught; the best result is also
    # saved the moment it is found, so at most the current schedule is lost.
    previous = {sig: signal.signal(sig, _raise_interrupted)
                for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)}
    progress = tqdm(total=total, unit="sched", desc="Schedules", dynamic_ncols=True)
    try:
        for line_number, schedule in schedule_source:
            if line_number < start_line:
                continue
            tried += 1
            if not _matches_counts(schedule, counts):
                invalid += 1   # wrong number of picks for some lane: cannot drain the image exactly
                status, result = "invalid", None
            else:
                # Stop a schedule as soon as it can no longer beat the best so far.
                cycle_limit = max_cycles if best is None else min(max_cycles, best["cycles"] - 1)
                with _silence_stdout():   # RTL assertion text from failed schedules is skipped
                    status, result = _run_schedule(env, trace, schedule, cycle_limit)
            if status == "failed":
                failed += 1
            elif status == "pruned":
                pruned += 1
            elif status == "ok":
                best = {
                    "schedule_line": line_number,
                    "schedule": list(schedule),
                    **result,
                }
                _save_best(best, output_path)
                progress.write(f"New best: {result['cycles']} cycles (schedule line {line_number}); "
                               f"saved to {output_path}")
            progress.set_postfix(best=best["cycles"] if best else None, failed=failed, pruned=pruned,
                                 invalid=invalid)
            progress.update(1)
            if max_schedules is not None and tried >= max_schedules:
                break
    except _Interrupted as exc:
        interrupted = exc
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
        progress.close()

    if interrupted is not None:
        print(f"\nStopped by {interrupted} after {tried} schedule(s) "
              f"({failed} failed, {pruned} pruned, {invalid} invalid).")
    elif tried == 0:
        raise ValueError(f"No schedules found at or after line {start_line}")
    if best is not None:
        _save_best(best, output_path)
        print(_describe_best(best, output_path))
    elif interrupted is not None:
        print("No schedule completed successfully yet.")
    if interrupted is not None:
        raise SystemExit(128 + interrupted.signum)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brute-force lane dispatch schedules for one prepared sample."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--backend", choices=["verilator", "xsim"], default="verilator",
                        help="Simulator behind the env (default: in-process Verilator).")
    parser.add_argument("--sim-dir", type=Path, default=None,
                        help="xsim simulation directory (only for --backend xsim).")
    parser.add_argument("--schedule-file", type=Path, help="One comma- or space-separated lane schedule per line.")
    parser.add_argument("--start-line", type=int, default=1, help="1-based line/schedule to resume from.")
    parser.add_argument("--output", type=Path, default=Path("brute_force_best.json"))
    parser.add_argument("--max-cycles", type=int, default=50000)
    parser.add_argument("--max-schedules", type=int, help="Stop after this many schedules (default: all).")
    args = parser.parse_args()
    if args.backend == "xsim" and args.sim_dir is None:
        parser.error("--backend xsim needs --sim-dir")

    run_brute_force(
        image_dir=args.image_dir,
        sim_dir=args.sim_dir,
        output_path=args.output,
        schedule_file=args.schedule_file,
        start_line=args.start_line,
        max_cycles=args.max_cycles,
        max_schedules=args.max_schedules,
        backend=args.backend,
    )


if __name__ == "__main__":
    main()



