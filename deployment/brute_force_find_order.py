"""Brute-force lane-selection schedules against one prepared XSim image."""
import argparse
import json
import math
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.xsim.constants import NO_OP_ACTION, NUM_DECODERS
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


def _schedule_counts(beats_dir: Path) -> list[int]:
    """Count the compressed rows available to each lane."""
    rows_dir = beats_dir.parent / "rows_hex"
    row_files = sorted(rows_dir.glob("*.hex"))
    counts = [
        sum(bool(line.strip()) for line in row_files[lane].read_text(encoding="ascii").splitlines())
        for lane in range(min(NUM_DECODERS, len(row_files)))
    ]
    if len(counts) != NUM_DECODERS:
        raise ValueError(f"Expected {NUM_DECODERS} lanes, found {len(counts)}")
    if not any(counts):
        raise ValueError("The sample contains no commands to schedule")
    return counts


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


def _read_action_hex(env) -> str:
    action_file = env.sim.action_file
    parts = action_file.read_text(encoding="ascii").split()
    if len(parts) != 2:
        raise RuntimeError(f"Unexpected simulator action format in {action_file}")
    action_hex = parts[1]
    if len(action_hex) != 34:
        raise RuntimeError(f"Expected a 34-digit action hex value, got {action_hex!r}")
    try:
        int(action_hex, 16)
    except ValueError as exc:
        raise RuntimeError(f"Simulator wrote an invalid action hex value: {action_hex!r}") from exc
    return action_hex


def _run_schedule(env, schedule: Sequence[int], max_cycles: int) -> dict | None:
    """Run one lane schedule; return its result only if the image drains successfully."""
    env.reset()
    actions = []
    cycle_count = 0
    terminated = truncated = False
    info = {}

    def step(action: int) -> None:
        nonlocal cycle_count, terminated, truncated, info
        _, _, terminated, truncated, info = env.step(action)
        cycle_count += 1
        actions.append({"cycle": cycle_count, "action_hex": _read_action_hex(env)})

    for lane in schedule:
        if terminated or truncated:
            break
        if not env.cmds[lane]:
            continue
        while not env.action_masks()[lane]:
            step(NO_OP_ACTION)
            if terminated or truncated or cycle_count >= max_cycles:
                break
        if terminated or truncated or cycle_count >= max_cycles:
            break
        step(lane)
        if cycle_count >= max_cycles:
            break

    while not terminated and not truncated and cycle_count < max_cycles:
        if any(env.cmds):
            return None
        step(NO_OP_ACTION)

    if info.get("deadlock") or truncated or not terminated:
        return None
    return {"cycles": cycle_count, "actions": actions}


def run_brute_force(
    image_dir: Path,
    sim_dir: Path,
    output_path: Path,
    schedule_file: Path | None = None,
    start_line: int = 1,
    max_cycles: int = 50000,
    max_schedules: int | None = None,
    obs_version: str = "v3",
) -> dict | None:
    if start_line < 1:
        raise ValueError("start_line must be at least 1")
    if max_cycles <= 0:
        raise ValueError("max_cycles must be greater than zero")
    if max_schedules is not None and max_schedules <= 0:
        raise ValueError("max_schedules must be greater than zero")

    beats_dir = _resolve_beats_dir(image_dir)
    if schedule_file is None:
        counts = _schedule_counts(beats_dir)
        schedule_source = (
            (line, schedule)
            for line, schedule in enumerate(generate_schedules(counts), start=1)
        )
        print(f"CMD groups per lane: {counts}")
        print(f"Distinct schedules: {math.factorial(sum(counts)) // math.prod(math.factorial(n) for n in counts)}")
    else:
        schedule_file = schedule_file.resolve()
        if not schedule_file.is_file():
            raise ValueError(f"Schedule file does not exist: {schedule_file}")
        schedule_source = _read_schedules(schedule_file)

    output_path = output_path.resolve()
    best = None
    if output_path.exists():
        try:
            best = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            raise ValueError(f"Could not load previous best result from {output_path}") from exc

    env = DecoderEnvXSim(
        data_dir=str(beats_dir),
        sim_dir=str(sim_dir),
        is_inference=True,
        obs_version=obs_version,
    )
    tried = 0
    try:
        for line_number, schedule in schedule_source:
            if line_number < start_line:
                continue
            result = _run_schedule(env, schedule, max_cycles)
            tried += 1
            if result is None:
                print(f"Schedule {line_number}: did not complete successfully")
            else:
                print(f"Schedule {line_number}: {result['cycles']} cycles")
                if best is None or result["cycles"] < best["cycles"]:
                    best = {
                        "schedule_line": line_number,
                        "schedule": list(schedule),
                        **result,
                    }
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    output_path.write_text(json.dumps(best, indent=2) + "\n", encoding="utf-8")
                    print(f"New best: {result['cycles']} cycles; saved to {output_path}")
            if max_schedules is not None and tried >= max_schedules:
                break
    finally:
        env.close()

    if tried == 0:
        raise ValueError(f"No schedules found at or after line {start_line}")
    if best is not None:
        print(f"Best result: {best['cycles']} cycles (schedule line {best['schedule_line']})")
    return best


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Brute-force lane dispatch schedules for one prepared XSim sample."
    )
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--sim-dir", type=Path, required=True)
    parser.add_argument("--schedule-file", type=Path, help="One comma- or space-separated lane schedule per line.")
    parser.add_argument("--start-line", type=int, default=1, help="1-based line/schedule to resume from.")
    parser.add_argument("--output", type=Path, default=Path("brute_force_best.json"))
    parser.add_argument("--max-cycles", type=int, default=50000)
    parser.add_argument("--max-schedules", type=int, help="Stop after this many schedules (default: all).")
    parser.add_argument("--obs-version", choices=("v1", "v2", "v3", "v4"), default="v3")
    args = parser.parse_args()

    run_brute_force(
        image_dir=args.image_dir,
        sim_dir=args.sim_dir,
        output_path=args.output,
        schedule_file=args.schedule_file,
        start_line=args.start_line,
        max_cycles=args.max_cycles,
        max_schedules=args.max_schedules,
        obs_version=args.obs_version,
    )


if __name__ == "__main__":
    main()



