"""Run a greedy, most-packable-lane policy on one prepared image sample in XSim."""
import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.xsim.constants import NO_OP_ACTION, NUM_DECODERS
from envs.xsim.env import DecoderEnvXSim
from envs.xsim.flow_control import packable_slots_by_lane


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


def _select_best_lane(slots_by_lane: list[int], action_mask) -> int:
    """Choose the eligible lane with the largest packable beat; ties go to the lowest lane."""
    eligible_lanes = [
        lane for lane in range(NUM_DECODERS) if action_mask[lane]
    ]
    if not eligible_lanes:
        return NO_OP_ACTION
    return max(eligible_lanes, key=lambda lane: (slots_by_lane[lane], -lane))


def run_inference(
    image_dir: Path,
    sim_dir: Path,
    output_path: Path,
    max_cycles: int = 50000,
) -> dict:
    """Run one greedy episode and save each dispatched beat as a raw hex value."""
    if max_cycles <= 0:
        raise ValueError("max_cycles must be greater than zero")

    beats_dir = _resolve_beats_dir(image_dir)
    rows_dir = beats_dir.parent / "rows_hex"
    output_path = output_path.resolve()
    if beats_dir in output_path.parents or rows_dir in output_path.parents:
        raise ValueError("Output path must not be inside beats_hex or rows_hex")

    env = DecoderEnvXSim(
        data_dir=str(beats_dir),
        sim_dir=str(sim_dir),
        is_inference=True,
    )
    records = []
    try:
        env.reset()

        for _ in range(max_cycles):
            slots_by_lane = packable_slots_by_lane(env.hw_state, env.cmds, env.lb.n_sent)
            action = _select_best_lane(slots_by_lane, env.action_masks())
            _, _, terminated, truncated, info = env.step(action)

            action_file = env.sim.action_file
            action_parts = action_file.read_text(encoding="ascii").split()
            if len(action_parts) != 2:
                raise RuntimeError(f"Unexpected simulator action format in {action_file}")
            action_hex = action_parts[1]
            if len(action_hex) != 34:
                raise RuntimeError(f"Expected a 34-digit action hex value, got {action_hex!r}")
            try:
                int(action_hex, 16)
            except ValueError as exc:
                raise RuntimeError(
                    f"Simulator wrote an invalid action hex value: {action_hex!r}"
                ) from exc
            records.append(action_hex)

            if terminated or truncated:
                if info.get("deadlock"):
                    raise RuntimeError("The greedy policy deadlocked; no output file was written.")
                break
        else:
            raise RuntimeError(
                f"Inference did not finish within {max_cycles} cycles; no output file was written."
            )
    finally:
        env.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(records) + "\n", encoding="ascii")
    print(f"Inference completed in {info.get('makespan', len(records))} cycles.")
    print(f"Dispatched beat records saved to: {output_path}")
    return info


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Greedily choose the eligible lane with the most packable slots in XSim."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Sample folder containing beats_hex/ and rows_hex/, or the beats_hex/ folder itself.",
    )
    parser.add_argument("--sim-dir", type=Path, required=True, help="Path to the XSim simulation directory.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("best_lane_beats.hex"),
        help="Output file for one raw action_hex value per cycle.",
    )
    parser.add_argument("--max-cycles", type=int, default=50000)
    args = parser.parse_args()

    run_inference(
        image_dir=args.image_dir,
        sim_dir=args.sim_dir,
        output_path=args.output,
        max_cycles=args.max_cycles,
    )


if __name__ == "__main__":
    main()
