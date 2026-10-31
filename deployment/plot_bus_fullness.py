"""Plot raw XSim beat occupancy and export the valid command count per beat."""
import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.xsim.constants import BEATS_ON_BUS, CMD_TYPE, NO_OP_ACTION
from envs.xsim.tokens import decode_token


BEAT_BITS = 135
HEADER_BITS = 7
SLOT_BITS = 16
PAYLOAD_BITS = BEATS_ON_BUS * SLOT_BITS
PAYLOAD_MASK = (1 << PAYLOAD_BITS) - 1


@dataclass(frozen=True)
class BeatMetrics:
    valid_commands: int
    occupied_bits: int


def decode_beat(line: str, line_number: int) -> BeatMetrics:
    try:
        beat = int(line, 16)
    except ValueError as exc:
        raise ValueError(f"Line {line_number}: expected a hexadecimal beat, got {line!r}") from exc

    if beat < 0 or beat.bit_length() > BEAT_BITS:
        raise ValueError(f"Line {line_number}: beat does not fit in {BEAT_BITS} bits")

    destination = beat >> 131
    command_count_field = (beat >> PAYLOAD_BITS) & 0x7
    if destination > NO_OP_ACTION:
        raise ValueError(f"Line {line_number}: invalid destination {destination}")
    if destination == NO_OP_ACTION:
        if command_count_field != 0:
            raise ValueError(f"Line {line_number}: no-op beat has a nonzero command count")
        return BeatMetrics(valid_commands=0, occupied_bits=0)

    valid_commands = command_count_field + 1
    payload = beat & PAYLOAD_MASK
    slots_used = 0

    for _ in range(valid_commands):
        if slots_used >= BEATS_ON_BUS:
            raise ValueError(f"Line {line_number}: command count exceeds payload capacity")

        payload_word = (payload >> (slots_used * SLOT_BITS)) & 0xFFFF
        token = (CMD_TYPE << 16) | payload_word
        token_info = decode_token(token)
        if token_info.category not in ("CMD", "RLE"):
            raise ValueError(
                f"Line {line_number}: expected a CMD or RLE group head, "
                f"found {token_info.category}"
            )
        if slots_used + token_info.slots > BEATS_ON_BUS:
            raise ValueError(f"Line {line_number}: command group exceeds payload capacity")
        slots_used += token_info.slots

    occupied_bits = HEADER_BITS + slots_used * SLOT_BITS
    return BeatMetrics(valid_commands=valid_commands, occupied_bits=occupied_bits)


def read_beats(input_path: Path) -> list[BeatMetrics]:
    metrics = []
    with input_path.open(encoding="ascii") as hex_file:
        for line_number, raw_line in enumerate(hex_file, start=1):
            line = raw_line.partition("//")[0].strip()
            if line:
                metrics.append(decode_beat(line, line_number))
    if not metrics:
        raise ValueError(f"No beat records found in {input_path}")
    return metrics


def write_valid_commands_csv(output_path: Path, metrics: list[BeatMetrics]) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(("valid_commands",))
        writer.writerows((metric.valid_commands,) for metric in metrics)


def plot_fullness(output_path: Path, metrics: list[BeatMetrics], input_path: Path) -> None:
    cycles = range(1, len(metrics) + 1)
    occupied_bits = [metric.occupied_bits for metric in metrics]

    figure, axis = plt.subplots(figsize=(12, 5))
    axis.scatter(cycles, occupied_bits, s=6)
    axis.set(
        title=f"Bus Beat Fullness: {input_path.name}",
        xlabel="Beat (cycle)",
        ylabel=f"Occupied bits (out of {BEAT_BITS})",
        ylim=(0, BEAT_BITS),
    )
    axis.set_xlim(1, len(metrics))
    axis.grid(True, alpha=0.3)
    for spine in axis.spines.values():
        spine.set_alpha(0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_size_histogram(
    output_path: Path,
    metrics: list[BeatMetrics],
    input_path: Path,
) -> None:
    sizes = [0] + [
        HEADER_BITS + SLOT_BITS * slots
        for slots in range(BEATS_ON_BUS + 1)
    ]
    bin_edges = [sizes[0] - (sizes[1] - sizes[0]) / 2]
    bin_edges.extend((left + right) / 2 for left, right in zip(sizes, sizes[1:]))
    bin_edges.append(sizes[-1] + (sizes[-1] - sizes[-2]) / 2)

    figure, axis = plt.subplots(figsize=(10, 5))
    axis.hist([metric.occupied_bits for metric in metrics], bins=bin_edges)
    axis.set(
        title=f"Bus Beat Size Distribution: {input_path.name}",
        xlabel=f"Occupied bits (out of {BEAT_BITS})",
        ylabel="Number of beats",
        xlim=(0, BEAT_BITS),
        xticks=sizes,
    )
    axis.grid(axis="y", alpha=0.3)
    for spine in axis.spines.values():
        spine.set_alpha(0.3)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot the occupied bits in each raw 135-bit XSim beat."
    )
    parser.add_argument("hex_file", type=Path, help="Raw file containing one 135-bit beat per line")
    parser.add_argument("--plot", type=Path, help="Output image path (default: <input>_bus_fullness.png)")
    parser.add_argument(
        "--histogram",
        type=Path,
        help="Histogram image path (default: <input>_bus_beat_sizes.png)",
    )
    parser.add_argument(
        "--csv",
        type=Path,
        help="Output CSV path (default: <input>_valid_commands.csv)",
    )
    args = parser.parse_args()

    input_path = args.hex_file
    metrics = read_beats(input_path)
    plot_path = args.plot or input_path.with_name(f"{input_path.stem}_bus_fullness.png")
    histogram_path = args.histogram or input_path.with_name(
        f"{input_path.stem}_bus_beat_sizes.png"
    )
    csv_path = args.csv or input_path.with_name(f"{input_path.stem}_valid_commands.csv")
    write_valid_commands_csv(csv_path, metrics)
    plot_fullness(plot_path, metrics, input_path)
    plot_size_histogram(histogram_path, metrics, input_path)

    print(f"Plotted {len(metrics)} beats to {plot_path}")
    print(f"Bus beat size histogram saved to {histogram_path}")
    print(f"Valid command counts written to {csv_path}")


if __name__ == "__main__":
    main()
