import csv
from pathlib import Path

import pytest

from deployment.plot_bus_fullness import (
    BEAT_BITS,
    BeatMetrics,
    decode_beat,
    plot_fullness,
    plot_size_histogram,
    read_beats,
    write_valid_commands_csv,
)


def _encode_beat(destination: int, command_count: int, payload_words: list[int]) -> str:
    count_field = max(command_count - 1, 0) if destination != 8 else 0
    payload = sum(word << (16 * index) for index, word in enumerate(payload_words))
    return f"{(destination << 131) | (count_field << 128) | payload:034x}"


def test_decode_beat_counts_command_groups_and_slot_footprint():
    cmd_with_one_literal = 1 << 11
    rle_head = 1 << 15
    beat = _encode_beat(2, 2, [cmd_with_one_literal, 0xABCD, rle_head, 0x1234])

    assert decode_beat(beat, 1) == BeatMetrics(valid_commands=2, occupied_bits=71)


def test_decode_beat_counts_no_op_as_empty():
    assert decode_beat(_encode_beat(8, 0, []), 1) == BeatMetrics(0, 0)


def test_decode_beat_rejects_group_exceeding_payload():
    beat = _encode_beat(0, 1, [15 << 11])

    with pytest.raises(ValueError, match="exceeds payload capacity"):
        decode_beat(beat, 1)


def test_csv_has_only_valid_commands_column(tmp_path):
    metrics = [BeatMetrics(2, 71), BeatMetrics(0, 0)]
    output_path = tmp_path / "counts.csv"

    write_valid_commands_csv(output_path, metrics)

    with output_path.open(newline="", encoding="utf-8") as csv_file:
        rows = list(csv.reader(csv_file))
    assert rows == [["valid_commands"], ["2"], ["0"]]


def test_read_beats_ignores_trailing_comments_and_comment_only_lines(tmp_path):
    input_path = tmp_path / "beats.hex"
    input_path.write_text(
        f"{_encode_beat(2, 1, [0])} // DATA cycle=1 ready=1\n"
        "// comment-only line\n"
        f"{_encode_beat(8, 0, [])}\n",
        encoding="ascii",
    )

    metrics = read_beats(input_path)

    assert [metric.valid_commands for metric in metrics] == [1, 0]


def test_sample_hex_file_decodes_and_plots(tmp_path):
    sample_path = Path(__file__).resolve().parents[1] / "100k_model_row_barrier.hex"
    metrics = read_beats(sample_path)
    plot_path = tmp_path / "fullness.png"
    histogram_path = tmp_path / "sizes.png"

    assert metrics
    assert all(0 <= metric.occupied_bits <= BEAT_BITS for metric in metrics)
    plot_fullness(plot_path, metrics, sample_path)
    plot_size_histogram(histogram_path, metrics, sample_path)
    assert plot_path.is_file()
    assert histogram_path.is_file()
