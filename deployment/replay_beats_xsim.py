"""Replay a recorded beat file into XSim: next beat whenever top_in_ready is 1, idle otherwise.

The beat file can come from the v20 testbench's DUMP_BUS (bus_beats.txt), testbench_but_python.py
--output, or infer_xsim.py --output. No scheduling and no tail_busy / CMD in-flight gating: the
recorded stream is sent as-is, and the testbench's checker counters are reported so a stream recorded
under different timing shows where it would have tripped the arbiter / FIFO gates.

Accepted line formats (blank lines and lines starting with '#' are ignored):
  <hex>            34-digit {dest[4], ncmd-1[3], payload[128]}  (bus_beats.txt / infer_xsim output)
  <valid> <hex>    action.txt format; valid=0 lines are dropped
Dropped beats: dest >= 8 (no-op records, or bypass beats -- the bypass is disabled in
tb_interactive.sv) and beats with an all-zero payload (empty actions recorded by infer_xsim; sending
them would push a bogus CMD 0000).
"""
import argparse
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.xsim.constants import (
    DEADLOCK_STALL_CYCLES,
    NO_OP_ACTION,
    NUM_DECODERS,
    NUM_ROWS,
    SUB_DECODER_FIELDS,
    TB_VIOLATION_KEYS,
)
from envs.xsim.dataset import _stem_path
from envs.xsim.protocol import encode_beat, parse_obs_line
from envs.xsim.sim_client import XSimClient

BEAT_HEX_DIGITS = 34
BEAT_BITS = 135
PAYLOAD_MASK = (1 << 128) - 1
DESER_BASELINE = NUM_ROWS * 8
CHECK_KEYS = TB_VIOLATION_KEYS + ("lane_mismatches",)
IDLE_HEX = encode_beat(NO_OP_ACTION, 0, 0)


def _resolve_beats_dir(image_dir: Path) -> Path:
    image_dir = image_dir.resolve()
    beats_dir = image_dir if image_dir.name == "beats_hex" else image_dir / "beats_hex"
    rows_dir = beats_dir.parent / "rows_hex"
    if not beats_dir.is_dir():
        raise ValueError(f"Expected a beats_hex folder at {beats_dir}")
    if not rows_dir.is_dir() or not any(rows_dir.glob("*.hex")):
        raise ValueError(f"Expected pre-generated row hex files in {rows_dir}")
    return beats_dir


def load_beats(path: Path) -> list[str]:
    """Read the beat file -> normalised 34-digit hex beats for lanes 0..7, in file order."""
    beats = []
    dropped = {"valid=0": 0, "dest>=8": 0, "empty payload": 0}
    for line_no, line in enumerate(path.read_text(encoding="ascii").splitlines(), start=1):
        parts = line.split()
        if not parts or parts[0].startswith("#"):
            continue
        if len(parts) == 2:
            if parts[0] not in ("0", "1"):
                raise ValueError(f"{path}:{line_no}: valid bit must be 0 or 1, got {parts[0]!r}")
            if parts[0] == "0":
                dropped["valid=0"] += 1
                continue
            hex_str = parts[1]
        elif len(parts) == 1:
            hex_str = parts[0]
        else:
            raise ValueError(f"{path}:{line_no}: expected '<hex>' or '<valid> <hex>', got {line!r}")

        try:
            value = int(hex_str, 16)
        except ValueError as exc:
            raise ValueError(f"{path}:{line_no}: invalid hex {hex_str!r}") from exc
        if len(hex_str) > BEAT_HEX_DIGITS or value >> BEAT_BITS:
            raise ValueError(f"{path}:{line_no}: beat {hex_str!r} is wider than {BEAT_BITS} bits")

        if value >> 131 >= NUM_DECODERS:
            dropped["dest>=8"] += 1
            continue
        if value & PAYLOAD_MASK == 0:
            dropped["empty payload"] += 1
            continue
        beats.append(f"{value:0{BEAT_HEX_DIGITS}x}")

    per_lane = [0] * NUM_DECODERS
    for beat in beats:
        per_lane[int(beat, 16) >> 131] += 1
    print(f"Loaded {len(beats)} beats from {path} (dropped: {dropped}), beats per lane = {per_lane}")
    if not beats:
        raise ValueError(f"No beats to replay in {path}")
    return beats


def _is_done(hw_state: dict, all_sent: bool) -> bool:
    """TB pass end: NUM_ROWS rows consumed (barrier pops, or the slowest lane's rows in free-run).

    Falls back to the env's drained check on an older testbench without the extended @OBS fields.
    """
    if "rows_consumed" in hw_state:
        return hw_state["rows_consumed"] >= NUM_ROWS
    sub_decoders = hw_state.get("sub_decoders", [])
    return (
        all_sent
        and hw_state.get("top_row_valid", 0) == 0
        and len(sub_decoders) == NUM_DECODERS
        and all(decoder[f] == 0 for decoder in sub_decoders for f in SUB_DECODER_FIELDS)
    )


def _progress(hw_state: dict) -> tuple:
    """Anything that moves while the DUT is still draining."""
    return (
        hw_state.get("rows_consumed"),
        hw_state.get("rows_out"),
        tuple(hw_state.get("cmd_popped", ())),
        tuple(d[f] for d in hw_state.get("sub_decoders", []) for f in SUB_DECODER_FIELDS),
    )


def replay(
    beats_path: Path,
    image_dir: Path,
    sim_dir: Path,
    max_cycles: int = 50000,
    verbose: bool = False,
) -> dict:
    if max_cycles <= 0:
        raise ValueError("max_cycles must be greater than zero")
    beats = load_beats(beats_path.resolve())
    stem = str(_stem_path(_resolve_beats_dir(image_dir))).replace("beats_hex", "rows_hex")

    hw_state = {}

    def advance():
        sim.resume()
        sim.wait_for_obs(lambda line: hw_state.update(parse_obs_line(line)))

    sim = XSimClient(sim_dir.resolve() / "xsim_run" / "run", verbose=verbose)
    sent = 0
    beats_per_lane = [0] * NUM_DECODERS
    not_ready_cycles = 0
    last_beat_cycle = None
    wall = None
    stall = 0
    reported = set()
    try:
        # Handshake, as in DecoderEnvXSim.reset(): @HANDSHAKE_READY -> stem path -> first @OBS.
        sim.start()
        advance()
        sim.write_action(stem)
        advance()
        base = {k: hw_state.get(k, 0) for k in CHECK_KEYS}
        last_progress = _progress(hw_state)

        for cycle in range(1, max_cycles + 1):
            beat = None
            if sent < len(beats):
                if hw_state.get("top_in_ready", 0):
                    beat = beats[sent]
                else:
                    not_ready_cycles += 1
            sim.write_action(f"1 {beat}" if beat else f"0 {IDLE_HEX}")
            advance()

            if beat:
                sent += 1
                beats_per_lane[int(beat, 16) >> 131] += 1
                last_beat_cycle = cycle
                if verbose:
                    print(f"[REPLAY] cyc {cycle}: beat {sent}/{len(beats)} {beat}")

            # Reported only: the replay ignores tail_busy / CMD in-flight on purpose.
            for key in CHECK_KEYS:
                if key not in reported and hw_state.get(key, 0) > base[key]:
                    reported.add(key)
                    print(f"[REPLAY] Testbench counter {key} rose at cycle {cycle} "
                          f"(+{hw_state[key] - base[key]}), last beat sent = {beats[sent - 1] if sent else None}")

            if _is_done(hw_state, sent == len(beats)):
                wall = cycle
                break

            # After the last beat, stop once the DUT stops making progress.
            progress = _progress(hw_state)
            stall = stall + 1 if sent == len(beats) and progress == last_progress else 0
            last_progress = progress
            if stall > DEADLOCK_STALL_CYCLES:
                print(f"[REPLAY] No progress for {stall} cycles after the last beat (cycle {cycle}); "
                      f"the stream does not decode to the full image.")
                break
    finally:
        sim.close()

    cycles = cycle if wall is None else wall
    rows = hw_state.get("rows_consumed", NUM_ROWS if wall else 0)
    deltas = {k: hw_state.get(k, 0) - base[k] for k in CHECK_KEYS}
    print()
    print(f">>> [REPLAY] wall = {wall if wall is not None else 'n/a'} cyc, rows = {rows}, "
          f"mismatches = {hw_state.get('mismatches', 0)}")
    if wall:
        print(f"    speedup vs deser     = {DESER_BASELINE / wall:.2f}x ({DESER_BASELINE} / {wall})")
    print(f"    beats sent           = {sent} / {len(beats)}, last beat at cycle {last_beat_cycle}")
    print(f"    cycles not ready     = {not_ready_cycles} (beats pending, top_in_ready = 0)")
    print(f"    beats per lane       = {beats_per_lane}")
    print(f"    testbench counters   = {deltas}")
    if wall is None:
        raise RuntimeError(f"Image did not finish within {cycles} cycles.")
    return {"wall": wall, "beats": sent, "not_ready_cycles": not_ready_cycles, **deltas}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay a recorded beat hex file into XSim, sending a beat whenever top_in_ready is 1."
    )
    parser.add_argument("--beats", type=Path, required=True, help="Beat file, one beat per line.")
    parser.add_argument(
        "--image-dir",
        type=Path,
        required=True,
        help="Sample folder containing beats_hex/ and rows_hex/ (or beats_hex/ itself); "
             "the testbench loads this image's golden rows.",
    )
    parser.add_argument("--sim-dir", type=Path, required=True, help="Path to the XSim simulation directory.")
    parser.add_argument("--max-cycles", type=int, default=50000)
    parser.add_argument("--verbose", action="store_true", help="Print every beat and the raw XSim output.")
    args = parser.parse_args()

    replay(
        beats_path=args.beats,
        image_dir=args.image_dir,
        sim_dir=args.sim_dir,
        max_cycles=args.max_cycles,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
