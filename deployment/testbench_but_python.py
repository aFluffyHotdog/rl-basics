"""Drive one image through XSim with the v20 testbench's BUS_MODEL=2 packer -- no RL agent.

Python port of `blk_packer_v17` from tb_top_complete_v20_real_8dec_1image.sv, stepped over the same
Tcl ping-pong as envs/xsim/env.py (tb_interactive.sv). Each cycle it reads the @OBS line and makes
the decision the testbench packer makes at negedge:

  * idle while top_in_ready is low (round-robin cursor unchanged);
  * scan channels round-robin from the cursor; a lane is eligible iff it has CMDs left, its arbiter
    tail is not busy, CMDs in flight (committed - popped) <= 8, and (optionally) the skew gate holds;
  * pack as many consecutive {CMD + its LITs / RLE word} groups as fit in the 8 payload slots,
    order-preserving, num_cmd-1 in the control field;
  * cursor -> served channel + 1, or cursor + 1 when nothing was sent.

Differences from the SystemVerilog testbench:
  * The bypass channel (sub-dec 9) is never fed: tb_interactive.sv builds with reg_disable_bypass=1.
    The cursor still runs modulo 9, as the TB does with the bypass gated off (DROP_BYPASS).
  * Wall cycles count from the first step after reset. The TB counts from test_run, after a
    50-cycle preload with an idle packer, so walls can differ by a cycle or two.
  * tail_busy / cmd_popped / rows_consumed come from @OBS instead of hierarchical references.
"""
import argparse
import sys
from pathlib import Path
from typing import NamedTuple

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from envs.xsim.constants import (
    BEATS_ON_BUS,
    CMD_TYPE,
    DEADLOCK_STALL_CYCLES,
    LIT_TYPE,
    NO_OP_ACTION,
    NUM_DECODERS,
    NUM_ROWS,
    RLE_TYPE,
    ROW_BYTES,
    SUB_DECODER_FIELDS,
    TB_INFLIGHT_GATE,
    TB_VIOLATION_KEYS,
)
from envs.xsim.dataset import _stem_path, load_hex_folder
from envs.xsim.flow_control import cmd_inflight, tb_rr_advance, tb_rr_pick
from envs.xsim.protocol import encode_beat, pack_payload, parse_obs_line
from envs.xsim.sim_client import XSimClient
from envs.xsim.tokens import token_type

DESER_BASELINE = NUM_ROWS * 8
CHECK_KEYS = TB_VIOLATION_KEYS + ("lane_mismatches",)


class Beat(NamedTuple):
    dest: int
    ncmd: int
    payload: int
    slots: int


# ---------------------------------------------------------------------- data prep
def split_groups(tokens: list[int], lane: int) -> list[list[int]]:
    """Split a lane's token stream into CMD groups: a CMD token followed by the chunks it consumes.

    Checks the same things the TB loader / packer / tb_interactive check_beat rely on: an RLE CMD
    owns exactly one RLE word, an LC CMD owns exactly lit_field LITs, and a group fits in 8 slots.
    """
    groups = []
    for pos, token in enumerate(tokens):
        kind = token_type(token)
        if kind == CMD_TYPE:
            groups.append([token])
        elif kind in (LIT_TYPE, RLE_TYPE):
            if not groups:
                raise ValueError(f"lane {lane}: token {pos} ({token:05x}) precedes the first CMD")
            groups[-1].append(token)
        else:
            raise ValueError(f"lane {lane}: unknown token type {kind} at index {pos} ({token:05x})")

    for ci, group in enumerate(groups):
        cw = group[0] & 0xFFFF
        chunk_types = [token_type(t) for t in group[1:]]
        if cw >> 15:
            ok = chunk_types == [RLE_TYPE]
            expected = "1 RLE word"
        else:
            lit_field = (cw >> 11) & 0xF
            ok = chunk_types == [LIT_TYPE] * lit_field
            expected = f"{lit_field} LITs"
        if not ok:
            raise ValueError(f"lane {lane} CMD {ci} ({cw:04x}): expected {expected}, got types {chunk_types}")
        if len(group) > BEATS_ON_BUS:
            raise ValueError(f"lane {lane} CMD {ci} ({cw:04x}) needs {len(group)} slots -- can never fit in "
                             f"an 8-slot beat (encoder bug, cap lit_field <= 7)")
    return groups


def cmd_out_bytes(cw: int) -> int:
    """Output bytes of one CMD (mirrors the TB's cmd_out_bytes / new_dec_sub_128b_cmd_disp.v)."""
    if cw >> 15:
        return ((cw >> 11) & 0xF) + 1
    lit_field = (cw >> 11) & 0xF
    copy_len = (cw >> 7) & 0xF
    if lit_field == 0 and copy_len == 0xF:
        return 0
    return 2 * lit_field + (0 if copy_len == 0xF else copy_len + 2)


def cmd_rows(groups: list[list[int]]) -> list[int]:
    """rows[ci] = rows fully delivered before CMD ci, plus a sentinel (the TB's v19 cmd_row[]).

    Not tokens.row_delta: the TB resets the byte accumulator on ROW_REPEAT, row_delta does not.
    """
    rows, acc, out = 0, 0, []
    for group in groups:
        out.append(rows)
        cw = group[0] & 0xFFFF
        if (cw >> 15) == 0 and ((cw >> 11) & 0xF) == 0 and ((cw >> 7) & 0xF) == 0xF:
            rows += (cw & 0x7F) + 1   # ROW_REPEAT
            acc = 0
        else:
            acc += cmd_out_bytes(cw)
            if acc >= ROW_BYTES:
                acc = 0
                rows += 1
    out.append(rows)
    return out


# ---------------------------------------------------------------------- packer
class TestbenchPacker:
    """blk_packer_v17 (BUS_MODEL=2) with the observation dict in place of hierarchical references."""

    def __init__(self, groups: list[list[list[int]]], skew_bound: int | None = None):
        self.groups = groups
        self.rows_before = [cmd_rows(lane) for lane in groups]
        self.n_sent = [0] * NUM_DECODERS   # pk_cmd_idx: CMD groups committed per lane
        self.rr = 0                        # pk_rr
        self.skew_bound = skew_bound

    def remaining(self) -> int:
        return sum(len(lane) - sent for lane, sent in zip(self.groups, self.n_sent))

    def _skew_ok(self, s: int, hw_state: dict) -> bool:
        """skew_ok(): a lane may not run more than skew_bound rows ahead of the barrier's pops."""
        if self.skew_bound is None:
            return True
        return self.rows_before[s][self.n_sent[s]] - hw_state["rows_consumed"] < self.skew_bound

    def _eligible(self, s: int, hw_state: dict) -> bool:
        if self.n_sent[s] >= len(self.groups[s]):
            return False
        tail_busy = hw_state.get("tail_busy")
        if tail_busy is not None and tail_busy[s]:
            return False
        # (pk_cmd_idx - cmd_popped) <= 8, so a beat of <= 8 CMDs never overfills the 16-deep CMD FIFO.
        if cmd_inflight(hw_state, self.n_sent, s) > TB_INFLIGHT_GATE:
            return False
        return self._skew_ok(s, hw_state)

    def _pack(self, s: int) -> Beat:
        """Greedily pack consecutive CMD groups (order-preserving) into the 8 slots."""
        tokens = []
        ncmd = 0
        for group in self.groups[s][self.n_sent[s]:]:
            if len(tokens) + len(group) > BEATS_ON_BUS:
                break
            tokens.extend(group)
            ncmd += 1
        payload, slots = pack_payload(tokens)
        self.n_sent[s] += ncmd
        return Beat(s, ncmd, payload, slots)

    def choose(self, hw_state: dict) -> tuple[Beat | None, str]:
        """Beat for this cycle, or (None, idle reason)."""
        if not hw_state.get("top_in_ready", 0):
            return None, "top_in_not_ready"

        # Cursor scan over the 8 lanes + the bypass slot (disabled in tb_interactive.sv).
        lane = tb_rr_pick(self.rr, [self._eligible(s, hw_state) for s in range(NUM_DECODERS)])
        self.rr = tb_rr_advance(self.rr, NO_OP_ACTION if lane is None else lane, in_ready=True)
        if lane is None:
            return None, "no_eligible_lane"
        return self._pack(lane), ""


# ---------------------------------------------------------------------- sim loop
def _resolve_beats_dir(image_dir: Path) -> Path:
    image_dir = image_dir.resolve()
    beats_dir = image_dir if image_dir.name == "beats_hex" else image_dir / "beats_hex"
    rows_dir = beats_dir.parent / "rows_hex"
    if not beats_dir.is_dir():
        raise ValueError(f"Expected a beats_hex folder at {beats_dir}")
    if not rows_dir.is_dir() or not any(rows_dir.glob("*.hex")):
        raise ValueError(f"Expected pre-generated row hex files in {rows_dir}")
    return beats_dir


def _is_done(hw_state: dict, packer: TestbenchPacker) -> bool:
    """TB pass end: NUM_ROWS rows consumed (barrier pops, or the slowest lane's rows in free-run).

    Falls back to the env's drained check on an older testbench without the extended @OBS fields.
    """
    if "rows_consumed" in hw_state:
        return hw_state["rows_consumed"] >= NUM_ROWS
    sub_decoders = hw_state.get("sub_decoders", [])
    return (
        packer.remaining() == 0
        and hw_state.get("top_row_valid", 0) == 0
        and len(sub_decoders) == NUM_DECODERS
        and all(decoder[f] == 0 for decoder in sub_decoders for f in SUB_DECODER_FIELDS)
    )


def run(
    image_dir: Path,
    sim_dir: Path,
    output_path: Path,
    skew_bound: int | None = None,
    max_cycles: int = 50000,
    verbose: bool = False,
) -> dict:
    if max_cycles <= 0:
        raise ValueError("max_cycles must be greater than zero")
    beats_dir = _resolve_beats_dir(image_dir)
    lanes = load_hex_folder(beats_dir)
    groups = [split_groups(list(queue), lane) for lane, queue in enumerate(lanes)]
    packer = TestbenchPacker(groups, skew_bound)
    stem = str(_stem_path(beats_dir)).replace("beats_hex", "rows_hex")
    print(f"Loaded {beats_dir}: CMDs per lane = {[len(lane) for lane in groups]}")

    hw_state = {}

    def advance():
        sim.resume()
        sim.wait_for_obs(lambda line: hw_state.update(parse_obs_line(line)))

    sim = XSimClient(sim_dir.resolve() / "xsim_run" / "run", verbose=verbose)
    accepted = []
    beats_per_lane = [0] * NUM_DECODERS
    idle = {"top_in_not_ready": 0, "no_eligible_lane": 0}
    slots_sent = 0
    wall = None
    stall = 0
    reported = set()
    try:
        # Handshake, as in DecoderEnvXSim.reset(): @HANDSHAKE_READY -> stem path -> first @OBS.
        sim.start()
        advance()
        sim.write_action(stem)
        advance()

        if "tail_busy" not in hw_state:
            print("[TB-PY] WARNING: testbench does not report tail_busy / cmd_popped / rows_consumed; "
                  "tail gate is off and in-flight falls back to CMD FIFO occupancy.")
        if skew_bound is not None and "rows_consumed" not in hw_state:
            raise RuntimeError("--skew-bound needs rows_consumed in @OBS (extended tb_interactive.sv)")
        base = {k: hw_state.get(k, 0) for k in CHECK_KEYS}

        for cycle in range(1, max_cycles + 1):
            beat, idle_reason = packer.choose(hw_state)
            if beat is None:
                idle[idle_reason] += 1
                action_hex = encode_beat(NO_OP_ACTION, 0, 0)
            else:
                action_hex = encode_beat(beat.dest, beat.ncmd, beat.payload)
            sim.write_action(f"{0 if beat is None else 1} {action_hex}")
            advance()

            if beat is not None:
                accepted.append(action_hex)
                beats_per_lane[beat.dest] += 1
                slots_sent += beat.slots
                if verbose:
                    print(f"[TB-PY] cyc {cycle}: lane {beat.dest} ncmd={beat.ncmd} slots={beat.slots} {action_hex}")

            # The packer applies the same gates as check_beat, so any increase is a port bug.
            for key in CHECK_KEYS:
                if key not in reported and hw_state.get(key, 0) > base[key]:
                    reported.add(key)
                    print(f"[TB-PY] Testbench check failed at cycle {cycle}: {key} "
                          f"(+{hw_state[key] - base[key]}) last beat={action_hex}")

            if _is_done(hw_state, packer):
                wall = cycle
                break

            stall = stall + 1 if beat is None and packer.remaining() else 0
            if stall > DEADLOCK_STALL_CYCLES:
                print(f"[TB-PY] Deadlock: {stall} cycles without a beat at cycle {cycle}, "
                      f"CMDs left per lane = {[len(g) - n for g, n in zip(groups, packer.n_sent)]}")
                break
    finally:
        sim.close()

    output_path = output_path.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("".join(f"{h}\n" for h in accepted), encoding="ascii")

    cycles = cycle if wall is None else wall
    rows = hw_state.get("rows_consumed", NUM_ROWS if wall else 0)
    deltas = {k: hw_state.get(k, 0) - base[k] for k in CHECK_KEYS}
    print()
    print(f">>> [v20 TDM SYNC] wall = {wall if wall is not None else 'n/a'} cyc, rows = {rows}, "
          f"mismatches = {hw_state.get('mismatches', 0)}")
    if wall:
        print(f"    speedup vs deser    = {DESER_BASELINE / wall:.2f}x ({DESER_BASELINE} / {wall})")
    print(f"    beats sent          = {len(accepted)} over {cycles} cycles "
          f"(avg {slots_sent / max(len(accepted), 1):.2f} / {BEATS_ON_BUS} slots per beat)")
    print(f"    idle cycles         = {idle}")
    print(f"    beats per lane      = {beats_per_lane}")
    print(f"    CMDs sent per lane  = {packer.n_sent}")
    print(f"    testbench counters  = {deltas}")
    print(f"Accepted bus beats saved to: {output_path}")
    if wall is None:
        raise RuntimeError(f"Image did not finish within {cycles} cycles.")
    return {"wall": wall, "beats": len(accepted), "idle": idle, "beats_per_lane": beats_per_lane, **deltas}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the v20 testbench's BUS_MODEL=2 round-robin packer on one image in XSim (no RL)."
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
        default=Path("tb_python_bus_beats.txt"),
        help="One 34-hex line per accepted beat (same format as the TB's DUMP_BUS bus_beats.txt).",
    )
    parser.add_argument(
        "--skew-bound",
        type=int,
        default=None,
        help="Enable the TB's skew gate (only meaningful on a BARRIER_EN build; the TB default is 4).",
    )
    parser.add_argument("--max-cycles", type=int, default=50000)
    parser.add_argument("--verbose", action="store_true", help="Print every beat and the raw XSim output.")
    args = parser.parse_args()

    run(
        image_dir=args.image_dir,
        sim_dir=args.sim_dir,
        output_path=args.output,
        skew_bound=args.skew_bound,
        max_cycles=args.max_cycles,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
