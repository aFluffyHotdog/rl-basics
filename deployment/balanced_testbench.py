"""Drive one image through XSim with a row-balancing packer -- no RL agent.

Same sim loop, eligibility gates and beat packing as deployment/testbench_but_python.py (the v20 TB
packer port). Only the lane choice differs: the packer feeds the eligible lane with the fewest
*projected rows*, so the rows produced per lane stay as even as possible on every cycle.

  projected rows[s] = rows lane s has output + rows its in-flight CMDs will output
                    = rows fully covered by the CMDs already sent to lane s

The second form is exact and needs no hardware feedback (TestbenchPacker.projected_rows()).
Hardware state is still used for eligibility: CMDs left, arbiter tail not busy, <= 8 CMDs in flight.

  * If the lowest lane is blocked, the next-lowest eligible lane is fed (never idles on purpose).
  * --max-lead N: a lane is also blocked while it is N or more projected rows ahead of the lowest
    unfinished lane. The bus then idles until the lowest lane catches up (idle reason "lead_cap").
  * Ties on projected rows break at random (--seed), or in TB round-robin cursor order (--tie-break rr).

  python deployment/balanced_testbench.py --image-dir sample_data_9_lanes/test_11_lantern --sim-dir <sim>
"""
import argparse
import random
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from deployment.testbench_but_python import Beat, TestbenchPacker, run
from envs.xsim.constants import NO_OP_ACTION, NUM_DECODERS
from envs.xsim.flow_control import tb_rr_advance, tb_rr_pick

TIE_BREAKS = ("random", "rr")


class BalancedPacker(TestbenchPacker):
    """TestbenchPacker that serves the eligible lane with the fewest projected rows."""

    def __init__(self, groups: list[list[list[int]]], max_lead: int | None = None,
                 tie_break: str = "random", seed: int | None = None):
        super().__init__(groups)
        if max_lead is not None and max_lead < 1:
            raise ValueError("max_lead must be at least 1 (0 would block every lane)")
        if tie_break not in TIE_BREAKS:
            raise ValueError(f"tie_break must be one of {TIE_BREAKS}, got {tie_break!r}")
        self.max_lead = max_lead
        self.tie_break = tie_break
        self.rng = random.Random(seed)

    def _floor(self, projected: list[int]) -> int:
        """Projected rows of the lowest lane that still has CMDs to send (finished lanes drop out)."""
        unfinished = [projected[s] for s in range(NUM_DECODERS) if self.n_sent[s] < len(self.groups[s])]
        return min(unfinished, default=0)

    def choose(self, hw_state: dict) -> tuple[Beat | None, str]:
        """Beat for this cycle, or (None, idle reason)."""
        if not hw_state.get("top_in_ready", 0):
            return None, "top_in_not_ready"   # cursor holds, as in the TB

        projected = self.projected_rows()
        free = [s for s in range(NUM_DECODERS) if self._eligible(s, hw_state)]
        eligible = free
        if self.max_lead is not None:
            floor = self._floor(projected)
            eligible = [s for s in free if projected[s] - floor < self.max_lead]

        if not eligible:
            self.rr = tb_rr_advance(self.rr, NO_OP_ACTION, in_ready=True)
            return None, "lead_cap" if free else "no_eligible_lane"

        low = min(projected[s] for s in eligible)
        tied = [s for s in eligible if projected[s] == low]
        if self.tie_break == "rr":
            lane = tb_rr_pick(self.rr, [s in tied for s in range(NUM_DECODERS)])
        else:
            lane = self.rng.choice(tied)
        self.rr = tb_rr_advance(self.rr, lane, in_ready=True)
        return self._pack(lane), ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a packer that keeps projected output rows per lane balanced, on one image in XSim."
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
        default=Path("tb_balanced_bus_beats.txt"),
        help="One 34-hex line per accepted beat (same format as the TB's DUMP_BUS bus_beats.txt).",
    )
    parser.add_argument(
        "--max-lead",
        type=int,
        default=None,
        help="Block a lane while it is this many projected rows ahead of the lowest unfinished lane "
             "(default: off, always feed the lowest eligible lane).",
    )
    parser.add_argument("--tie-break", choices=TIE_BREAKS, default="random",
                        help="How to choose among lanes tied on projected rows.")
    parser.add_argument("--seed", type=int, default=None,
                        help="Seed for --tie-break random (default: a fresh seed, printed for replay).")
    parser.add_argument("--max-cycles", type=int, default=50000)
    parser.add_argument("--verbose", action="store_true", help="Print every beat and the raw XSim output.")
    args = parser.parse_args()

    seed = args.seed
    if args.tie_break == "random" and seed is None:
        seed = random.randrange(2 ** 32)
    print(f"Balanced packer: max_lead={args.max_lead} tie_break={args.tie_break}"
          + (f" seed={seed}" if args.tie_break == "random" else ""))

    run(
        image_dir=args.image_dir,
        sim_dir=args.sim_dir,
        output_path=args.output,
        max_cycles=args.max_cycles,
        verbose=args.verbose,
        make_packer=lambda groups: BalancedPacker(groups, args.max_lead, args.tie_break, seed),
    )


if __name__ == "__main__":
    main()
