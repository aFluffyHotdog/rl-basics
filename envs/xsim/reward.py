"""Potential-based reward shaping from a makespan lower bound and an output-lag penalty."""
import numpy as np

from .constants import BEATS_ON_BUS, C_LAG, LAG_MARGIN, LIT_FIELDS, NUM_DECODERS
from .tokens import decode_token_slots, group_ticks


class LowerBoundTracker:
    """Lower bound (in cycles) on the remaining makespan: max(bus beats left, worst lane work left).

    Also owns the per-lane group accounting (n_sent / consumed) that flow control relies on.
    Everything is indexed by command GROUP.
    """

    def __init__(self):
        self.n_sent = None        # groups sent per queue; None until build()
        self.consumed = None      # groups whose CMD has been popped from the RTL FIFO
        self.lb_parts = None

    def build(self, cmds: list):
        """Call in reset() after the queues are loaded, before any pops."""
        self.bus_after, self.work_prefix, self.groups_total = [], [], []
        self.slot_prefix, self.beat_end = [], []
        for i, q in enumerate(cmds):
            toks = list(q)
            needs, costs = [], []
            pos = 0
            while pos < len(toks):
                need = decode_token_slots(toks[pos])[1]
                if need > BEATS_ON_BUS:
                    raise ValueError(f"queue {i}: group of {need} slots can never fit on the bus")
                need = min(need, len(toks) - pos)
                needs.append(need)
                costs.append(group_ticks(toks[pos]))
                pos += need

            G = len(needs)
            ba = np.zeros(G + 1, dtype=np.int64)                 # min beats needed from group g onward
            be = np.arange(G + 1, dtype=np.int64)                # first group after a dense beat from g
            for g in range(G - 1, -1, -1):
                used, k = 0, 0
                while g + k < G and used + needs[g + k] <= BEATS_ON_BUS:
                    used += needs[g + k]; k += 1
                ba[g] = 1 + ba[g + k]
                be[g] = g + k
            self.bus_after.append(ba)
            self.beat_end.append(be)
            self.work_prefix.append(np.concatenate([[0], np.cumsum(costs)]).astype(np.int64))
            self.slot_prefix.append(np.concatenate([[0], np.cumsum(needs)]).astype(np.int64))
            self.groups_total.append(G)

        self.n_sent = [0] * NUM_DECODERS
        self.consumed = [0] * NUM_DECODERS

        # initial-bound diagnostics: which term sets lb0?
        work0 = [int(wp[-1]) for wp in self.work_prefix]
        self.bus0 = int(sum(int(ba[0]) for ba in self.bus_after))
        self.work0_max = max(work0)
        self.work0_lane = int(np.argmax(work0))

    def record_sent(self, lane: int, groups: int):
        self.n_sent[lane] += groups

    # Per-lane quantities, in cycles / beats. `consumed` is only current after compute().
    def work_left(self, i: int) -> int:
        """Decode work of groups whose CMD the RTL has not popped yet."""
        return int(self.work_prefix[i][-1] - self.work_prefix[i][self.consumed[i]])

    def unsent_work(self, i: int) -> int:
        """Decode work of groups not yet put on the bus."""
        return int(self.work_prefix[i][-1] - self.work_prefix[i][self.n_sent[i]])

    def runway(self, i: int) -> int:
        """Decode work already sent but not yet popped (buffered in the arbiter / CMD FIFO)."""
        return int(self.work_prefix[i][self.n_sent[i]] - self.work_prefix[i][self.consumed[i]])

    def beats_left(self, i: int) -> int:
        """Minimum (dense) beats still needed to send the rest of lane i."""
        return int(self.bus_after[i][self.n_sent[i]])

    def work_between(self, i: int, g_start: int, g_end: int) -> int:
        return int(self.work_prefix[i][g_end] - self.work_prefix[i][g_start])

    def dense_beat(self, i: int, g: int) -> tuple[int, int]:
        """(slots, work) of a fully packed beat starting at group g, ignoring flow control."""
        end = self.beat_end[i][g]
        return int(self.slot_prefix[i][end] - self.slot_prefix[i][g]), self.work_between(i, g, end)

    def compute(self, hw_state: dict) -> int:
        """Call exactly once per step, after hw_state is parsed."""
        sd = hw_state.get("sub_decoders", [])
        popped = hw_state.get("cmd_popped")
        bus = sum(int(self.bus_after[i][self.n_sent[i]]) for i in range(NUM_DECODERS))

        work_max = 0
        for i in range(NUM_DECODERS):
            if popped is not None:
                c = popped[i]                                     # exact: pops reported by the RTL
            else:
                c = self.n_sent[i] - (sd[i]["cmd"] if i < len(sd) else 0)   # older testbench: infer
            c = min(max(self.consumed[i], c), self.n_sent[i])     # monotone, never above groups sent
            self.consumed[i] = c
            work_max = max(work_max, int(self.work_prefix[i][-1] - self.work_prefix[i][c]))

        self.lb_parts = (bus, work_max)
        return max(bus, work_max)


def lag_excess(hw_state: dict, cmds: list) -> int:
    """Rows by which the output spread exceeds LAG_MARGIN, over lanes that are still active.

    A lane that has sent everything and fully drained no longer takes part in the row
    synchronization, so it is excluded (otherwise a finished lane at out=0 looks like lag).
    """
    outs = []
    for i, d in enumerate(hw_state.get("sub_decoders", [])[:NUM_DECODERS]):
        active = (
            len(cmds[i]) > 0 or d["cmd"] > 0 or d["out"] > 0
            or any(d[field] > 0 for field in LIT_FIELDS)
        )
        if active:
            outs.append(d["out"])
    if len(outs) < 2:
        return 0
    return max(0, (max(outs) - min(outs)) - LAG_MARGIN)


def potential(lb: float, lag: int) -> float:
    """Phi(s) = -(LB + C_LAG * lag_excess^2), in cycles."""
    return -(float(lb) + C_LAG * lag ** 2)
