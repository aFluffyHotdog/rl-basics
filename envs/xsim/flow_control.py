"""Beat packing and action masking: which lanes can accept a beat right now."""
import numpy as np

from .constants import (
    BEATS_ON_BUS,
    CMD_FIFO_SIZE,
    LIT_FIFO_CAPACITIES,
    LIT_TYPE,
    NO_OP_ACTION,
    NUM_ACTIONS,
    NUM_DECODERS,
    TB_RR_CHANNELS,
)
from .tokens import decode_token_slots, token_type


def cmd_inflight(hw_state: dict, n_sent: list, i: int) -> int:
    """Commands sent to lane i and not yet popped from its CMD FIFO.

    Counts commands still travelling through the arbiter FIFO / tail / sub_rx, which FIFO
    occupancy alone cannot see. Falls back to occupancy if the testbench is the older one.
    """
    popped = hw_state.get("cmd_popped")
    if popped is not None:
        return n_sent[i] - popped[i]
    return hw_state["sub_decoders"][i]["cmd"]


def plan_beat(queue, sub_decoder: dict, inflight: int) -> tuple[int, int, int]:
    """Dry run of the packer for one lane -> (groups, tokens, slots) the next beat would carry.

    Single source of truth: the action mask enables a lane iff this packs at least one group,
    and step() packs exactly what this returns, so the mask can never offer an action that
    silently sends nothing.

    Return Values:
     - groups: no. of whole commands a beat carries
     - tokens: how many items are popped from a lane's queue
     - slots: no. of slots it will use on the bus
    """
    groups = pos = slots = pending_lits = 0

    # keep adding a cmd till the bus is full (only calculates, does not interact with the bus)
    while pos < len(queue) and slots < BEATS_ON_BUS:
        _, needed = decode_token_slots(queue[pos])
        # Stop when the next whole command group would overflow the beat. Matches the v20 testbench
        # packer (blk_packer_v17 / deployment/testbench_but_python.py TestbenchPacker._pack): it is
        # order-preserving and breaks on the first group that does not fit, never skipping ahead.
        if slots + needed > BEATS_ON_BUS:
            break
        # if the amount of slots needed is actually more than how much data there are, something's gone wrong buddy
        if needed > len(queue) - pos:
            break
        # check if all the inflight commands + the number of commands we packed + this next one will exceed the FIFO buffer
        if inflight + groups + 1 > CMD_FIFO_SIZE:
            break
        group_lits = sum(token_type(queue[pos + k]) == LIT_TYPE for k in range(needed))
        # NOTE: jerry's TB actually does not check the LIT fifo banks
        # if any(
        #     sub_decoder[f"lit{bank}"] + pending_lits + group_lits > capacity
        #     for bank, capacity in enumerate(LIT_FIFO_CAPACITIES, start=1)
        # ):
        #     break

        # increment all the counters
        slots += needed
        pos += needed
        pending_lits += group_lits
        groups += 1
    return groups, pos, slots


def packable_slots_by_lane(hw_state: dict, cmds: list, n_sent: list, inflight_gate: int | None = None) -> list[int]:
    """Return slots packable on each currently selectable lane, or zero if blocked.

    inflight_gate: if set, a lane with more CMDs in flight than this is blocked as well (the v20
    testbench packer uses TB_INFLIGHT_GATE). None keeps the CMD-FIFO-depth limit only.
    """
    slots_by_lane = [0] * NUM_DECODERS
    if hw_state.get("top_in_ready", 0) == 0:
        return slots_by_lane

    tail_busy = hw_state.get("tail_busy")
    for i in range(NUM_DECODERS):
        if not cmds[i]:
            continue
        # The arbiter dispatches on tail_busy alone; a beat sent to a busy lane can sit at the
        # head of its FIFO and block every beat behind it.
        if tail_busy is not None and tail_busy[i]:
            continue
        inflight = cmd_inflight(hw_state, n_sent, i)
        if inflight_gate is not None and inflight > inflight_gate:
            continue
        _, _, slots = plan_beat(cmds[i], hw_state["sub_decoders"][i], inflight)
        slots_by_lane[i] = slots
    return slots_by_lane


def tb_rr_pick(rr: int, eligible) -> int | None:
    """First eligible lane scanning the testbench cursor rr, rr+1, ... mod TB_RR_CHANNELS, else None.

    The bypass slot (index 8) takes a turn in the scan but is never eligible (disabled in the TB).
    """
    for k in range(TB_RR_CHANNELS):
        s = (rr + k) % TB_RR_CHANNELS
        if s < NUM_DECODERS and eligible[s]:
            return s
    return None


def tb_rr_advance(rr: int, action: int, in_ready: bool) -> int:
    """Testbench cursor after a cycle: served lane + 1; +1 on an idle cycle with the bus ready;
    unchanged while top_in_ready is low."""
    if action != NO_OP_ACTION:
        return (action + 1) % TB_RR_CHANNELS
    if in_ready:
        return (rr + 1) % TB_RR_CHANNELS
    return rr


def compute_action_mask(hw_state: dict, cmds: list, n_sent: list, inflight_gate: int | None = None) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    slots_by_lane = packable_slots_by_lane(hw_state, cmds, n_sent, inflight_gate)
    mask[:NUM_DECODERS] = np.asarray(slots_by_lane) > 0
    if not mask[:NUM_DECODERS].any():
        mask[NO_OP_ACTION] = True
    return mask


def no_op_reason(hw_state: dict, mask: np.ndarray) -> str:
    """Classify why the agent yielded the bus (see constants.NO_OP_REASONS)."""
    if hw_state.get("top_in_ready", 0) == 0:
        return "top_in_not_ready"
    if not mask[:NUM_DECODERS].any():
        return "no_valid_subdecoder_actions"
    return "chosen_while_actions_valid"
