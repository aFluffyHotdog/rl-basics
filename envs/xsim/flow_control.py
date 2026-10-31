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
        # TODO: this is sus, check what the test bench will do
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


def packable_slots_by_lane(hw_state: dict, cmds: list, n_sent: list) -> list[int]:
    """Return slots packable on each currently selectable lane, or zero if blocked."""
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
        _, _, slots = plan_beat(
            cmds[i],
            hw_state["sub_decoders"][i],
            cmd_inflight(hw_state, n_sent, i),
        )
        slots_by_lane[i] = slots
    return slots_by_lane


def compute_action_mask(hw_state: dict, cmds: list, n_sent: list) -> np.ndarray:
    mask = np.zeros(NUM_ACTIONS, dtype=bool)
    slots_by_lane = packable_slots_by_lane(hw_state, cmds, n_sent)
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
