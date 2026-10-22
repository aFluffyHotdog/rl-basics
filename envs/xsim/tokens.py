"""Decoding of the subdecoder tokens (4-bit type above a 16-bit payload)."""
from typing import NamedTuple

from .constants import CMD_TYPE, LIT_TYPE, RLE_TYPE

# One-hot index of each category in the "lookahead_types" observation.
TYPE_INDEX = {"CMD": 0, "LIT": 1, "RLE": 2, "SEED": 3}


class TokenInfo(NamedTuple):
    category: str      # "CMD", "LIT", "RLE" or "SEED"
    slots: int         # bus slots taken by the command group this token heads
    cycles: int        # estimated decode cycles (0 for LIT / SEED)
    is_barrier: bool   # CMD with lit_field == 0 and copy_len == 15


def token_type(token: int) -> int:
    return (int(token) >> 16) & 0xF


def decode_token(token: int) -> TokenInfo:
    """Parses a raw integer ONCE and returns its category, bus footprint and hardware traits."""
    t = int(token)
    payload = t & 0xFFFF
    type_code = (t >> 16) & 0xF

    if type_code == LIT_TYPE:
        return TokenInfo("LIT", 1, 0, False)
    if type_code not in (CMD_TYPE, LIT_TYPE, RLE_TYPE):
        return TokenInfo("SEED", 1, 0, False)

    is_rle = (type_code == RLE_TYPE) or (type_code == CMD_TYPE and ((payload >> 15) & 0x1 == 1))

    if is_rle:
        rle_length = (payload >> 11) & 0xF
        cycles = 2 if rle_length > 7 else 1
        return TokenInfo("RLE", 2, cycles, False)

    lit_field = (payload >> 11) & 0xF
    copy_len = (payload >> 7) & 0xF
    is_barrier = (lit_field == 0 and copy_len == 15)

    cycles = 1
    if lit_field >= 4:
        cycles = (lit_field // 4) + (1 if (lit_field % 4) else 0)

    return TokenInfo("CMD", 1 + lit_field, cycles, is_barrier)


def decode_token_slots(token: int) -> tuple[str, int]:
    """Return a token's category and the number of bus slots in its command group."""
    info = decode_token(token)
    return info.category, info.slots


def group_ticks(token) -> int:
    """Estimated decode ticks of one command group, from its head token.

    Old SubDecoder timing: 1 start tick + `cycles`. A group is one command word (CMD with its
    LITs, or RLE with its pattern word) = one CMD-FIFO entry in the RTL, which is also what
    cmd_popped counts. A stray LIT/SEED head carries no command of its own (cost 0).
    """
    info = decode_token(token)
    if info.category not in ("CMD", "RLE"):
        return 0
    return info.cycles + 1
