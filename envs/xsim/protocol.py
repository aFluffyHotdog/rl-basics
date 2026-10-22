"""Text protocol spoken with tb_interactive.sv: @OBS parsing and 135-bit beat encoding."""
from .constants import (
    BASE_OBS_LEN,
    EXT_OBS_LEN,
    NO_OP_ACTION,
    NUM_DECODERS,
    SUB_DECODER_FIELDS,
)

# Idle beat (valid=0, all-zero payload), written before quitting the sim.
IDLE_ACTION = "0 " + "0" * 34


def parse_obs_line(obs_str: str) -> dict:
    """Parses the minimum required physical signals from an XSIM @OBS line.

    The extended keys (EXT_OBS_KEYS) are only present when the testbench reports them.
    """
    parts = obs_str.split()[1:]
    obs = {
        "top_in_ready": int(parts[0]),
        "top_row_valid": int(parts[1]),
        "mismatches": int(parts[2]),
    }

    sub_decoders = []
    idx = 3
    for _ in range(NUM_DECODERS):
        sub_decoders.append({
            field: int(parts[idx + k]) for k, field in enumerate(SUB_DECODER_FIELDS)
        })
        idx += len(SUB_DECODER_FIELDS)
    obs["sub_decoders"] = sub_decoders

    # Extended fields (testbench with arbiter / flow-control checks). Older testbenches stop at 51.
    if len(parts) >= EXT_OBS_LEN:
        b = BASE_OBS_LEN
        obs["tail_busy"] = [int(x) for x in parts[b:b + NUM_DECODERS]]
        obs["cmd_popped"] = [int(x) for x in parts[b + NUM_DECODERS:b + 2 * NUM_DECODERS]]
        (obs["rows_out"], obs["rows_consumed"], obs["lane_mismatches"],
         obs["proto_err"], obs["tailbusy_hits"], obs["cmd_overcommit"],
         obs["ready_viol"]) = (int(x) for x in parts[b + 2 * NUM_DECODERS:EXT_OBS_LEN])
    return obs


def pack_payload(tokens) -> tuple[int, int]:
    """Pack tokens into 16-bit slots of the 128-bit payload -> (payload_128b, slots_used)."""
    payload_128b = 0
    slots_used = 0
    for token in tokens:
        clean_16b_token = token & 0xFFFF
        payload_128b = payload_128b | (clean_16b_token << (slots_used * 16))
        slots_used += 1
    return payload_128b, slots_used


def encode_beat(action_dest: int, commands_packed: int, payload_128b: int) -> str:
    """Encode dest[4] | ncmd-1[3] | payload[128] as a 34-digit hex string."""
    dest_bin = f"{action_dest:04b}"

    if action_dest == NO_OP_ACTION or commands_packed == 0:
        ncmd_bin = "000"
    else:
        ncmd_bin = f"{commands_packed - 1:03b}"

    payload_bin = f"{payload_128b:0128b}"
    full_135b_string = dest_bin + ncmd_bin + payload_bin
    return f"{int(full_135b_string, 2):034x}"
