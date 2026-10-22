"""Optional debugging dumps (not called during normal training)."""
import json
from pathlib import Path

from .constants import NO_OP_ACTION


def log_top_not_ready_fifo_state(log_path: Path, cycles: int, action: int, hw_state: dict):
    """Append the FIFO state to a JSON file, keyed by cycle, whenever top_in_ready is low."""
    if hw_state["top_in_ready"] != 0:
        return

    events = {}
    if log_path.exists():
        with log_path.open("r", encoding="utf-8") as json_file:
            events = json.load(json_file)
        if not isinstance(events, dict):
            raise ValueError(f"Expected a JSON object in {log_path}")

    action_name = "no_op" if action == NO_OP_ACTION else f"decoder_{action}"
    events[str(cycles)] = {
        "action": action,
        "action_name": action_name,
        "top_in_ready": hw_state["top_in_ready"],
        "top_row_valid": hw_state["top_row_valid"],
        "mismatches": hw_state["mismatches"],
        "subdecoders": [
            {"subdecoder": decoder, **fifo_state}
            for decoder, fifo_state in enumerate(hw_state["sub_decoders"])
        ],
    }
    with log_path.open("w", encoding="utf-8") as json_file:
        json.dump(events, json_file, indent=2)
        json_file.write("\n")
