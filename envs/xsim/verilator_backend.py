"""Verilator backend: a drop-in replacement for XSimClient backed by the decomp_sim extension.

The extension and its client live in verilator_harness/python (build with `make` there; see
INSTALL_macOS.md). Importing is lazy so the xsim path never needs the compiled module.
"""
import sys
from pathlib import Path

HARNESS_PYTHON = Path(__file__).resolve().parents[2] / "verilator_harness" / "python"


def load_verilator_module():
    """Returns the decomp_sim module (for decomp_sim.RtlStop)."""
    if str(HARNESS_PYTHON) not in sys.path:
        sys.path.insert(0, str(HARNESS_PYTHON))
    try:
        import decomp_sim
    except ImportError as e:
        raise ImportError(
            f"cannot import decomp_sim from {HARNESS_PYTHON}: {e}\n"
            "Build it first (verilator_harness/Makefile, see INSTALL_macOS.md) with the same "
            "python you train with.") from e
    return decomp_sim


def make_verilator_client(verbose: bool = False):
    load_verilator_module()
    from verilator_sim_client import VerilatorSimClient
    return VerilatorSimClient(verbose=verbose)
