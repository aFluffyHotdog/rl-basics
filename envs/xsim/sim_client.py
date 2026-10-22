"""Vivado xsim process management and the Tcl ping-pong handshake."""
import os
import subprocess
from pathlib import Path
from typing import Callable

from torch import signal

from .protocol import IDLE_ACTION


class XSimClient:
    """Owns the live xsim process. Each step: write action.txt -> `run -all` -> read until @OBS."""

    def __init__(self, run_dir: Path, verbose: bool = False):
        self.run_dir = run_dir
        self.action_file = self.run_dir / "action.txt"
        self.verbose = verbose
        self.sim_proc = None

    @property
    def is_running(self) -> bool:
        return self.sim_proc is not None

    def start(self):
        self.close()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        self.sim_proc = subprocess.Popen(
            ["xsim.bat", "tb_rl_interactive_snap", "-tclbatch", "quiet_run.tcl", "-R"],
            cwd=str(self.run_dir),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            creationflags=CREATE_NEW_PROCESS_GROUP
        )

    def restart(self):
        self.sim_proc.stdin.write("restart\n")
        self.sim_proc.stdin.flush()

    def write_action(self, content: str):
        with open(self.action_file, "w") as f:
            f.write(f"{content}\n")

    def resume(self):
        self.sim_proc.stdin.write("run -all\n")
        self.sim_proc.stdin.flush()

    def wait_for_obs(self, on_obs: Callable[[str], None]) -> str:
        """Read sim output until it pauses; every @OBS line is passed to on_obs."""
        found_obs = False
        while True:
            line = self.sim_proc.stdout.readline()
            if self.verbose: print(f"[XSIM] {line.strip()}")

            if "err" in line:
                print(f"[XSIM ERROR] {line.strip()}")
            if not line:
                raise RuntimeError("Simulation crashed or closed unexpectedly.")

            if line.startswith("@HANDSHAKE_READY"):
                return line

            if line.startswith("@OBS"):
                on_obs(line)
                if self.verbose: print(f"[XSIM] Observation received: {line.strip()}")
                found_obs = True
                continue

            if "$stop called at time" in line:
                if found_obs: return line
                else: continue

    def close(self):
        if not self.sim_proc:
            return

        try:
            if self.sim_proc.poll() is None:
                try:
                    self.write_action(IDLE_ACTION)
                    self.resume()
                except Exception:
                    pass

                try:
                    if self.sim_proc.stdin and not self.sim_proc.stdin.closed:
                        self.sim_proc.stdin.write("quit\n")
                        self.sim_proc.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass

                try:
                    self.sim_proc.wait(timeout=2)
                except Exception:
                    try:
                        self.sim_proc.terminate()
                        self.sim_proc.wait(timeout=2)
                    except Exception:
                        pass
        finally:
            try:
                os.kill(self.sim_proc.pid, signal.CTRL_BREAK_EVENT)
                self.sim_proc.wait(timeout=2)
                if self.sim_proc.stdin and not self.sim_proc.stdin.closed:
                    self.sim_proc.stdin.close()
            except Exception:
                pass
            self.sim_proc = None
