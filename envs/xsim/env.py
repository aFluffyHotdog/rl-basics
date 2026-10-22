"""
Hardware-in-the-Loop Environment for the T8 Decompressor.
Uses Tcl-Ping-Pong to step a live Vivado SystemVerilog simulation.
"""
from collections import deque
from pathlib import Path

import gymnasium as gym
import numpy as np

from .constants import (
    DEADLOCK_STALL_CYCLES,
    DEBUG_CHECKS,
    EXT_OBS_KEYS,
    FAIL_ON_LANE_MISMATCH,
    FAIL_PENALTY,
    GAMMA,
    NO_OP_ACTION,
    NO_OP_REASONS,
    NUM_ACTIONS,
    NUM_DECODERS,
    SUB_DECODER_FIELDS,
    TB_VIOLATION_KEYS,
    VIVADO_VERBOSE,
)
from .dataset import load_dataset_pool
from .diagnostics import log_top_not_ready_fifo_state  # noqa: F401  (see commented call in step)
from .flow_control import cmd_inflight, compute_action_mask, no_op_reason, plan_beat
from .observation import build_observation, build_observation_space
from .observation_v2 import build_observation_space_v2, build_observation_v2
from .protocol import encode_beat, pack_payload, parse_obs_line
from .reward import LowerBoundTracker, lag_excess, potential
from .sim_client import XSimClient


OBS_VERSIONS = ("v1", "v2")


class DecoderEnvXSim(gym.Env):
    def __init__(self, data_dir: str, sim_dir: str, is_inference: bool = False, obs_version: str = "v2"):
        """obs_version: "v2" = compact per-lane features (observation_v2.py);
        "v1" = original token-level lookahead, needed to load models trained before v2."""
        super().__init__()
        if obs_version not in OBS_VERSIONS:
            raise ValueError(f"obs_version must be one of {OBS_VERSIONS}, got {obs_version!r}")
        self.obs_version = obs_version
        self.data_dir = Path(data_dir).resolve()
        self.sim_dir = Path(sim_dir).resolve()
        self.run_dir = self.sim_dir / "xsim_run" / "run"
        self.dataset_pool = load_dataset_pool(self.data_dir, is_inference)

        # Action Space: 0-7 (Regular), 8 (Yield Bus / NO_OP)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        # Observation Space combining live hardware state and input lookahead
        self.observation_space = (
            build_observation_space_v2() if obs_version == "v2" else build_observation_space()
        )

        self.sim = XSimClient(self.run_dir, verbose=VIVADO_VERBOSE)
        self.lb = LowerBoundTracker()
        self.hw_state = {"top_in_ready": 1, "top_row_valid": 0, "mismatches": 0}
        # The action mask is needed twice per step (observation + MaskablePPO). It only depends on
        # hw_state and the pending queues, so it is computed once per state and then reused.
        self._mask_cache = None
        self.no_op_reason_counts = {reason: 0 for reason in NO_OP_REASONS}
        self.top_not_ready_log_path = Path("top_not_ready_fifo_states.json")
        self.completed_images = 0
        self.episodes_since_vivado_restart = 0
        self.max_episodes_per_vivado_process = 100

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        restarted = self._restart_sim_if_needed()

        selected_idx = self.np_random.integers(0, len(self.dataset_pool))
        selected_stem_path, selected_dataset = self.dataset_pool[selected_idx]
        self.current_sample_name = str(selected_stem_path.relative_to(self.data_dir))

        self.cmds = [deque(list(queue)) for queue in selected_dataset]
        self._mask_cache = None
        self.initial_tokens = sum(len(cmd) for cmd in self.cmds)
        self.cycles = 0
        self.stall_cycles = 0

        if not self.sim.is_running:
            self.sim.start()
        elif not restarted:
            self.sim.restart()

        self._advance_sim()

        stem_str = str(selected_stem_path).replace("\\", "/")
        stem_str = str(selected_stem_path).replace("beats_hex", "rows_hex")
        self.sim.write_action(stem_str)
        self._advance_sim()

        # Testbench violation counters start from whatever the sim reports right after init.
        self._chk_base = {k: self.hw_state.get(k, 0) for k in TB_VIOLATION_KEYS + ("lane_mismatches",)}

        self.lb.build(self.cmds)
        self.lb_prev = self.lb.compute(self.hw_state)
        self.lb0 = max(self.lb_prev, 1)
        self.phi_prev = self._phi(self.lb_prev)

        return self._get_obs(), {}

    def step(self, action):
        self.cycles += 1
        action = int(action)
        # log_top_not_ready_fifo_state(self.top_not_ready_log_path, self.cycles, action, self.hw_state)

        valid_bit, commands_packed, payload_128b, slots_used = self._pack_beat(action)
        self._mask_cache = None   # queues / n_sent changed

        action_hex = encode_beat(action, commands_packed, payload_128b)
        self.sim.write_action(f"{valid_bit} {action_hex}")
        self._advance_sim()

        if VIVADO_VERBOSE:
            self._print_step_debug(action, commands_packed, slots_used, action_hex)

        # --- failure terminals (no shaping on these steps) ---
        self._check_hw_consistency(action, action_hex)

        if self._update_stall(slots_used):
            remaining_cmds = [len(q) for q in self.cmds]
            print(f"[Env] Deadlock detected: stall_cycles={self.stall_cycles}, remaining_cmds={remaining_cmds}")
            return self._fail(deadlock=True)

        # --- completion ---
        terminated = self._is_drained()
        if terminated:
            self.completed_images += 1

        reward = self._shaped_reward(terminated)
        return self._get_obs(), reward, terminated, False, self._get_info(terminated)

    def action_masks(self) -> np.ndarray:
        if self._mask_cache is None:
            self._mask_cache = compute_action_mask(self.hw_state, self.cmds, self.lb.n_sent)
        return self._mask_cache.copy()

    def close(self):
        self.sim.close()

    # ------------------------------------------------------------------ simulator I/O
    def _restart_sim_if_needed(self):
        """Restart the live Vivado simulator once every N episodes to avoid drift/lockups."""
        self.episodes_since_vivado_restart += 1
        if self.episodes_since_vivado_restart < self.max_episodes_per_vivado_process:
            return False

        self.sim.start()
        self.episodes_since_vivado_restart = 0
        return True

    def _advance_sim(self):
        """Let the sim run until its next pause, updating hw_state from every @OBS line."""
        self.sim.resume()
        self.sim.wait_for_obs(self._on_obs)

    def _on_obs(self, line: str):
        self._mask_cache = None   # hardware state changed -> recompute mask / mask observation
        obs = parse_obs_line(line)
        self.hw_state.update(obs)
        for key in EXT_OBS_KEYS:   # older testbench: drop stale extended fields
            if key not in obs:
                self.hw_state.pop(key, None)

    # ------------------------------------------------------------------ action handling
    def _cmd_inflight(self, i: int) -> int:
        return cmd_inflight(self.hw_state, self.lb.n_sent, i)

    def _pack_beat(self, action: int) -> tuple[int, int, int, int]:
        """Pop the tokens for this action -> (valid_bit, commands_packed, payload_128b, slots_used)."""
        if action == NO_OP_ACTION:
            self.no_op_reason_counts[no_op_reason(self.hw_state, self.action_masks())] += 1
            return 0, 0, 0, 0

        # Regular lanes: whole command groups packed into 16-bit slots, exactly as planned.
        groups, n_tokens, _ = plan_beat(
            self.cmds[action], self.hw_state["sub_decoders"][action], self._cmd_inflight(action)
        )
        tokens = [self.cmds[action].popleft() for _ in range(n_tokens)]
        payload_128b, slots_used = pack_payload(tokens)
        self.lb.record_sent(action, groups)

        valid_bit = 1 if groups > 0 else 0   # never send an empty beat flagged valid
        return valid_bit, groups, payload_128b, slots_used

    def _print_step_debug(self, action_dest, commands_packed, slots_used, action_hex):
        print(f"[Env] action_dest: {action_dest}, commands_packed: {commands_packed}, slots_used: {slots_used}")
        print(f"[Env] action: {action_hex}")
        decoder_rows = self.hw_state.get("sub_decoders", [])
        if 0 <= action_dest < len(decoder_rows):
            for i in range(NUM_DECODERS):
                fields = ", ".join(f"{f}={decoder_rows[i][f]}" for f in SUB_DECODER_FIELDS)
                print(f"[Env] current subdecoder {i}: {fields}")
                print()
        else:
            print(f"[Env] current subdecoder {action_dest}: unavailable")

    # ------------------------------------------------------------------ failure checks
    def _delta(self, key: str) -> int:
        """Testbench counter change since reset (the sim restarts, but don't rely on it)."""
        return self.hw_state.get(key, 0) - self._chk_base.get(key, 0)

    def _check_tb_counters(self):
        """Return the name of the first testbench violation counter that has increased, else None."""
        for key in TB_VIOLATION_KEYS:
            if self._delta(key) > 0:
                return key
        return None

    def _check_hw_consistency(self, action: int, action_hex: str):
        """Report testbench violations / mismatches and assert CMD accounting. Does not terminate."""
        # Testbench flow-control / format checks. The mask enforces the same gates, so any increase
        # means the mask or packer disagrees with the RTL; the beat is already suspect.
        violation = self._check_tb_counters()
        if violation:
            print(f"[Env] Testbench check failed: {violation} (+{self._delta(violation)}) "
                  f"action={action} action_hex={action_hex}")

        if DEBUG_CHECKS and "cmd_popped" in self.hw_state:
            for i in range(NUM_DECODERS):
                assert self.hw_state["cmd_popped"][i] <= self.lb.n_sent[i], (
                    f"lane {i}: RTL popped {self.hw_state['cmd_popped'][i]} CMDs but only "
                    f"{self.lb.n_sent[i]} groups were sent -- group != CMD-FIFO entry, check the unit")

        if self.hw_state["mismatches"] > 0 or (FAIL_ON_LANE_MISMATCH and self._delta("lane_mismatches") > 0):
            print(f"[Env] Hardware Mismatch Detected! Total Mismatches: {self.hw_state['mismatches']}, "
                  f"lane mismatches since reset: {self._delta('lane_mismatches')}")
            print(f"[Env] action_hex: {action_hex}")

    def _update_stall(self, slots_used: int) -> bool:
        """Track consecutive empty beats while work remains; True once that looks like a deadlock."""
        if slots_used == 0 and any(len(q) > 0 for q in self.cmds):
            self.stall_cycles += 1
        else:
            self.stall_cycles = 0
        return self.stall_cycles > DEADLOCK_STALL_CYCLES

    def _is_drained(self) -> bool:
        """All queues sent and every subdecoder FIFO / output row empty."""
        if sum(len(q) for q in self.cmds) != 0:
            return False
        sub_decoders = self.hw_state.get("sub_decoders", [])
        return (
            self.hw_state.get("top_row_valid", 0) == 0
            and len(sub_decoders) == NUM_DECODERS
            and all(decoder[f] == 0 for decoder in sub_decoders for f in SUB_DECODER_FIELDS)
        )

    def _fail(self, *, deadlock=False, mismatch=False, violation=None):
        """Failure terminal. No shaping term here: using Phi(terminal)=0 on a failure would pay the
        agent the whole remaining potential for crashing. The penalty exceeds any progress credit
        already earned, so failing is always worse than finishing."""
        lb_now = self.lb.compute(self.hw_state)
        reward = -(FAIL_PENALTY + lb_now / self.lb0)
        info = {"deadlock": deadlock, "mismatch": mismatch, "violation": violation}
        return self._get_obs(), reward, True, False, info

    # ------------------------------------------------------------------ reward / obs / info
    def _phi(self, lb: float) -> float:
        """Potential in cycles. Takes lb as an argument so the LB is computed once per step."""
        return potential(lb, lag_excess(self.hw_state, self.cmds))

    def _shaped_reward(self, terminated: bool) -> float:
        # reward = (-1 + gamma * Phi(s') - Phi(s)) / lb0, Phi = -(LB + C_LAG * lag_excess^2), Phi(terminal) = 0
        lb_new = self.lb.compute(self.hw_state)
        if DEBUG_CHECKS:
            assert lb_new <= self.lb_prev, f"lower bound increased: {self.lb_prev} -> {lb_new}"
            if terminated:
                assert lb_new == 0, f"episode finished but lb={lb_new} (parts: {self.lb.lb_parts})"
                if "cmd_popped" in self.hw_state:
                    assert self.hw_state["cmd_popped"] == self.lb.n_sent, (
                        f"drained but RTL pops {self.hw_state['cmd_popped']} != groups sent {self.lb.n_sent}")
        phi_new = 0.0 if terminated else self._phi(lb_new)
        reward = (-1.0 + GAMMA * phi_new - self.phi_prev) / self.lb0
        self.phi_prev = phi_new
        self.lb_prev = lb_new
        return reward

    def _get_obs(self):
        if self.obs_version == "v2":
            return build_observation_v2(self.hw_state, self.cmds, self.lb, self.lb0, self.action_masks())
        return build_observation(self.hw_state, self.cmds, self.initial_tokens, self.lb.n_sent, self.action_masks)

    def _get_info(self, terminated: bool) -> dict:
        if not terminated:
            return {}
        return {
            "makespan": self.cycles,
            "images_completed": self.completed_images,
            "sample_name": self.current_sample_name,
            "lb0": self.lb0,
            "lb_bus0": self.lb.bus0,
            "lb_work0": self.lb.work0_max,
            "lb_work0_lane": self.lb.work0_lane,
            "lane_mismatches": self._delta("lane_mismatches"),
            "tailbusy_hits": self._delta("tailbusy_hits"),
            "cmd_overcommit": self._delta("cmd_overcommit"),
            "proto_err": self._delta("proto_err"),
            "ready_viol": self._delta("ready_viol"),
        }
