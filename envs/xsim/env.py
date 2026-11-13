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
    FAIL_PENALTY,
    LAMBDA_T,
    NO_OP_ACTION,
    NO_OP_REASONS,
    NUM_ACTIONS,
    NUM_DECODERS,
    NUM_ROWS,
    REWARD_ALPHA,
    REWARD_VERSION,
    REWARD_SCALE,
    SUB_DECODER_FIELDS,
    THROUGHPUT_WINDOW,
    TB_VIOLATION_KEYS,
    VIVADO_VERBOSE,
)
from .dataset import load_dataset_pool
from .diagnostics import log_top_not_ready_fifo_state  # noqa: F401  (see commented call in step)
from .flow_control import (
    cmd_inflight,
    compute_action_mask,
    no_op_reason,
    plan_beat,
    tb_rr_advance,
)
from .observation_v3 import build_observation_space_v3, build_observation_v3
from .protocol import IDLE_ACTION, encode_beat, pack_payload, parse_obs_line
from .reward import LowerBoundTracker
from .reward_v4 import reward_v4
from .reward_v5 import reward_v5
from .sim_client import XSimClient


class DecoderEnvXSim(gym.Env):
    def __init__(self, data_dir: str, sim_dir: str | None = None, is_inference: bool = False,
                 inflight_gate: int | None = None, backend: str = "xsim",
                 log_rtl_stop: bool = True):
        """Hardware-in-the-loop decoder scheduling env.

        Each RL step is one real scheduling decision: after the chosen beat is sent, the env
        auto-ticks the simulator with no-ops while top_in is not ready, so every observation the
        agent sees is a ready decision point (the clock is decoupled from the RL step).
        inflight_gate: if set, the mask also blocks lanes with more CMDs in flight than this
        (TB_INFLIGHT_GATE = the v20 testbench packer's rule). None = CMD FIFO depth only.
        backend: "xsim" (live Vivado, needs sim_dir) or "verilator" (in-process decomp_sim
        extension, no sim_dir; an RTL assertion ends the episode as a failure).
        log_rtl_stop: print a line when the RTL stops (False for searches that expect many stops)."""
        super().__init__()
        if backend not in ("xsim", "verilator"):
            raise ValueError(f"backend must be 'xsim' or 'verilator', got {backend!r}")
        self.backend = backend
        self.log_rtl_stop = log_rtl_stop
        self.inflight_gate = inflight_gate
        self.data_dir = Path(data_dir).resolve()
        self.sim_dir = Path(sim_dir).resolve() if sim_dir else None
        self.run_dir = self.sim_dir / "xsim_run" / "run" if self.sim_dir else None
        self.dataset_pool = load_dataset_pool(self.data_dir, is_inference)

        # Action Space: 0-7 (Regular), 8 (Yield Bus / NO_OP)
        self.action_space = gym.spaces.Discrete(NUM_ACTIONS)
        # Observation Space combining live hardware state and input lookahead
        self.observation_space = build_observation_space_v3()

        if backend == "verilator":
            from .verilator_backend import load_verilator_module, make_verilator_client
            self._rtl_stop = load_verilator_module().RtlStop
            self.sim = make_verilator_client(verbose=VIVADO_VERBOSE)
        else:
            if self.run_dir is None:
                raise ValueError("backend='xsim' needs sim_dir")
            self._rtl_stop = ()   # `except ()` catches nothing
            self.sim = XSimClient(self.run_dir, verbose=VIVADO_VERBOSE)
        self.lb = LowerBoundTracker()
        self.hw_state = {"top_in_ready": 1, "top_row_valid": 0, "mismatches": 0}
        # The action mask is needed twice per step (observation + MaskablePPO). It only depends on
        # hw_state and the pending queues, so it is computed once per state and then reused.
        self._mask_cache = None
        self._obs_batch = []   # parsed @OBS lines received since the last _tick (one per simulated clock)
        self.no_op_reason_counts = {reason: 0 for reason in NO_OP_REASONS}
        self.top_not_ready_log_path = Path("top_not_ready_fifo_states.json")
        self.completed_images = 0
        self.episodes_since_vivado_restart = 0
        self.max_episodes_per_vivado_process = 100

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        restarted = self._restart_sim_if_needed()

        if options and "sample_idx" in options:   # fixed sample, e.g. for evaluation
            selected_idx = int(options["sample_idx"]) % len(self.dataset_pool)
        else:
            selected_idx = self.np_random.integers(0, len(self.dataset_pool))
        selected_stem_path, selected_dataset = self.dataset_pool[selected_idx]
        self.current_sample_name = str(selected_stem_path.relative_to(self.data_dir))

        self.cmds = [deque(list(queue)) for queue in selected_dataset]
        self._mask_cache = None
        self.cycles = 0
        self.stall_cycles = 0
        # Testbench round-robin cursor and per-lane cycles since the last beat (teacher tracking).
        self.rr_cursor = 0
        self.cycles_since_fed = np.zeros(NUM_DECODERS, dtype=np.int64)
        # Rolling window of complete rows produced per cycle -> windowed throughput signal.
        self.throughput_window = deque(maxlen=THROUGHPUT_WINDOW)
        self._prev_total_rows = 0

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
        self._prev_total_rows = self._total_rows_decoded()
        # Reward v4 baselines: the first step is charged for every cycle since reset (leading
        # not-ready skips included), so the cycle part of the episode's summed reward is exactly -makespan / REWARD_SCALE.
        self._last_step_cycles = 0
        self._last_step_rows = self._prev_total_rows
        self._last_cycles_elapsed = 0
        self.episode_reward_sum = 0.0
        self._beats_left0 = self._prev_beats_left = self._beats_left_total()   # reward v5 potential

        # Start the episode at a real decision point: skip any leading not-ready cycles.
        self._skip_until_ready()
        return self._get_obs(), {}

    def step(self, action):
        # RtlStop = the RTL hit $stop / $fatal (an assertion on a malformed beat). Any simulated
        # cycle can raise it (_tick and _skip_until_ready both advance the clock).
        try:
            return self._step(action)
        except self._rtl_stop as e:
            if self.log_rtl_stop:
                print(f"[Env] RTL stopped: {e}")
            return self._fail(violation="rtl_stop", slots_used=0)

    def _step(self, action):
        action = int(action)
        # log_top_not_ready_fifo_state(self.top_not_ready_log_path, self.cycles, action, self.hw_state)
        in_ready = bool(self.hw_state.get("top_in_ready", 0))
        valid_bit, commands_packed, payload_128b, slots_used = self._pack_beat(action)
        self._mask_cache = None   # queues / n_sent changed

        action_hex = encode_beat(action, commands_packed, payload_128b)
        deadlock = self._tick(f"{valid_bit} {action_hex}", action if valid_bit else NO_OP_ACTION,
                              in_ready, slots_used)

        if VIVADO_VERBOSE:
            self._print_step_debug(action, commands_packed, slots_used, action_hex)

        # --- failure terminals (no shaping on these steps) ---
        self._check_hw_consistency(action, action_hex)
        if deadlock:
            remaining_cmds = [len(q) for q in self.cmds]
            print(f"[Env] Deadlock detected: stall_cycles={self.stall_cycles}, remaining_cmds={remaining_cmds}")
            return self._fail(deadlock=True, slots_used=slots_used)

        terminated = self._is_drained()
        # Decouple the clock from the RL step: tick no-ops until top_in is ready for the next
        # decision, so the returned observation is always a ready decision point.
        if not terminated and not self.hw_state.get("top_in_ready", 0):
            deadlock = self._skip_until_ready()
            if deadlock:
                return self._fail(deadlock=True, slots_used=0)
            terminated = self._is_drained()

        if terminated:
            self.completed_images += 1

        reward = self._reward()
        if terminated and REWARD_VERSION == 5:
            # Shaping telescopes: sum r = (B_left(s_0) - B_left(s_end) - makespan) / REWARD_SCALE
            expected = (self._beats_left0 - self._prev_beats_left - self.cycles) / REWARD_SCALE
            assert abs(self.episode_reward_sum - expected) < 1e-6, (
                f"episode reward {self.episode_reward_sum} != (B0 - B_end - makespan)/REWARD_SCALE {expected}")
        elif terminated and REWARD_VERSION == 4 and REWARD_ALPHA == 0.0:
            expected = -self.cycles / REWARD_SCALE
            assert abs(self.episode_reward_sum - expected) < 1e-6, (
                f"episode reward {self.episode_reward_sum} != -makespan/REWARD_SCALE {expected}")
        return self._get_obs(), reward, terminated, False, self._get_info(terminated, slots_used)

    def action_masks(self) -> np.ndarray:
        if self._mask_cache is None:
            self._mask_cache = compute_action_mask(self.hw_state, self.cmds, self.lb.n_sent, self.inflight_gate)
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

    def _tick(self, action_str: str, sent_lane: int, in_ready: bool, slots_used: int) -> bool:
        """Advance exactly one testbench cycle with action_str and update all per-cycle bookkeeping.

        lb.compute() must run once per @OBS line to keep the free-run row counter exact, so it (and
        the throughput window and the schedule / stall counters) lives here, not once per RL step.
        Returns True if this cycle tripped the deadlock guard.
        """
        self._obs_batch = []
        self.sim.write_action(action_str)
        self._advance_sim()
        # One @OBS line = one simulated clock. xsim can run several clocks per pause (the sim skips
        # timesteps itself), so count the lines actually received; verilator always gives exactly 1.
        batch, self._obs_batch = self._obs_batch, []
        self.cycles += max(1, len(batch))
        # Replay the skipped clocks so lb's free-run row counter sees every cycle, then the final state.
        for obs in batch[:-1]:
            self.lb.compute({**self.hw_state, **obs})

        lb_new = self.lb.compute(self.hw_state)
        if DEBUG_CHECKS:
            assert lb_new <= self.lb_prev, f"lower bound increased: {self.lb_prev} -> {lb_new}"
        self.lb_prev = lb_new

        self._update_throughput()
        self._track_schedule(sent_lane, in_ready)
        return self._update_stall(slots_used)

    def _skip_until_ready(self) -> bool:
        """Tick no-ops while top_in is not ready, until a decision point, drain, or deadlock.

        Returns True if the deadlock guard tripped during the skip."""
        while not self.hw_state.get("top_in_ready", 0) and not self._is_drained():
            self.no_op_reason_counts["top_in_not_ready"] += 1
            if self._tick(IDLE_ACTION, NO_OP_ACTION, in_ready=False, slots_used=0):
                return True
        return False

    def _total_rows_decoded(self) -> int:
        return sum(self.lb.rows_decoded(i, self.hw_state) for i in range(NUM_DECODERS))

    def _update_throughput(self):
        """Push this cycle's count of newly completed rows (all lanes) into the rolling window."""
        total = self._total_rows_decoded()
        self.throughput_window.append(max(0, total - self._prev_total_rows))
        self._prev_total_rows = total

    def _on_obs(self, line: str):
        self._mask_cache = None   # hardware state changed -> recompute mask / mask observation
        obs = parse_obs_line(line)
        self._obs_batch.append(obs)
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

    def _track_schedule(self, sent_lane: int, in_ready: bool):
        """Advance the testbench round-robin cursor and the per-lane cycles-since-fed counters."""
        self.rr_cursor = tb_rr_advance(self.rr_cursor, sent_lane, in_ready)
        self.cycles_since_fed += 1
        if sent_lane != NO_OP_ACTION:
            self.cycles_since_fed[sent_lane] = 0

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

    def _pin_skew_mismatches(self) -> int:
        """Skew-aware output-pin mismatches (verilator backend only; 0 on xsim)."""
        return int(self.sim.sim.pin_skew_mismatches) if self.backend == "verilator" and self.sim.sim else 0

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

        # The testbench's output-pin counter ("mismatches") compares every lane with reference row k
        # on the k-th pin pulse, but the free-run wrapper lets lanes drift apart, so it fires on a
        # correct decode. Use the per-lane golden check instead; on the verilator backend also the
        # skew-aware pin check. Only an old testbench without lane_mismatches falls back to it.
        if "lane_mismatches" in self.hw_state:
            bad = self._delta("lane_mismatches") > 0 or self._pin_skew_mismatches() > 0
        else:
            bad = self.hw_state["mismatches"] > 0
        if bad:
            print(f"[Env] Hardware Mismatch Detected! lane mismatches since reset: "
                  f"{self._delta('lane_mismatches')}, skew-aware pin mismatches: {self._pin_skew_mismatches()}")
            print(f"[Env] action_hex: {action_hex}")

    def _update_stall(self, slots_used: int) -> bool:
        """Track consecutive empty beats while work remains; True once that looks like a deadlock."""
        if slots_used == 0 and any(len(q) > 0 for q in self.cmds):
            self.stall_cycles += 1
        else:
            self.stall_cycles = 0
        return self.stall_cycles > DEADLOCK_STALL_CYCLES

    def _is_drained(self) -> bool:
        """All queues sent, every subdecoder FIFO / output row empty, and every lane has output all
        of its rows.

        Empty FIFOs alone are not enough: an executor can still be producing rows from a command it
        has already popped (e.g. a run of repeated rows), which no FIFO counter shows. Without the
        row check the episode ended up to ~12 cycles early. rows_consumed is the testbench's
        slowest-lane row count; an older testbench without it falls back to the FIFO check."""
        if sum(len(q) for q in self.cmds) != 0:
            return False
        if "rows_consumed" in self.hw_state and self.hw_state["rows_consumed"] < NUM_ROWS:
            return False
        sub_decoders = self.hw_state.get("sub_decoders", [])
        return (
            self.hw_state.get("top_row_valid", 0) == 0
            and len(sub_decoders) == NUM_DECODERS
            and all(decoder[f] == 0 for decoder in sub_decoders for f in SUB_DECODER_FIELDS)
        )

    def _fail(self, *, deadlock=False, mismatch=False, violation=None, slots_used: int):
        """Failure terminal. Each step costs LAMBDA_T, so crashing would otherwise be cheaper than
        the >= lb_now cycles still needed: charge those plus FAIL_PENALTY episodes' worth."""
        lb_now = self.lb.compute(self.hw_state)
        reward = -(FAIL_PENALTY * self.lb0 + lb_now) * LAMBDA_T / REWARD_SCALE
        info = {
            "deadlock": deadlock,
            "mismatch": mismatch,
            "violation": violation,
            "slots_used": slots_used,
            "cycle": self.cycles,
            "sample_name": self.current_sample_name,
        }
        return self._get_obs(), reward, True, False, info

    # ------------------------------------------------------------------ reward / obs / info
    def _beats_left_total(self) -> int:
        """B_left(s): dense beats still needed to send what is left, summed over lanes."""
        return sum(self.lb.beats_left(i) for i in range(NUM_DECODERS))

    def _reward(self) -> float:
        """Reward v5 (default): -cycles since the previous env step (incl. skipped cycles) plus the
        drop in B_left over the step, all / REWARD_SCALE. REWARD_VERSION = 4 gives the old reward
        (-cycles + alpha * new rows).

        The row counter is advanced in _tick (once per cycle), so this just reads it."""
        cycles_elapsed = self.cycles - self._last_step_cycles
        rows_done_delta = self._prev_total_rows - self._last_step_rows
        self._last_step_cycles = self.cycles
        self._last_step_rows = self._prev_total_rows
        self._last_cycles_elapsed = cycles_elapsed
        beats_now = self._beats_left_total()
        if REWARD_VERSION == 5:
            reward = reward_v5(cycles_elapsed, self._prev_beats_left, beats_now)
        else:
            reward = reward_v4(cycles_elapsed, rows_done_delta, REWARD_ALPHA)
        self._prev_beats_left = beats_now
        self.episode_reward_sum += reward
        return reward

    def _get_obs(self):
        return build_observation_v3(self.hw_state, self.cmds, self.lb, self.action_masks(),
                                    sum(self.throughput_window))

    def _get_info(self, terminated: bool, slots_used: int) -> dict:
        info = {
            "slots_used": slots_used,
            "cycle": self.cycles,
            "cycles_elapsed": self._last_cycles_elapsed,   # clock cycles this env step advanced
            "sample_name": self.current_sample_name,
        }
        if not terminated:
            return info
        info.update({
            "makespan": self.cycles,
            "images_completed": self.completed_images,
            "lb0": self.lb0,
            "lb_bus0": self.lb.bus0,
            "lb_work0": self.lb.work0_max,
            "lb_work0_lane": self.lb.work0_lane,
            "lane_mismatches": self._delta("lane_mismatches"),
            "tailbusy_hits": self._delta("tailbusy_hits"),
            "cmd_overcommit": self._delta("cmd_overcommit"),
            "proto_err": self._delta("proto_err"),
            "ready_viol": self._delta("ready_viol"),
        })
        return info
