from collections import deque

import numpy as np

from deployment import xsim_single_lane_test as slt
from envs.xsim.constants import NO_OP_ACTION, NUM_ACTIONS, NUM_DECODERS, SUB_DECODER_FIELDS

LANE = 2
TAIL_BUSY_CYCLES = 2


class FakeSim:
    """Toy testbench for one lane: a beat makes the tail busy for TAIL_BUSY_CYCLES clocks, a beat
    that arrives while it is busy counts a tailbusy hit, and one CMD is popped per clock.
    clocks_per_resume > 1 models the testbench running extra clocks and re-driving the same beat."""

    def __init__(self, clocks_per_resume=1):
        self.clocks_per_resume = clocks_per_resume
        self.busy = 0
        self.committed = 0
        self.popped = 0
        self.hits = 0
        self.beat = (0, 0)   # (valid, groups)

    def drive(self, valid, groups):
        self.beat = (valid, groups)

    def wait_for_obs(self, on_obs):
        for _ in range(self.clocks_per_resume):
            valid, groups = self.beat
            if valid:
                if self.busy:
                    self.hits += 1
                self.busy = TAIL_BUSY_CYCLES
                self.committed += groups
            else:
                self.busy = max(0, self.busy - 1)
            if self.committed > self.popped:
                self.popped += 1
            on_obs(self.obs())

    def obs(self):
        tail_busy = [0] * NUM_DECODERS
        tail_busy[LANE] = int(self.busy > 0)
        popped = [0] * NUM_DECODERS
        popped[LANE] = self.popped
        subs = [{f: 0 for f in SUB_DECODER_FIELDS} for _ in range(NUM_DECODERS)]
        subs[LANE]["cmd"] = self.committed - self.popped
        return {"top_in_ready": 1, "tail_busy": tail_busy, "cmd_popped": popped,
                "sub_decoders": subs, "tailbusy_hits": self.hits, "proto_err": 0,
                "cmd_overcommit": 0, "ready_viol": 0, "lane_mismatches": 0}


class FakeEnv:
    """The parts of DecoderEnvXSim the single-lane test uses. Every beat carries one group."""

    def __init__(self, n_beats=5, clocks_per_resume=1):
        self.n_beats = n_beats
        self.clocks_per_resume = clocks_per_resume

    def reset(self, options=None):
        self.sim = FakeSim(self.clocks_per_resume) if not hasattr(self, "sim") else self.sim
        self.sim.__init__(self.clocks_per_resume)
        self.cmds = [deque() for _ in range(NUM_DECODERS)]
        self.cmds[LANE].extend(range(self.n_beats))
        self.cmds[0].extend(range(3))   # another lane with work, never fed
        self.lb = type("LB", (), {})()
        self.lb.n_sent = [0] * NUM_DECODERS
        self.cycles = 0
        self.hw_state = self.sim.obs()
        return {}, {}

    def action_masks(self):
        mask = np.zeros(NUM_ACTIONS, dtype=bool)
        for i in range(NUM_DECODERS):
            mask[i] = bool(self.cmds[i]) and not self.hw_state["tail_busy"][i]
        mask[NO_OP_ACTION] = not mask[:NUM_DECODERS].any()
        return mask

    def step(self, action):
        if action != NO_OP_ACTION and self.cmds[action]:
            self.cmds[action].popleft()
            self.lb.n_sent[action] += 1
            self.sim.drive(1, 1)
        else:
            self.sim.drive(0, 0)
        self.sim.wait_for_obs(self._on_obs)
        self.cycles += 1
        return {}, 0.0, False, False, {}

    def _on_obs(self, obs):
        self.hw_state = obs

    def _delta(self, key):
        return self.hw_state.get(key, 0)


def _run(mode, clocks_per_resume=1):
    env = FakeEnv(clocks_per_resume=clocks_per_resume)
    env.reset()
    counter = slt.count_obs_lines(env)
    return slt.run_single_lane(env, 0, LANE, mode, max_cycles=200, obs_counter=counter)


def test_mask_mode_waits_for_the_tail_and_is_clean():
    result = _run("mask")
    assert result["status"] == "ok"
    assert result["beats_sent"] == 5
    assert result["anomaly_counts"] == {}
    assert result["counter_increases"] == {}
    assert slt.passed(result)
    assert all(t["obs_lines"] == 1 for t in result["trace"])


def test_bypass_mode_hits_are_blamed_on_sending_to_a_visible_busy_tail():
    result = _run("bypass")
    assert result["status"] == "ok"
    assert result["counter_increases"]["tailbusy_hits"] == 4   # every beat after the first
    assert result["anomaly_counts"] == {"sent_to_busy_tail": 4}
    assert "bypassing the mask" in slt.verdict(result)
    assert not slt.passed(result)


def test_extra_clocks_per_resume_are_reported_as_a_sync_problem():
    result = _run("mask", clocks_per_resume=2)
    counts = result["anomaly_counts"]
    assert counts["obs_lines_not_1"] == len(result["trace"])
    assert counts["hit_unseen_by_python"] >= 1        # the re-driven copy hits the busy tail
    assert counts["popped_more_than_sent"] >= 1       # and its CMD is popped too
    assert "disagree" in slt.verdict(result)


def test_classify_flags_a_testbench_hit_on_a_cycle_python_sent_nothing():
    assert slt.classify(sent=False, tail_busy_seen=False, new_counts={"tailbusy_hits": 1},
                        obs_lines=1, popped=0, n_sent=0) == ["hit_without_send"]


def test_lane_drained_needs_pops_and_empty_fifos():
    env = FakeEnv(n_beats=0)
    env.reset()
    assert slt.lane_drained(env, LANE)
    env.lb.n_sent[LANE] = 1
    assert not slt.lane_drained(env, LANE)
