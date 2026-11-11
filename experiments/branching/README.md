# Decision-branching experiment

Measures how much the lane choice at a decision point (>= 2 eligible lanes) changes makespan.
Run from the repo root on the Windows machine with Vivado (`xsim.bat`). `<sample>` is a sample folder
(or its `beats_hex/`), `<sim>` the xsim simulation directory.

1. `python -m experiments.branching.check_determinism --data-dir <sample> --sim-dir <sim>`
   Prints `DETERMINISM: PASS` or FAIL (then measures noise over 5 more baseline episodes and stops).
2. `python -m experiments.branching.run_branching --data-dir <sample> --sim-dir <sim> --n-states 50 --workers 4`
   (`--resume` continues after an interruption; `--continuation mwkr` writes to `results/branching_mwkr/`;
   `--states 120 480 ...` forces specific decision steps; `--timeout`, `--seed`). Each worker gets its own
   copy of `<sim>` under `results/branching/workers/`.
3. `python -m experiments.branching.analyze --dir results/branching`

Outputs in `results/branching/`:
- `baseline.json`: teacher makespan, actions, decision steps, masks, per-step cycles, top_in_ready.
- `branches.csv`: one row per branch (`state_id, step, cycle_at_step, n_eligible, eligible_lanes,
  teacher_action, forced_action, makespan, delta_vs_baseline, wall_seconds, error`); makespan -1 = failed.
- `obs.npz`: policy observation at each selected decision step, keys `<state_id>__<obs key>`.
- `hw_states.jsonl`: raw `hw_state` dict per selected state.
- `report.md`, `spread_hist.png`, `spread_by_band.png`: from `analyze.py`.

Notes: the teacher is `policies/round_robin.py` (cursor starts at 0, i.e. `last_served = 7`; a forced action
moves the cursor to `action + 1`, i.e. `last_served = action`). `envs/` is untouched; `XSimClient.close()`
imports `signal` from `torch`, so the harness kills the xsim process tree itself.
