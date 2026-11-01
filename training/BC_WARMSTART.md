# Warm-starting the lane scheduler from the round-robin teacher

The teacher is the v20 testbench packer (`deployment/testbench_but_python.py`). It keeps a cursor
over 9 channels: the 8 lanes plus the disabled bypass slot. Each cycle it sends to the first
eligible lane at or after the cursor. The cursor then moves to the lane after the one it served.
It also moves one step on an idle cycle while the bus is ready, and holds while `top_in_ready` is
low. A lane is eligible when its tail is not busy and it has 8 or fewer CMDs in flight.

The env reproduces this with two opt-in pieces:

- `inflight_gate=TB_INFLIGHT_GATE`, which `--tb-gate` turns on. It adds the 8-in-flight rule to the
  action mask, so the mask allows exactly the lanes the testbench would consider. Without it, the
  teacher would sometimes want a no-op that the mask forbids.
- Observation `v4`, which is v3 plus two per-lane features: `cycles_since_fed` and `rr_distance`
  (scan distance from the cursor, which the env tracks itself). Without these, BC plateaus around
  88% match.

## 0. Check the teacher against the SV testbench

```
python -m policies.round_robin --data-dir <sample dir> --sim-dir <sim dir> --episodes 3 --expected <SV wall>
```

This prints each episode's makespan and `>>> TEACHER MAKESPAN: mean = ...`. With `--expected`, it
exits non-zero if the mean is more than 1% off. **If it is off, stop: the env and the testbench
disagree.** Note that the env's makespan ends when every lane has fully drained. The testbench wall
ends at the 720th row, which can be a few cycles earlier.

## 1. Collect teacher data

```
python training/collect_teacher.py --data-dir <pool or sample dir> --sim-dir <sim dir> --episodes 30
```

This runs the teacher through the same wrapped env as PPO (with the TB gate and obs v4). It writes:

- `data/teacher_<ts>.npz`: observations, masks, actions, episode ids, and return-to-go (raw and
  VecNormalize-scaled).
- `data/teacher_<ts>_vecnormalize.pkl`: the VecNormalize statistics.

Forced no-op steps are skipped. `--seed` sets the order in which samples are picked from the pool.

## 2. Behavior cloning

```
python training/bc_pretrain.py --data data/teacher_<ts>.npz
```

Trains with masked cross-entropy, plus a value-head MSE fit unless you pass `--no-fit-value`.
Whole episodes are held out (`--holdout 0.1`).

- It stops when held-out teacher match reaches 99%, or after `--max-epochs 50`.
- If match plateaus below 95% (no gain for `--patience 10` epochs), it exits non-zero without
  saving.
- It writes `models/bc_<ts>.zip` and the frozen policy `models/bc_<ts>_clone.pt`.

Other options: `--lr 3e-4`, `--batch-size 512`.

## 3. PPO fine-tuning

```
python training/train_xsim.py --data-dir <pool> --sim-dir <sim dir> --obs_version v4 --tb-gate \
    --vecnorm-stats data/teacher_<ts>_vecnormalize.pkl \
    --init-from models/bc_<ts>.zip --kl-prior models/bc_<ts>_clone.pt --kl-beta 0.1 --kl-decay-episodes 500 \
    --lr 3e-5 --ent-coef-schedule 0:0,100:0.001
```

| flag | effect |
|---|---|
| `--tb-gate` | Testbench mask gate. Always pass it with a BC model: the BC model was trained on that mask. |
| `--vecnorm-stats` | Start from the teacher run's VecNormalize statistics, so rewards are scaled as they were when the value head was fit. |
| `--init-from` | Copy the BC policy and value weights into the fresh model. The optimizer starts fresh. |
| `--kl-prior`, `--kl-beta`, `--kl-decay-episodes` | Add `beta * KL(pi \|\| clone)` over the eligible actions to the PPO loss. Beta falls linearly to 0 over that many Monitor episodes. |
| `--lr` | Constant learning rate. Without it, the rate decays linearly from 1e-4. |
| `--ent-coef-schedule` | Entropy coefficient by episode count, as `episode:value,...`. |
| `--eval-every N`, `--eval-episodes 2` | Makespan evaluation; see below. |

Evaluation runs once before training and then every N episodes. It is on by default (N = 25)
whenever `--init-from` or `--kl-prior` is given. Each evaluation does the following:

- runs `--eval-episodes` deterministic episodes on samples 0..n-1 of the pool;
- runs the teacher on the same samples (cached after the first time);
- saves `models/best.zip` (and `best_vecnormalize.pkl`) whenever the mean makespan improves.

Periodic checkpoints are kept as before. Without any of these flags, `train_xsim.py` behaves exactly
as before.

## Logged metrics (TensorBoard)

| metric | meaning |
|---|---|
| `bc/kl_to_clone` | Mean KL(pi_theta \|\| pi_clone) over eligible actions in the last update. It should stay below 0.05 while `beta > 0`. Logged with sb3's `train/*`, so it first appears from the second rollout. |
| `bc/beta` | Current KL weight. |
| `bc/teacher_match_rate` | Share of rollout steps (with any lane eligible) where the policy's greedy action equals what the teacher would pick from the same state. That teacher follows the agent's actual actions. |
| `bc/episodes`, `bc/ent_coef` | Episode count driving the schedules, and the current entropy coefficient. |
| `eval/makespan` | Mean deterministic makespan over the evaluation samples, in cycles. |
| `eval/makespan_vs_teacher` | Mean of policy makespan / teacher makespan, per sample. 1.0 means teacher parity; below 1.0 means the policy beats the teacher. |

The collapse guard: during the first 200 episodes, `eval/makespan_vs_teacher` should stay below
1.05. If it doesn't, report it rather than tuning around it.
