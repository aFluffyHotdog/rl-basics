"""Summarise branches.csv -> report.md + two PNG plots (CSV only; no xsim, no other inputs except baseline.json).

  python -m experiments.branching.analyze --dir results/branching
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

N_BANDS = 5
THRESHOLDS = (1, 5, 20)


def load_rows(csv_path):
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            rows.append(dict(state_id=r["state_id"], step=int(r["step"]), n_eligible=int(r["n_eligible"]),
                             teacher_action=int(r["teacher_action"]), forced=int(r["forced_action"]),
                             makespan=int(r["makespan"]), error=r.get("error", "") or ""))
    return rows


def per_state(rows):
    """-> (states, n_failed_branches). Failed branch = error set or makespan < 0; it is excluded.

    A state needs >= 2 valid branches for a spread, and a valid teacher branch for a regret."""
    by_state = defaultdict(list)
    failed = 0
    for r in rows:
        if r["error"] or r["makespan"] < 0:
            failed += 1
            continue
        by_state[r["state_id"]].append(r)
    states = []
    for sid, rs in by_state.items():
        spans = [r["makespan"] for r in rs]
        t = [r["makespan"] for r in rs if r["forced"] == r["teacher_action"]]
        states.append(dict(state_id=sid, step=rs[0]["step"], n_eligible=rs[0]["n_eligible"],
                           spread=max(spans) - min(spans) if len(spans) >= 2 else None,
                           regret=t[0] - min(spans) if t and len(spans) >= 2 else None,
                           best=min(spans), teacher=t[0] if t else None))
    return sorted(states, key=lambda s: s["step"]), failed


def dist(vals):
    v = np.asarray(vals, dtype=float)
    if len(v) == 0:
        return dict(mean=np.nan, median=np.nan, p90=np.nan, max=np.nan)
    return dict(mean=v.mean(), median=np.median(v), p90=np.percentile(v, 90), max=v.max())


def fractions(vals):
    v = np.asarray(vals, dtype=float)
    out = {"= 0": float((v == 0).mean()) if len(v) else np.nan}
    for t in THRESHOLDS:
        out[f">= {t}"] = float((v >= t).mean()) if len(v) else np.nan
    return out


def bands(states):
    """Selected states sorted by step, split into N_BANDS equal-count groups (matches the sampling bands)."""
    return [list(b) for b in np.array_split(np.array(states, dtype=object), N_BANDS) if len(b)]


def verdict(spreads):
    """Which of the three cases the data supports (thresholds: spread 0 / <=5 cycles / minority-vs-frequent)."""
    if not spreads:
        return "No valid states: no case can be supported."
    f1, f5, f20 = (float(np.mean(np.asarray(spreads) >= t)) for t in (1, 5, 20))
    med, mx = float(np.median(spreads)), float(max(spreads))
    if f1 < 0.1 or mx < 2:
        return (f"Case 1 (little leverage): spread is 0 at {100*(1-f1):.0f}% of states, median {med:g}, max {mx:g} cycles.")
    if f20 >= 0.25 or med >= 20:
        return (f"Case 3 (real headroom): spread >= 20 cycles at {100*f20:.0f}% of states, median {med:g}, max {mx:g}.")
    return (f"Case 2 (learnable but small): spread >= 1 cycle at {100*f1:.0f}% of states, >= 5 at {100*f5:.0f}%, "
            f">= 20 at {100*f20:.0f}%, median {med:g}, max {mx:g} cycles.")


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def dist_lines(name, vals):
    d, f = dist(vals), fractions(vals)
    return (f"- {name}: mean {d['mean']:.2f}, median {d['median']:.1f}, p90 {d['p90']:.1f}, max {d['max']:.0f}\n"
            f"- {name} fraction: " + ", ".join(f"{k}: {100*v:.0f}%" for k, v in f.items()))


def plots(states, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    spreads = [s["spread"] for s in states if s["spread"] is not None]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(spreads, bins=range(0, int(max(spreads, default=0)) + 3), color="#4477aa")
    ax.set(xlabel="spread (max - min makespan, cycles)", ylabel="states", title="Spread per decision state")
    fig.tight_layout(); fig.savefig(out_dir / "spread_hist.png", dpi=120); plt.close(fig)

    bs = bands([s for s in states if s["spread"] is not None])
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.boxplot([[s["spread"] for s in b] for b in bs], tick_labels=[f"{i+1}" for i in range(len(bs))])
    ax.set(xlabel="episode band (1 = earliest)", ylabel="spread (cycles)", title="Spread by episode band")
    fig.tight_layout(); fig.savefig(out_dir / "spread_by_band.png", dpi=120); plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/branching")
    args = ap.parse_args(argv)
    d = Path(args.dir)
    rows = load_rows(d / "branches.csv")
    states, failed = per_state(rows)
    spreads = [s["spread"] for s in states if s["spread"] is not None]
    regrets = [s["regret"] for s in states if s["regret"] is not None]
    base = json.loads((d / "baseline.json").read_text())["makespan"] if (d / "baseline.json").exists() else "n/a"

    L = ["# Decision-branching report", "",
         f"- Baseline (round-robin) makespan: {base}",
         f"- States: {len({r['state_id'] for r in rows})} ({len(spreads)} with a valid spread)",
         f"- Branches: {len(rows)}; failed (error or makespan -1): {failed}", "",
         "## Spread (max - min makespan over a state's branches, cycles)", dist_lines("spread", spreads), "",
         "## Teacher regret (makespan of teacher's action - best branch, cycles)", dist_lines("regret", regrets),
         f"- States where some action beats the teacher: {sum(r > 0 for r in regrets)} / {len(regrets)}",
         f"- Best single-state improvement: {max(regrets, default=0)} cycles", "",
         "## Spread vs episode position (bands of selected states, equal count, by step)"]
    trs = []
    for i, b in enumerate(bands([s for s in states if s["spread"] is not None])):
        sp = [s["spread"] for s in b]
        trs.append([i + 1, f"{b[0]['step']}-{b[-1]['step']}", len(b), f"{np.mean(sp):.2f}", f"{np.median(sp):g}",
                    max(sp), f"{100*np.mean(np.asarray(sp) > 0):.0f}%"])
    L += [table(["band", "steps", "states", "mean", "median", "max", "% spread>0"], trs), "",
          "## Spread vs n_eligible"]
    ne = defaultdict(list)
    for s in states:
        if s["spread"] is not None:
            ne[s["n_eligible"]].append(s["spread"])
    L += [table(["n_eligible", "states", "mean", "median", "max", "% spread>0"],
                [[k, len(v), f"{np.mean(v):.2f}", f"{np.median(v):g}", max(v), f"{100*np.mean(np.asarray(v) > 0):.0f}%"]
                 for k, v in sorted(ne.items())]), "",
          "## Verdict", verdict(spreads), "",
          "Plots: spread_hist.png, spread_by_band.png"]
    (d / "report.md").write_text("\n".join(L) + "\n")
    if spreads:
        plots(states, d)
    print(f"wrote {d / 'report.md'}")


if __name__ == "__main__":
    main()
