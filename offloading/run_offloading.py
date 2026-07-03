"""Objective 3 — Energy-aware offloading driven by the MoE predictor.

Pipeline:
    workloads → segment into a job workflow → MoE predicts each job's energy
    → energy-first greedy offload onto a greener, capacity-limited secondary
    cluster → score every strategy on the *actual* measured energy/cost/makespan.

The point: the MoE is the decision driver. We compare offloading guided by the
MoE against the single global model, an oracle (true energy — the achievable
ceiling), a random baseline, and all-primary (no offloading). A better energy
predictor should save more real energy under the same capacity budget.

Usage:
    python -m offloading.run_offloading
    python -m offloading.run_offloading --expert linear --capacity-frac 0.3
    python -m offloading.run_offloading --plot results/offloading/energy_saved.png
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from moe import data
from moe.moe import MixtureOfExperts, MoEConfig, SingleModel

from .clusters import default_clusters
from .decision import decide
from .workflow import build_workflow, predict_job_energy

STRATEGIES = ["all_primary", "random", "single", "moe", "oracle"]


def main():
    parser = argparse.ArgumentParser(description="MoE-driven energy-aware offloading (Objective 3)")
    parser.add_argument("--expert", choices=["rf", "linear"], default="linear",
                        help="Expert/model class for the MoE and single model. "
                             "Default 'linear': RF predicts per-job energy near-perfectly, "
                             "so all informed strategies converge on the oracle and the "
                             "MoE-vs-single difference only shows under the weaker linear class.")
    parser.add_argument("--jobs-per-workload", type=int, default=25,
                        help="How many jobs to segment each workload into.")
    parser.add_argument("--capacity-frac", type=float, default=0.15,
                        help="Secondary capacity as a fraction of all-primary makespan. "
                             "Tighter budgets make the ranking (prediction quality) matter more.")
    parser.add_argument("--deadline-frac", type=float, default=1.00,
                        help="Makespan deadline as a fraction of all-primary makespan.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", default=None, help="Optional path to save a bar chart.")
    parser.add_argument("--sweep", default=None,
                        help="Optional path to save an energy-saved-vs-capacity line chart.")
    args = parser.parse_args()

    print("Loading workloads …")
    datasets = data.load_all()

    # Train predictors on the full interval data (held-out quality is established
    # in Task 4; here we use them as the deployed predictors).
    X, y, w = data.combined_frame(datasets)
    cfg = MoEConfig(expert=args.expert)
    print(f"Training MoE + single model (expert={args.expert}) …")
    moe = MixtureOfExperts(cfg).fit(X, y, w)
    single = SingleModel(cfg).fit(X, y)

    print(f"Building workflow ({args.jobs_per_workload} jobs/workload) …")
    jobs = build_workflow(datasets, jobs_per_workload=args.jobs_per_workload, seed=args.seed)
    # Predict each job's energy as the sum of its interval-level predictions
    # (predictors are trained on interval rows).
    pred_moe = predict_job_energy(jobs, moe.predict)
    pred_single = predict_job_energy(jobs, single.predict)
    true_energy = np.array([j.true_energy_wh for j in jobs])

    # Per-job prediction quality (informational): how well each predictor ranks jobs.
    from sklearn.metrics import r2_score
    print(f"Workflow: {len(jobs)} jobs  |  total measured energy "
          f"{true_energy.sum()/1000:.2f} kWh")
    print(f"Per-job energy R²:  MoE={r2_score(true_energy, pred_moe):.3f}  "
          f"Single={r2_score(true_energy, pred_single):.3f}")

    primary, secondary = default_clusters()
    print(f"Clusters: secondary energy×{secondary.energy_factor}, "
          f"speed×{secondary.speed_factor}, ${secondary.cost_per_kwh}/kWh  "
          f"| capacity={args.capacity_frac:.0%} of makespan\n")

    pred_for = {"moe": pred_moe, "single": pred_single}
    rows = []
    for strat in STRATEGIES:
        pe = pred_for.get(strat, np.zeros(len(jobs)))
        out = decide(jobs, pe, primary, secondary, strat,
                     args.capacity_frac, args.deadline_frac, seed=args.seed)
        rows.append(out)

    _report(rows)
    if args.plot:
        _plot(rows, args.plot, args.expert)
    if args.sweep:
        _sweep(jobs, pred_for, primary, secondary, args)


def _sweep(jobs, pred_for, primary, secondary, args):
    """Energy saved vs secondary capacity, per strategy — the most telling view."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    caps = np.linspace(0.05, 0.6, 12)
    series = {s: [] for s in STRATEGIES if s != "all_primary"}
    for cap in caps:
        for s in series:
            pe = pred_for.get(s, np.zeros(len(jobs)))
            o = decide(jobs, pe, primary, secondary, s, float(cap),
                       args.deadline_frac, seed=args.seed)
            series[s].append(o.energy_saved_pct)
    style = {"random": ("#9aa7b5", "-"), "single": ("#f0a202", "-"),
             "moe": ("#3477eb", "-"), "oracle": ("#2fb380", "--")}
    fig, ax = plt.subplots(figsize=(8, 4.5))
    for s, vals in series.items():
        c, ls = style.get(s, ("#777", "-"))
        ax.plot(caps * 100, vals, ls, color=c, marker="o", ms=3, label=s)
    ax.set_xlabel("Secondary capacity (% of all-primary makespan)")
    ax.set_ylabel("Energy saved vs all-primary (%)")
    ax.set_title(f"Energy saved vs capacity — MoE tracks the oracle (expert={args.expert})")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.savefig(args.sweep, dpi=150)
    print(f"Saved sweep plot: {args.sweep}")


def _report(rows):
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    df = pd.DataFrame([{
        "strategy": o.strategy,
        "offloaded": o.n_offloaded,
        "energy_kWh": o.total_energy_wh / 1000.0,
        "saved_kWh": o.energy_saved_wh / 1000.0,
        "saved_%": o.energy_saved_pct,
        "cost": o.total_cost,
        "makespan_s": o.makespan_s,
        "deadline": "ok" if o.deadline_met else "MISS",
    } for o in rows])
    print("── Objective 3: Energy-aware offloading ──────────────────────")
    print(df.to_string(index=False))
    print("──────────────────────────────────────────────────────────────")
    moe = next(o for o in rows if o.strategy == "moe")
    single = next(o for o in rows if o.strategy == "single")
    oracle = next(o for o in rows if o.strategy == "oracle")
    gap = (oracle.energy_saved_pct - single.energy_saved_pct)
    captured = (moe.energy_saved_pct - single.energy_saved_pct) / gap * 100 if gap > 1e-9 else float("nan")
    print(f"MoE saved {moe.energy_saved_pct:.1f}% vs single {single.energy_saved_pct:.1f}% "
          f"(oracle ceiling {oracle.energy_saved_pct:.1f}%).")
    if gap > 1e-9:
        print(f"MoE closes {captured:.0f}% of the single→oracle decision gap.")


def _plot(rows, path, expert):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [o.strategy for o in rows]
    saved = [o.energy_saved_pct for o in rows]
    colors = {"all_primary": "#cbd2d9", "random": "#9aa7b5", "single": "#f0a202",
              "moe": "#3477eb", "oracle": "#2fb380"}
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, saved, color=[colors.get(n, "#777") for n in names])
    ax.set_ylabel("Energy saved vs all-primary (%)")
    ax.set_title(f"Energy-aware offloading — decision strategy comparison (expert={expert})")
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for i, v in enumerate(saved):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
