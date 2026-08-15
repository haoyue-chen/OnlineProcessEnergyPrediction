"""Task 5 — Online learning: static MoE vs MoE + online updating.

Phase-1 problem #1 is that the model is trained once and then frozen, so it can
not adapt to workload drift. Here we stream intervals in time order and, crucially,
concatenate the four workloads end-to-end to *induce* drift: the regime changes
each time one workload ends and the next begins.

We compare three strategies under a prequential (test-then-train) protocol — every
interval is first predicted, the error is recorded, and only then is the sample
revealed to the model for updating:

  1. Static-Single   — one linear model, frozen after an initial warm-up fit.
  2. Static-MoE      — gate + per-workload linear experts, frozen after warm-up.
  3. Online-MoE      — same gate, but each expert is a River online regressor
                       updated sample-by-sample as data arrives.

River provides the online regressors (SGD-style linear model and the Hoeffding
adaptive tree, both from the plan's reading list).

Usage:
    python -m moe.run_online                 # text report
    python -m moe.run_online --plot out.png  # also save the error-over-time chart
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data
from .moe import MixtureOfExperts, MoEConfig
import time as _time

WARMUP_FRAC = 0.15  # fraction of each workload used to warm-start static models + gate


def make_online_expert(kind: str):
    """Construct a fresh River online regressor from the registry.

    Options: linear (Adam-optimised; plain SGD diverges on unscaled ~300 Wh
    targets), pa (passive-aggressive), htr (Hoeffding tree), hatr (Hoeffding
    adaptive tree), arf (Adaptive Random Forest — the plan's Task-2 model), knn.
    """
    from .registry import make_online_model

    # Back-compat alias: the old --online-expert "tree" meant the adaptive tree.
    if kind == "tree":
        kind = "hatr"
    return make_online_model(kind)


def build_stream(datasets: dict[str, data.IntervalDataset]):
    """Concatenate workloads in time order to create a drifting stream.

    Returns the full streamed (X, y, w) and a boolean ``is_warm`` mask marking the
    head ``WARMUP_FRAC`` of each workload. Warm rows are used to fit the static
    baselines / gate and to warm-start the online experts; evaluation is reported
    only on the remaining (post-warm-up) rows so all models are scored fairly.
    """
    stream_X, stream_y, stream_w, warm = [], [], [], []
    for label, ds in datasets.items():
        n = len(ds)
        cut = int(n * WARMUP_FRAC)
        stream_X.append(ds.X)
        stream_y.append(ds.y)
        stream_w.append(pd.Series([label] * n))
        warm.append(pd.Series([True] * cut + [False] * (n - cut)))
    return (
        pd.concat(stream_X, ignore_index=True),
        pd.concat(stream_y, ignore_index=True),
        pd.concat(stream_w, ignore_index=True),
        pd.concat(warm, ignore_index=True).values,
    )


def _feat_dict(row: np.ndarray, names: list[str]) -> dict:
    return {n: float(v) for n, v in zip(names, row)}


def run_online_moe(gate, X, y, w, is_warm, feature_names, expert_kind):
    """Prequential loop for the online MoE: route by gate, predict, then update.

    Warm rows are learned (no prediction recorded) to warm-start each expert and
    let River's running StandardScaler accumulate stats; from then on every row is
    predicted before being learned (test-then-train). Returns predictions for all
    rows (warm-row predictions are left as NaN and ignored downstream).
    """
    experts = {label: make_online_expert(expert_kind) for label in gate.labels_}
    preds = np.full(len(X), np.nan, dtype=float)

    routed = gate.route(X)
    Xv, yv = X.values, y.values

    for i in range(len(X)):
        expert = experts[routed[i]]
        xd = _feat_dict(Xv[i], feature_names)
        if is_warm[i]:
            expert.learn_one(xd, float(yv[i]))
            continue
        yhat = expert.predict_one(xd)
        preds[i] = yhat if yhat is not None else 0.0
        expert.learn_one(xd, float(yv[i]))
    return preds

def run_periodic_retrain_moe(gate, X, y, w, is_warm, expert_kind, retrain_every):
    """Prequential loop for a periodic-retrain baseline: refit a batch MoE
    every `retrain_every` post-warm-up samples; frozen in between.

    Returns predictions (NaN on warm rows, matching run_online_moe) plus
    timing stats — this is the number that answers "what does online
    learning actually save you over just retraining periodically".
    """
    from .moe import MixtureOfExperts, MoEConfig

    preds = np.full(len(X), np.nan, dtype=float)
    buf_idx = list(np.where(is_warm)[0])  # buffer starts as the warm-up rows
    model = MixtureOfExperts(MoEConfig(expert=expert_kind)).fit(
        X.iloc[buf_idx], y.iloc[buf_idx], w.iloc[buf_idx]
    )
    n_since_refit, n_refits, total_refit_s = 0, 1, 0.0

    for i in range(len(X)):
        if is_warm[i]:
            continue
        preds[i] = model.predict(X.iloc[[i]])[0]   # predict BEFORE learning (test-then-train)
        buf_idx.append(i)
        n_since_refit += 1
        if n_since_refit >= retrain_every:
            t0 = _time.perf_counter()
            model = MixtureOfExperts(MoEConfig(expert=expert_kind)).fit(
                X.iloc[buf_idx], y.iloc[buf_idx], w.iloc[buf_idx]
            )
            total_refit_s += _time.perf_counter() - t0
            n_since_refit = 0
            n_refits += 1

    return preds, {"n_refits": n_refits, "total_refit_seconds": total_refit_s}


def main():
    parser = argparse.ArgumentParser(description="Static MoE vs Online MoE (Task 5)")
    parser.add_argument("--online-expert",
                        choices=["linear", "pa", "htr", "hatr", "arf", "knn", "tree"],
                        default="arf",
                        help="River online expert. 'arf' (Adaptive Random Forest) is "
                             "strongest; 'tree' is a back-compat alias for 'hatr'.")
    parser.add_argument("--retrain-every", type=int, default=500,
                        help="Periodic-retrain baseline: refit every N post-warm-up intervals.")
    parser.add_argument("--periodic-expert", default="rf",
                        help="Batch model class for the periodic-retrain baseline "
                             "(use 'rf' to fairly compare against Online-MoE's ARF).")
    parser.add_argument("--plot", default=None, help="Optional path to save error-over-time chart")
    args = parser.parse_args()

    print("Loading workloads …")
    datasets = data.load_all()
    order = list(datasets)
    X, y, w, is_warm = build_stream(datasets)
    feats = list(X.columns)
    n_warm = int(is_warm.sum())
    print(f"Stream length: {len(X):,} intervals  |  order: {' -> '.join(order)}")
    print(f"Warm-up rows: {n_warm:,}  ({WARMUP_FRAC:.0%} head of each workload); evaluated on the rest")

    # Static models train on the warm-up rows only, then freeze.
    wX, wy, ww = X[is_warm], y[is_warm], w[is_warm]

    # --- Static baselines (frozen after warm-up) -------------------------------
    print("\nFitting static baselines on warm-up slice …")
    single = make_pipeline(StandardScaler(), LinearRegression()).fit(wX.values, wy.values)
    static_moe = MixtureOfExperts(MoEConfig(expert="linear")).fit(wX, wy, ww)

    pred_single = single.predict(X.values)
    pred_static_moe = static_moe.predict(X)

    # --- Online MoE (gate frozen, experts update) ------------------------------
    print(f"Running online MoE (expert={args.online_expert}) …")
    pred_online = run_online_moe(static_moe, X, y, w, is_warm, feats, args.online_expert)

    # --- Periodic-retrain baseline (gate + experts both refit every N samples) -
    print(f"Running periodic-retrain baseline (expert={args.periodic_expert}, every={args.retrain_every}) …")
    pred_periodic, stats = run_periodic_retrain_moe(
        static_moe, X, y, w, is_warm, args.periodic_expert, args.retrain_every
    )
    print(f"  {stats['n_refits']} refits, {stats['total_refit_seconds']:.1f}s total refit time "
          f"({stats['total_refit_seconds']/stats['n_refits']:.2f}s/refit)")

    # Evaluate everyone on the post-warm-up rows only (online/periodic preds are NaN on warm rows).
    eval_mask = ~is_warm
    results = {
        "Static-Single": pred_single,
        "Static-MoE": pred_static_moe,
        "Periodic-Retrain": pred_periodic,
        "Online-MoE": pred_online,
    }
    _report(y.values, w.values, eval_mask, order, results)
    if args.plot:
        _plot(y.values, w.values, eval_mask, order, results, args.plot)


def _report(actual, wv, eval_mask, order, results):
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("\n── Task 5: Static vs Online (prequential, post-warm-up rows) ──")
    rows = []
    for name, pred in results.items():
        row = {"model": name}
        for label in order:
            m = eval_mask & (wv == label)
            row[label] = r2_score(actual[m], pred[m])
        row["OVERALL_r2"] = r2_score(actual[eval_mask], pred[eval_mask])
        row["OVERALL_mae"] = mean_absolute_error(actual[eval_mask], pred[eval_mask])
        rows.append(row)
    print(pd.DataFrame(rows).to_string(index=False))
    print("──────────────────────────────────────────────────────────────")


def _plot(actual, wv, eval_mask, order, results, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Rolling absolute error over the stream, to visualise adaptation after drift.
    # Warm-up rows are excluded (NaN) so the curves reflect evaluated intervals only.
    win = 200
    fig, ax = plt.subplots(figsize=(11, 4))
    for name, pred in results.items():
        err = np.abs(actual - pred)
        err = np.where(eval_mask, err, np.nan)
        roll = pd.Series(err).rolling(win, min_periods=1).mean()
        ax.plot(roll, label=name, linewidth=1.2)
    # Mark workload boundaries.
    idx = 0
    for label in order:
        idx += int((wv == label).sum())
        ax.axvline(idx, color="gray", linestyle=":", alpha=0.5)
    ax.set_xlabel("Interval (streamed in time order; dotted = workload boundary)")
    ax.set_ylabel(f"Rolling MAE (window={win})")
    ax.set_title("Adaptation under workload drift: static vs online MoE")
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
