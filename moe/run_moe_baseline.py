"""Task 4 — MoE baseline experiment: Mixture-of-Experts vs a single global model.

Evaluation modes
----------------
* Default — **K-fold cross-validation** (``--cv K``, default 5). Out-of-fold
  predictions are collected across all folds, then scored per workload and
  overall. This is the robust methodology that matches the mid-term report's
  baseline numbers: a single random hold-out is sensitive to which "hard" rows
  land in the test split (Phoronix in particular), whereas CV averages that out.
* ``--time-split`` — chronological hold-out (train on the first ``1 - test_size``
  of each workload, test on the tail). Use this when you specifically want the
  online-prediction framing rather than a like-for-like comparison with the
  report; it is the framing Task 5 (online learning) builds on.

Usage:
    python -m moe.run_moe_baseline                       # 5-fold CV, RF experts
    python -m moe.run_moe_baseline --expert linear       # show MoE's benefit
    python -m moe.run_moe_baseline --time-split          # chronological hold-out
    python -m moe.run_moe_baseline --plot out.png        # also save a bar chart
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from . import data
from .moe import MixtureOfExperts, MoEConfig, SingleModel

TEST_SIZE = 0.2
DEFAULT_CV = 5


def _metrics_frame(y_true, y_pred, w_labels, order, gate_acc=None):
    """Per-workload + overall R²/MAE from full-length prediction vectors."""
    rows = []
    for label in order:
        m = w_labels == label
        rows.append(
            {
                "workload": label,
                "n": int(m.sum()),
                "r2": r2_score(y_true[m], y_pred[m]),
                "mae": mean_absolute_error(y_true[m], y_pred[m]),
                "gate_acc": gate_acc.get(label, np.nan) if gate_acc else np.nan,
            }
        )
    rows.append(
        {
            "workload": "OVERALL",
            "n": len(y_true),
            "r2": r2_score(y_true, y_pred),
            "mae": mean_absolute_error(y_true, y_pred),
            "gate_acc": np.nan,
        }
    )
    return pd.DataFrame(rows)


def run_cv(datasets, cfg, n_splits):
    """K-fold CV over the combined data; collect out-of-fold predictions.

    The same fold indices are used for the MoE and the single model so the
    comparison is paired. The gate is trained on each fold's training part and
    its routing accuracy is averaged over folds.
    """
    X, y, w = data.combined_frame(datasets)
    order = list(datasets)
    Xv, yv, wv = X.values, y.values, w.values

    oof_moe = np.empty(len(y), dtype=float)
    oof_single = np.empty(len(y), dtype=float)
    gate_hits = {label: 0 for label in order}
    gate_total = {label: 0 for label in order}

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_state)
    for fold, (tr, te) in enumerate(kf.split(Xv), 1):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr = y.iloc[tr]
        wtr, wte = w.iloc[tr], w.iloc[te]

        moe = MixtureOfExperts(cfg).fit(Xtr, ytr, wtr)
        oof_moe[te] = moe.predict(Xte)

        single = SingleModel(cfg).fit(Xtr, ytr)
        oof_single[te] = single.predict(Xte)

        routed = moe.route(Xte)
        for label in order:
            m = wte.values == label
            gate_hits[label] += int((routed[m] == label).sum())
            gate_total[label] += int(m.sum())
        print(f"  fold {fold}/{n_splits} done")

    gate_acc = {l: gate_hits[l] / gate_total[l] for l in order if gate_total[l]}
    moe_metrics = _metrics_frame(yv, oof_moe, wv, order, gate_acc)
    single_metrics = _metrics_frame(yv, oof_single, wv, order)
    return moe_metrics, single_metrics


def run_time_split(datasets, cfg, test_size):
    """Chronological hold-out: train on the head of each workload, test on tail."""
    order = list(datasets)
    tr_X, tr_y, tr_w = [], [], []
    te_X, te_y, te_w = [], [], []
    for label, ds in datasets.items():
        n = len(ds)
        cut = int(n * (1 - test_size))
        tr_X.append(ds.X.iloc[:cut]); tr_y.append(ds.y.iloc[:cut])
        tr_w.append(pd.Series([label] * cut))
        te_X.append(ds.X.iloc[cut:]); te_y.append(ds.y.iloc[cut:])
        te_w.append(pd.Series([label] * (n - cut)))
    X_train = pd.concat(tr_X, ignore_index=True)
    y_train = pd.concat(tr_y, ignore_index=True)
    w_train = pd.concat(tr_w, ignore_index=True)
    X_test = pd.concat(te_X, ignore_index=True)
    y_test = pd.concat(te_y, ignore_index=True).values
    w_test = pd.concat(te_w, ignore_index=True).values

    moe = MixtureOfExperts(cfg).fit(X_train, y_train, w_train)
    single = SingleModel(cfg).fit(X_train, y_train)

    routed = moe.route(X_test)
    gate_acc = {}
    for label in order:
        m = w_test == label
        gate_acc[label] = float((routed[m] == label).mean()) if m.any() else np.nan

    moe_metrics = _metrics_frame(y_test, moe.predict(X_test), w_test, order, gate_acc)
    single_metrics = _metrics_frame(y_test, single.predict(X_test), w_test, order)
    return moe_metrics, single_metrics


def main():
    parser = argparse.ArgumentParser(description="MoE vs Single Model (Task 4)")
    parser.add_argument("--expert", default="rf",
                        help="Expert/model class — any registry model "
                             "(rf, linear, extra_trees, hgb, knn, mlp, svr). "
                             "'rf' = tree ceiling; 'linear' = paper's class "
                             "(MoE benefit is clearest).")
    parser.add_argument("--cv", type=int, default=DEFAULT_CV,
                        help="K-fold cross-validation folds (default 5). Ignored with --time-split.")
    parser.add_argument("--time-split", action="store_true",
                        help="Use chronological hold-out instead of CV.")
    parser.add_argument("--test-size", type=float, default=TEST_SIZE,
                        help="Tail fraction held out per workload when --time-split is set.")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--plot", default=None, help="Optional path to save a comparison bar chart")
    args = parser.parse_args()

    print("Loading workloads …")
    datasets = data.load_all()
    cfg = MoEConfig(n_estimators=args.n_estimators, expert=args.expert)

    if args.time_split:
        mode = f"chronological hold-out (test_size={args.test_size})"
        print(f"Eval mode: {mode}  |  expert: {args.expert}")
        moe_metrics, single_metrics = run_time_split(datasets, cfg, args.test_size)
    else:
        mode = f"{args.cv}-fold cross-validation"
        print(f"Eval mode: {mode}  |  expert: {args.expert}")
        moe_metrics, single_metrics = run_cv(datasets, cfg, args.cv)

    report = moe_metrics.merge(single_metrics, on=["workload", "n"], suffixes=("_moe", "_single"))
    report["r2_gain"] = report["r2_moe"] - report["r2_single"]

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(f"\n── Task 4: MoE vs Single Model  [{mode}] ──────────────")
    cols = ["workload", "n", "r2_single", "r2_moe", "r2_gain", "mae_single", "mae_moe", "gate_acc_moe"]
    print(report[cols].to_string(index=False))
    print("───────────────────────────────────────────────────────────────")

    if args.plot:
        _plot(report, args.plot, mode)


def _plot(report: pd.DataFrame, path: str, mode: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = report[report["workload"] != "OVERALL"]
    x = np.arange(len(df))
    width = 0.38
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width / 2, df["r2_single"], width, label="Single model", color="#9aa7b5")
    ax.bar(x + width / 2, df["r2_moe"], width, label="MoE", color="#3477eb")
    ax.set_xticks(x)
    ax.set_xticklabels(df["workload"])
    ax.set_ylabel("R² (interval-level)")
    ax.set_title(f"MoE vs Single Model — per workload\n({mode})")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
