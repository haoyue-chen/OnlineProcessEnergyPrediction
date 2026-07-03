"""Baseline comparison across all registered models (batch + online).

Two tables:
  * batch  — each sklearn model as a single global regressor, 5-fold CV R²/MAE.
             Also reports the MoE-vs-single gain for that model class.
  * online — each river model run prequentially over the drifting workload stream.

This is the "baseline enhancement" view: instead of just linear-vs-RF and
SGD-only, we compare whole model *families* (bagged trees, boosted trees, kNN,
MLP, SVR; online linear/PA/Hoeffding/adaptive-Hoeffding/ARF/kNN), so the choice
of expert is grounded in evidence rather than assumption.

Usage:
    python -m moe.compare_models                 # both tables
    python -m moe.compare_models --batch-only
    python -m moe.compare_models --online-only
    python -m moe.compare_models --plot results/moe/model_comparison.png
"""

from __future__ import annotations

import argparse
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from . import data
from .moe import MixtureOfExperts, MoEConfig, SingleModel
from .registry import BATCH_MODELS, ONLINE_MODEL_NAMES, make_online_model

warnings.filterwarnings("ignore")


def compare_batch(datasets, cv: int, max_rows: int | None = 8000) -> pd.DataFrame:
    X, y, w = data.combined_frame(datasets)
    # Kernel/NN models (SVR, MLP) are O(n^2)-ish; subsample for a tractable,
    # apples-to-apples survey. Family ordering is stable under subsampling.
    if max_rows and len(X) > max_rows:
        idx = np.random.RandomState(0).choice(len(X), max_rows, replace=False)
        idx.sort()
        X, y, w = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True), w.iloc[idx].reset_index(drop=True)
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    rows = []
    for name in BATCH_MODELS:
        cfg = MoEConfig(expert=name, n_estimators=150)
        # Manual CV loop yields both the single-global and MoE out-of-fold preds.
        t = time.time()
        single_pred = np.empty(len(y))
        moe_pred = np.empty(len(y))
        for tr, te in kf.split(X):
            Xtr, Xte = X.iloc[tr], X.iloc[te]
            ytr, wtr = y.iloc[tr], w.iloc[tr]
            single_pred[te] = SingleModel(cfg).fit(Xtr, ytr).predict(Xte)
            moe_pred[te] = MixtureOfExperts(cfg).fit(Xtr, ytr, wtr).predict(Xte)
        dt = time.time() - t
        rows.append({
            "model": name,
            "single_r2": r2_score(y, single_pred),
            "moe_r2": r2_score(y, moe_pred),
            "r2_gain": r2_score(y, moe_pred) - r2_score(y, single_pred),
            "single_mae": mean_absolute_error(y, single_pred),
            "moe_mae": mean_absolute_error(y, moe_pred),
            "time_s": dt,
        })
        print(f"  batch {name:12s} single R²={rows[-1]['single_r2']:.3f}  "
              f"moe R²={rows[-1]['moe_r2']:.3f}  ({dt:.1f}s)")
    return pd.DataFrame(rows)


def compare_online(datasets, warmup: int = 500) -> pd.DataFrame:
    X, y, w = data.combined_frame(datasets)
    feats = list(X.columns)
    Xv, yv = X.values, y.values
    n = len(Xv)
    rows = []
    for name in ONLINE_MODEL_NAMES:
        model = make_online_model(name)
        preds = np.empty(n)
        t = time.time()
        for i in range(n):
            xd = {f: float(v) for f, v in zip(feats, Xv[i])}
            p = model.predict_one(xd)
            preds[i] = p if p is not None else 0.0
            model.learn_one(xd, float(yv[i]))
        dt = time.time() - t
        rows.append({
            "model": name,
            "r2": r2_score(yv[warmup:], preds[warmup:]),
            "mae": mean_absolute_error(yv[warmup:], preds[warmup:]),
            "time_s": dt,
        })
        print(f"  online {name:8s} R²={rows[-1]['r2']:.3f}  "
              f"MAE={rows[-1]['mae']:.2f}  ({dt:.1f}s)")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Compare all batch + online baselines")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--batch-only", action="store_true")
    ap.add_argument("--online-only", action="store_true")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print("Loading workloads …")
    datasets = data.load_all()

    batch_df = online_df = None
    if not args.online_only:
        print("\nBatch models (5-fold CV; single global vs MoE):")
        batch_df = compare_batch(datasets, args.cv)
        batch_df.to_csv("results/moe/batch_comparison.csv", index=False)
        print("\n── Batch baseline comparison ─────────────────────────────────")
        print(batch_df.sort_values("single_r2", ascending=False).to_string(index=False))
        print("───────────────────────────────────────────────────────────────")

    if not args.batch_only:
        print("\nOnline models (prequential over drifting stream):")
        online_df = compare_online(datasets)
        online_df.to_csv("results/moe/online_comparison.csv", index=False)
        print("\n── Online baseline comparison ────────────────────────────────")
        print(online_df.sort_values("r2", ascending=False).to_string(index=False))
        print("───────────────────────────────────────────────────────────────")

    if args.plot and (batch_df is not None or online_df is not None):
        _plot(batch_df, online_df, args.plot)


def _plot(batch_df, online_df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n = sum(x is not None for x in (batch_df, online_df))
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4.2))
    if n == 1:
        axes = [axes]
    ax_i = 0
    if batch_df is not None:
        df = batch_df.sort_values("single_r2")
        ax = axes[ax_i]; ax_i += 1
        y = np.arange(len(df))
        ax.barh(y - 0.2, df["single_r2"], 0.4, label="single", color="#9aa7b5")
        ax.barh(y + 0.2, df["moe_r2"], 0.4, label="MoE", color="#3477eb")
        ax.set_yticks(y); ax.set_yticklabels(df["model"])
        ax.set_xlabel("R² (5-fold CV)"); ax.set_xlim(0, 1)
        ax.set_title("Batch baselines: single vs MoE"); ax.legend()
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    if online_df is not None:
        df = online_df.sort_values("r2")
        ax = axes[ax_i]
        ax.barh(df["model"], df["r2"], color="#2fb380")
        ax.set_xlabel("R² (prequential)"); ax.set_xlim(0, 1)
        ax.set_title("Online baselines (drifting stream)")
        ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
