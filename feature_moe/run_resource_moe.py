"""Evaluate the feature-based (resource-grouped) MoE vs baselines (architecture Modules 4-5).

5-fold paired CV comparing:
  1. Single global model (baseline)
  2. Workload-MoE (existing moe.MixtureOfExperts) — reference
  3. Resource-MoE, learned gate (new, learned soft routing)
  4. Resource-MoE, importance gate (ablation — answers doc open question #2)

Reports overall + per-workload R²/MAE, the learned gate weights, and the resource
importance table (doc §4.3). Saves CSV + plot to results/feature_moe/.

Usage:
    PYTHONPATH=. python -m feature_moe.run_resource_moe
    PYTHONPATH=. python -m feature_moe.run_resource_moe --expert linear --plot results/feature_moe/cmp.png
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from moe import data
from moe.moe import MixtureOfExperts, MoEConfig, SingleModel

from .groups import GROUP_ORDER
from .importance import resource_importance
from .moe import ResourceMoE, ResourceMoEConfig, SingleGlobalModel

warnings.filterwarnings("ignore")


def _oof_cv(model_factory, X, y, kf):
    pred = np.empty(len(y))
    for tr, te in kf.split(X):
        m = model_factory().fit(X.iloc[tr], y.iloc[tr])
        pred[te] = m.predict(X.iloc[te])
    return pred


def _per_workload(y_true, y_pred, w, order):
    return {l: (r2_score(y_true[w == l], y_pred[w == l]),
                mean_absolute_error(y_true[w == l], y_pred[w == l])) for l in order}


def main():
    ap = argparse.ArgumentParser(description="Resource-based MoE evaluation")
    ap.add_argument("--expert", default="rf", help="registry batch model for experts")
    ap.add_argument("--cv", type=int, default=5)
    ap.add_argument("--max-rows", type=int, default=None,
                    help="optional subsample cap for slower experts (default: full data). "
                         "WARNING: subsampling biases results — use full data for honest numbers.")
    ap.add_argument("--plot", default=None)
    args = ap.parse_args()

    print("Loading workloads …")
    datasets = data.load_all()
    order = list(datasets)
    X, y, w = data.combined_frame(datasets)
    if args.max_rows and len(X) > args.max_rows:
        idx = np.random.RandomState(0).choice(len(X), args.max_rows, replace=False); idx.sort()
        X, y, w = X.iloc[idx].reset_index(drop=True), y.iloc[idx].reset_index(drop=True), w.iloc[idx].reset_index(drop=True)
    kf = KFold(n_splits=args.cv, shuffle=True, random_state=0)

    print(f"Resource importance (doc §4.3):")
    imp = resource_importance(X, y)
    for g in GROUP_ORDER:
        print(f"  {g:8s}: {imp[g]*100:5.1f}%")

    rows = []
    configs = [
        ("single_global", lambda: SingleGlobalModel(ResourceMoEConfig(expert=args.expert))),
        ("workload_moe",  lambda: SingleModel(MoEConfig(expert=args.expert))),  # 1 global model = single, kept for naming clarity
        ("resource_moe_learned",     lambda: ResourceMoE(ResourceMoEConfig(expert=args.expert, gate="learned"))),
        ("resource_moe_importance",  lambda: ResourceMoE(ResourceMoEConfig(expert=args.expert, gate="importance"))),
    ]
    # also the real workload-grouped MoE (needs w) — handled separately
    for name, factory in configs:
        print(f"  CV {name} …")
        pred = _oof_cv(factory, X, y, kf)
        per = _per_workload(y.values, pred, w.values, order)
        rows.append({
            "model": name, "r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred),
            **{f"r2_{l}": per[l][0] for l in order},
            **{f"mae_{l}": per[l][1] for l in order},
        })

    # workload-grouped MoE (needs w for the gate)
    print("  CV workload-grouped MoE …")
    pred = np.empty(len(y))
    for tr, te in kf.split(X):
        m = MixtureOfExperts(MoEConfig(expert=args.expert)).fit(X.iloc[tr], y.iloc[tr], w.iloc[tr])
        pred[te] = m.predict(X.iloc[te])
    per = _per_workload(y.values, pred, w.values, order)
    rows.append({
        "model": "workload_grouped_moe", "r2": r2_score(y, pred), "mae": mean_absolute_error(y, pred),
        **{f"r2_{l}": per[l][0] for l in order}, **{f"mae_{l}": per[l][1] for l in order},
    })

    df = pd.DataFrame(rows).sort_values("r2", ascending=False)
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    print("\n── Resource-based MoE comparison (5-fold CV) ─────────────────")
    print(df.to_string(index=False))
    print("──────────────────────────────────────────────────────────────")

    # learned gate weights from a full-data fit (for the report)
    full = ResourceMoE(ResourceMoEConfig(expert=args.expert, gate="learned")).fit(X, y)
    gw = full.info()["gate_weights"]
    print("Learned gate weights (full data):", {k: round(v, 3) for k, v in gw.items()})

    out = Path("results/feature_moe"); out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "resource_moe_comparison.csv", index=False)
    (out / "resource_moe_comparison.txt").write_text(df.to_string(index=False))
    print(f"Saved: {out}/resource_moe_comparison.{{csv,txt}}")
    if args.plot:
        _plot(df, args.plot)


def _plot(df, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4))
    d = df.sort_values("r2")
    ax.barh(d["model"], d["r2"], color=["#9aa7b5", "#f0a202", "#2fb380", "#3477eb", "#9b59b6"][:len(d)])
    ax.set_xlabel("R² (5-fold CV)"); ax.set_xlim(0, 1)
    ax.set_title("Feature-based (resource) MoE vs baselines")
    ax.grid(axis="x", linestyle=":", alpha=0.4)
    plt.tight_layout(); plt.savefig(path, dpi=150)
    print(f"Saved plot: {path}")


if __name__ == "__main__":
    main()
