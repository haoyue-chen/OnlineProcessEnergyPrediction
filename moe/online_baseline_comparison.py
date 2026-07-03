"""Unified online-vs-offline baseline comparison with per-workload metrics.

Trains and evaluates the seven model families from the project's online-learning
table in ONE place, so they sit in a single comparison:

    Linear (OLS)        offline   batch  "linear"
    SGD (linear+Adam)   online    river  "linear"
    RF                  offline   batch  "rf"
    LightGBM / HGB      offline   batch  "hgb"   (HistGB stand-in for LightGBM)
    Hoeffding           online    river  "htr"
    Hoeffding Adaptive  online    river  "hatr"
    Adaptive RF         online    river  "arf"

Protocols are honestly different and labelled as such:
  * offline (batch)  — 5-fold CV; per-workload metrics from out-of-fold preds.
  * online           — prequential (test-then-train) over the drift stream
                       DAW1 → DAW2 → Phoronix → Stress; per-workload metrics from
                       the stream rows of each workload (warm-up rows excluded).

Outputs overall + per-workload R²/MAE to
``results/moe/online_baseline_comparison.md`` (and .txt).

Usage:
    python -m moe.online_baseline_comparison
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold

from . import data
from .moe import MoEConfig, SingleModel
from .registry import make_batch_model, make_online_model

warnings.filterwarnings("ignore")

# (display name, family, registry kind, registry key, online?)
MODELS = [
    ("Linear (OLS)", "linear", "batch", "linear", False),
    ("SGD (linear+Adam)", "SGD", "online", "linear", True),
    ("RF", "random forest", "batch", "rf", False),
    ("LightGBM/HGB", "boosted trees", "batch", "hgb", False),
    ("Hoeffding", "Hoeffding tree", "online", "htr", True),
    ("Hoeffding Adaptive", "Hoeffding adaptive", "online", "hatr", True),
    ("Adaptive RF", "adaptive forest", "online", "arf", True),
]

WARMUP_FRAC = 0.15


def _per_workload(y_true, y_pred, w, order):
    out = {}
    for lbl in order:
        m = w == lbl
        out[lbl] = (r2_score(y_true[m], y_pred[m]), mean_absolute_error(y_true[m], y_pred[m]))
    return out


def eval_batch(key, X, y, w, order, cv=5):
    """5-fold CV out-of-fold predictions for a batch model; per-workload metrics."""
    cfg = MoEConfig(expert=key, n_estimators=150)
    kf = KFold(n_splits=cv, shuffle=True, random_state=0)
    pred = np.empty(len(y))
    for tr, te in kf.split(X):
        pred[te] = SingleModel(cfg).fit(X.iloc[tr], y.iloc[tr]).predict(X.iloc[te])
    yv, wv = y.values, w.values
    overall = (r2_score(yv, pred), mean_absolute_error(yv, pred))
    return overall, _per_workload(yv, pred, wv, order)


def eval_online(key, datasets, order):
    """Prequential streaming for an online model; per-workload metrics.

    Warm-up head of each workload is learned but excluded from scoring so the
    comparison is fair (model has seen some of each regime before being scored).
    """
    # Build the drift stream + warm mask in workload order.
    Xs, ys, ws, warm = [], [], [], []
    for lbl in order:
        ds = datasets[lbl]
        n = len(ds); cut = int(n * WARMUP_FRAC)
        Xs.append(ds.X); ys.append(ds.y)
        ws.append(pd.Series([lbl] * n))
        warm.append(pd.Series([True] * cut + [False] * (n - cut)))
    X = pd.concat(Xs, ignore_index=True)
    y = pd.concat(ys, ignore_index=True)
    w = pd.concat(ws, ignore_index=True).values
    is_warm = pd.concat(warm, ignore_index=True).values
    feats = list(X.columns); Xv, yv = X.values, y.values

    model = make_online_model(key)
    pred = np.full(len(yv), np.nan)
    for i in range(len(Xv)):
        xd = {f: float(v) for f, v in zip(feats, Xv[i])}
        if is_warm[i]:
            model.learn_one(xd, float(yv[i]))
            continue
        p = model.predict_one(xd)
        pred[i] = p if p is not None else 0.0
        model.learn_one(xd, float(yv[i]))

    ev = ~is_warm
    overall = (r2_score(yv[ev], pred[ev]), mean_absolute_error(yv[ev], pred[ev]))
    per = {}
    for lbl in order:
        m = ev & (w == lbl)
        per[lbl] = (r2_score(yv[m], pred[m]), mean_absolute_error(yv[m], pred[m]))
    return overall, per


def build_table(datasets) -> pd.DataFrame:
    order = list(datasets)
    X, y, w = data.combined_frame(datasets)
    rows = []
    for disp, family, kind, key, online in MODELS:
        if kind == "batch":
            overall, per = eval_batch(key, X, y, w, order)
            protocol = "5-fold CV"
        else:
            overall, per = eval_online(key, datasets, order)
            protocol = "prequential"
        row = {
            "model": disp, "family": family,
            "online": "yes" if online else "no", "protocol": protocol,
            "R2_overall": overall[0], "MAE_overall": overall[1],
        }
        for lbl in order:
            row[f"R2_{lbl}"] = per[lbl][0]
            row[f"MAE_{lbl}"] = per[lbl][1]
        rows.append(row)
        print(f"  {disp:20s} [{protocol:11s}] R²={overall[0]:.3f} MAE={overall[1]:.2f}")
    return pd.DataFrame(rows)


def render_markdown(df: pd.DataFrame, order) -> str:
    lines = []
    lines.append("# Online vs Offline Baseline Comparison\n")
    lines.append("Seven model families on the same datasets. **Protocol differs by "
                 "type and is labelled**: offline models use 5-fold CV; online models "
                 "use prequential (test-then-train) over the drift stream "
                 "`DAW1 → DAW2 → Phoronix → Stress` (warm-up rows excluded).\n")
    lines.append("## Overall\n")
    lines.append("| Model | Family | Online | Protocol | R² | MAE (Wh) |")
    lines.append("|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        lines.append(f"| {r['model']} | {r['family']} | {r['online']} | {r['protocol']} "
                     f"| {r['R2_overall']:.3f} | {r['MAE_overall']:.2f} |")
    lines.append("\n## Per-workload R²\n")
    hdr = "| Model | Online | " + " | ".join(order) + " |"
    lines.append(hdr)
    lines.append("|" + "---|" * (len(order) + 2))
    for _, r in df.iterrows():
        vals = " | ".join(f"{r[f'R2_{l}']:.3f}" for l in order)
        lines.append(f"| {r['model']} | {r['online']} | {vals} |")
    lines.append("\n## Per-workload MAE (Wh)\n")
    lines.append(hdr)
    lines.append("|" + "---|" * (len(order) + 2))
    for _, r in df.iterrows():
        vals = " | ".join(f"{r[f'MAE_{l}']:.2f}" for l in order)
        lines.append(f"| {r['model']} | {r['online']} | {vals} |")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Unified online/offline baseline comparison")
    ap.add_argument("--out-dir", default="results/moe")
    args = ap.parse_args()

    print("Loading workloads …")
    datasets = data.load_all()
    order = list(datasets)
    print("Evaluating models (this trains all 7 families) …")
    df = build_table(datasets)

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    md = render_markdown(df, order)
    (out_dir / "online_baseline_comparison.md").write_text(md)
    df.to_csv(out_dir / "online_baseline_comparison.csv", index=False)
    # Plain-text copy too (requirement allows .txt).
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    (out_dir / "online_baseline_comparison.txt").write_text(df.to_string(index=False))

    print("\n" + md)
    print(f"Saved: {out_dir}/online_baseline_comparison.{{md,csv,txt}}")


if __name__ == "__main__":
    main()
