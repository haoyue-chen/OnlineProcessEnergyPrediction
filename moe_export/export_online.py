"""Export an *online* MoE base model for the live online-learning service.

Unlike ``export_moe.py`` (static sklearn experts), this builds:
  * gate    — sklearn RandomForest classifier (features -> workload), routing only;
  * experts — one **River online regressor per workload** (default ARF, the
              strongest online model in the baseline survey), warm-started on that
              workload's data so the service starts useful rather than cold.

The artifact is self-contained: it pickles the trained gate + the warm-started
river expert objects + feature order + labels. At serve time the online service
loads this as the *base*, then keeps updating the experts incrementally via
``/update`` and persists the evolving state separately (online_state.pkl).

Usage:
    python -m moe_export.export_online --online-expert arf --out models/online_base.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from moe import data
from moe.moe import MixtureOfExperts, MoEConfig
from moe.registry import make_online_model

WARMUP_FRAC = 0.15


def build_online_artifact(online_expert: str) -> dict:
    datasets = data.load_all()
    X, y, w = data.combined_frame(datasets)
    features = list(X.columns)
    labels = list(datasets.keys())

    # Gate: reuse the proven RF gate (routing only — never updated online).
    gate = MixtureOfExperts(MoEConfig(expert="rf")).fit(X, y, w).gate

    # One online expert per workload, warm-started on the head of that workload.
    experts = {}
    for label in labels:
        ds = datasets[label]
        cut = max(1, int(len(ds) * WARMUP_FRAC))
        model = make_online_model(online_expert)
        Xv = ds.X.values[:cut]
        yv = ds.y.values[:cut]
        for i in range(len(Xv)):
            xd = {f: float(v) for f, v in zip(features, Xv[i])}
            model.learn_one(xd, float(yv[i]))
        experts[label] = model
        print(f"  warm-started {label} expert on {cut} samples")

    return {
        "model_type": "moe-online",
        "online_expert": online_expert,
        "features": features,
        "target": data.TARGET,
        "labels": labels,
        "gate": gate,
        "experts": experts,
        "num_updates": 0,
        "model_version": 0,
        "source_workloads": labels,
    }


def main():
    ap = argparse.ArgumentParser(description="Export an online MoE base model")
    ap.add_argument("--online-expert", choices=["arf", "linear", "htr", "hatr", "knn", "pa"],
                    default="arf", help="River online expert per workload (default arf).")
    ap.add_argument("--out", required=True, help="Output .pkl path")
    args = ap.parse_args()

    artifact = build_online_artifact(args.online_expert)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(artifact, fh)

    print(f"Wrote {out}")
    print(f"  model_type    : {artifact['model_type']}")
    print(f"  online_expert : {artifact['online_expert']}")
    print(f"  features      : {len(artifact['features'])}")
    print(f"  experts       : {list(artifact['experts'])}")


if __name__ == "__main__":
    main()
