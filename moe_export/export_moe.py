"""Export a trained MoE as a self-contained inference artifact.

The monitor's inference service must not depend on the ``moe`` training package at
runtime (it ships in a separate container). So we unpack the fitted
``MixtureOfExperts`` into its raw scikit-learn parts — the gate classifier, the
per-label expert regressors, the feature order and labels — and pickle just those.
Every stored object (RandomForestClassifier, RandomForestRegressor, or a
StandardScaler+LinearRegression Pipeline) is standalone-picklable, so the artifact
loads with only scikit-learn present.

The artifact dict carries ``model_type="moe"`` so ``InferenceRequest`` can tell it
apart from the legacy flat linear artifacts (which have no ``model_type``).

Usage:
    python -m offloading_export.export_moe --expert linear \
        --out ../../monitor/models/moe_linear.pkl
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

from moe import data
from moe.moe import MixtureOfExperts, MoEConfig


def build_artifact(expert: str, n_estimators: int) -> dict:
    datasets = data.load_all()
    X, y, w = data.combined_frame(datasets)
    cfg = MoEConfig(expert=expert, n_estimators=n_estimators)
    moe = MixtureOfExperts(cfg).fit(X, y, w)

    return {
        "model_type": "moe",
        "features": list(X.columns),
        "target": data.TARGET,
        "labels": list(moe.labels_),
        "gate": moe.gate,                 # sklearn classifier: features -> workload
        "experts": dict(moe.experts),     # label -> sklearn regressor
        "expert_class": expert,
        "source_workloads": list(datasets.keys()),
    }


def main():
    ap = argparse.ArgumentParser(description="Export MoE to a self-contained inference artifact")
    ap.add_argument("--expert", choices=["rf", "linear"], default="linear")
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--out", required=True, help="Output .pkl path")
    args = ap.parse_args()

    artifact = build_artifact(args.expert, args.n_estimators)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as fh:
        pickle.dump(artifact, fh)

    print(f"Wrote {out}")
    print(f"  model_type : {artifact['model_type']}")
    print(f"  expert     : {artifact['expert_class']}")
    print(f"  features   : {len(artifact['features'])}")
    print(f"  experts    : {list(artifact['experts'])}")


if __name__ == "__main__":
    main()
