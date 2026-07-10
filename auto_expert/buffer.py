"""Training buffer (step 3): turn labeled samples into a model-trainable frame.

Loads labeled_samples.jsonl, returns (X DataFrame, y Series, meta). Also persists
a parquet snapshot for inspection. Deduplicates by request_id.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from . import label_ingestion

DEFAULT_BUFFER = Path("data/training_buffer/buffer.parquet")


def build_buffer(
    *,
    labeled_path: str | Path = label_ingestion.DEFAULT_LABELED,
    buffer_path: str | Path = DEFAULT_BUFFER,
) -> dict:
    """Build X/y from labeled samples and snapshot to parquet. Returns summary."""
    rows = label_ingestion.read_labeled(labeled_path)
    if not rows:
        return {"n": 0, "status": "empty"}
    # dedup by request_id (keep last)
    by_id = {r["request_id"]: r for r in rows}
    rows = list(by_id.values())

    # feature columns = union of all feature keys seen (so new features show up)
    feat_keys: list[str] = []
    seen = set()
    for r in rows:
        for k in r.get("features", {}).keys():
            if k not in seen:
                seen.add(k); feat_keys.append(k)

    X = pd.DataFrame([{k: float(r["features"].get(k, 0.0)) for k in feat_keys} for r in rows])
    y = pd.Series([float(r["true_energy"]) for r in rows], name="true_energy")
    meta = pd.DataFrame([{
        "request_id": r["request_id"],
        "timestamp": r["timestamp"],
        "prediction": r["prediction"],
        "model_version": r["model_version"],
        "expert": r.get("expert"),
        "run_name": r.get("run_name"),
        "workload_kind": r.get("workload_kind"),
        "source": r.get("source", "online"),
    } for r in rows])

    buffer_path = Path(buffer_path)
    buffer_path.parent.mkdir(parents=True, exist_ok=True)
    snapshot = X.copy()
    snapshot["true_energy"] = y.values
    snapshot["error"] = meta["prediction"].astype(float).values - y.values
    snapshot["abs_error"] = snapshot["error"].abs()
    for c in meta.columns:
        snapshot[c] = meta[c].values
    snapshot.to_parquet(buffer_path, index=False)
    return {
        "n": len(rows),
        "n_features": len(feat_keys),
        "feature_keys": feat_keys,
        "status": "ok",
        "buffer_path": str(buffer_path),
    }


def load_buffer(buffer_path: str | Path = DEFAULT_BUFFER) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame] | None:
    p = Path(buffer_path)
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    if len(df) == 0:
        return None
    meta_cols = ["request_id", "timestamp", "prediction", "model_version", "expert", "run_name", "workload_kind", "source", "error", "abs_error"]
    meta = df[[c for c in meta_cols if c in df.columns]].copy()
    y = df["true_energy"].copy()
    drop = {"true_energy", "error", "abs_error"} | set(meta_cols)
    X = df[[c for c in df.columns if c not in drop]].copy()
    return X, y, meta
