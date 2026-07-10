"""Label ingestion (step 2): match true_energy back to logged predictions.

Reads predictions.jsonl + a stream of (request_id, true_energy) labels, writes
matched samples to labeled_samples.jsonl. Unknown request_ids are recorded as
errors, never crash.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import logging as predlog

DEFAULT_LABELED = Path("data/online_logs/labeled_samples.jsonl")
DEFAULT_ERRORS = Path("data/online_logs/label_errors.jsonl")


def _index_predictions(preds: list[dict]) -> dict[str, dict]:
    # last write wins if a request_id repeats
    return {p["request_id"]: p for p in preds}


def ingest_labels(
    labels: list[dict],
    *,
    log_path: str | Path = predlog.DEFAULT_LOG,
    labeled_path: str | Path = DEFAULT_LABELED,
    errors_path: str | Path = DEFAULT_ERRORS,
) -> dict:
    """labels: [{'request_id':..., 'true_energy':...}, ...].

    Returns a summary dict. Matched samples are appended to labeled_path;
    unmatched request_ids go to errors_path.
    """
    preds = _index_predictions(predlog.read_predictions(log_path))
    labeled_path = Path(labeled_path)
    errors_path = Path(errors_path)
    labeled_path.parent.mkdir(parents=True, exist_ok=True)

    n_matched = n_dup = n_missing = 0
    seen = set()
    with labeled_path.open("a") as lf, errors_path.open("a") as ef:
        for lab in labels:
            rid = lab.get("request_id")
            te = lab.get("true_energy")
            if rid is None or te is None:
                ef.write(json.dumps({"reason": "missing_field", "label": lab}) + "\n")
                n_missing += 1
                continue
            if rid not in preds:
                ef.write(json.dumps({"reason": "unknown_request_id", "request_id": rid}) + "\n")
                n_missing += 1
                continue
            if rid in seen:
                ef.write(json.dumps({"reason": "duplicate_label", "request_id": rid}) + "\n")
                n_dup += 1
                continue
            seen.add(rid)
            p = preds[rid]
            rec = {
                "request_id": rid,
                "timestamp": p["timestamp"],
                "features": p["features"],
                "prediction": p["prediction"],
                "true_energy": float(te),
                "model_version": p["model_version"],
                "expert": p.get("expert"),
                "gate_weights": p.get("gate_weights"),
                "run_name": lab.get("run_name") or p.get("run_name") or p.get("features", {}).get("run_name"),
                "workload_kind": lab.get("workload_kind") or p.get("workload_kind") or p.get("features", {}).get("workload_kind"),
                "source": lab.get("source", "online"),
            }
            lf.write(json.dumps(rec) + "\n")
            n_matched += 1
    return {"matched": n_matched, "missing": n_missing, "duplicate": n_dup}


def read_labeled(labeled_path: str | Path = DEFAULT_LABELED) -> list[dict]:
    p = Path(labeled_path)
    if not p.exists():
        return []
    out = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
