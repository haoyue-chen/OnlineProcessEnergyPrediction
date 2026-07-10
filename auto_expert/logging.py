"""Online prediction logging (step 1 of the safe online-learning pipeline).

Every serve-online /predict is appended as a JSON line so labels can be matched
back later by request_id. Thread-safe (the server may call concurrently).
"""
from __future__ import annotations

import json
import os
import threading
import time
import uuid
from pathlib import Path

_LOCK = threading.Lock()
DEFAULT_LOG = Path("data/online_logs/predictions.jsonl")


def new_request_id() -> str:
    return uuid.uuid4().hex


def log_prediction(
    *,
    log_path: str | Path = DEFAULT_LOG,
    request_id: str | None = None,
    features: dict,
    prediction: float,
    model_version,
    expert: str | None = None,
    gate_weights: dict | None = None,
    timestamp: float | None = None,
) -> str:
    """Append one prediction record. Returns the request_id used.

    ``model_version`` may be an int (legacy online artifacts) or a string
    (approved candidate versions like ``cand-...-retrain``).
    """
    request_id = request_id or new_request_id()
    if isinstance(model_version, bool):
        mv = model_version
    else:
        try:
            mv = int(model_version)
        except (TypeError, ValueError):
            mv = model_version
    rec = {
        "request_id": request_id,
        "timestamp": timestamp if timestamp is not None else time.time(),
        "features": features,
        "prediction": float(prediction),
        "model_version": mv,
        "expert": expert,
        "gate_weights": gate_weights,
    }
    p = Path(log_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _LOCK, p.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")
    return request_id


def read_predictions(log_path: str | Path = DEFAULT_LOG) -> list[dict]:
    p = Path(log_path)
    if not p.exists():
        return []
    out = []
    with p.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out
