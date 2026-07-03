"""Online predictor: gate-routed River experts that update incrementally.

Wraps a "moe-online" artifact (gate + per-workload River online experts). Supports:
  * predict(features)         -> (energy_wh, expert, prediction_id)
  * update(features, y_true)  -> incremental learn on the routed expert
  * persistence: load from / save to a state pickle so updates survive restarts
  * a lock so concurrent /update calls don't corrupt model state

State precedence: if a persisted ``state_path`` exists, it is loaded (continues
prior online learning); otherwise the immutable ``base_path`` warm-started model is
loaded and the first save writes the state file.

Depends only on river + sklearn + numpy (no training-side ``moe`` package).
"""

from __future__ import annotations

import pickle
import threading
import uuid
from pathlib import Path
from typing import Any

import numpy as np


class OnlinePredictor:
    def __init__(self, base_path: str, state_path: str):
        self.base_path = Path(base_path)
        self.state_path = Path(state_path)
        self._lock = threading.Lock()
        # Pending predictions awaiting ground truth: prediction_id -> (features, expert)
        self._pending: dict[str, tuple[dict, str]] = {}
        self._load()

    # --- persistence -------------------------------------------------------
    def _load(self) -> None:
        path = self.state_path if self.state_path.exists() else self.base_path
        with path.open("rb") as fh:
            art = pickle.load(fh)
        if art.get("model_type") != "moe-online":
            raise ValueError(f"Expected moe-online artifact, got {art.get('model_type')!r}")
        self.features: list[str] = list(art["features"])
        self.labels: list[str] = list(art["labels"])
        self.gate = art["gate"]
        self.experts: dict[str, Any] = dict(art["experts"])
        self.online_expert = art.get("online_expert", "?")
        self.num_updates = int(art.get("num_updates", 0))
        self.model_version = int(art.get("model_version", 0))
        self._loaded_from = str(path)

    def _save(self) -> None:
        art = {
            "model_type": "moe-online",
            "online_expert": self.online_expert,
            "features": self.features,
            "labels": self.labels,
            "gate": self.gate,
            "experts": self.experts,
            "num_updates": self.num_updates,
            "model_version": self.model_version,
            "source_workloads": self.labels,
        }
        # Atomic write: temp file then replace, so a crash mid-write can't corrupt.
        tmp = self.state_path.with_suffix(".tmp")
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        with tmp.open("wb") as fh:
            pickle.dump(art, fh)
        tmp.replace(self.state_path)

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _to_float(v: Any) -> float:
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _feat_dict(self, features: dict[str, Any]) -> dict[str, float]:
        return {f: self._to_float(features.get(f, 0.0)) for f in self.features}

    def _route(self, fd: dict[str, float]) -> str:
        row = np.array([[fd[f] for f in self.features]], dtype=float)
        return str(self.gate.predict(row)[0])

    # --- API ---------------------------------------------------------------
    def predict(self, features: dict[str, Any]) -> dict:
        fd = self._feat_dict(features)
        with self._lock:
            expert = self._route(fd)
            yhat = self.experts[expert].predict_one(fd)
            pid = uuid.uuid4().hex
            self._pending[pid] = (fd, expert)
            return {
                "prediction_id": pid,
                "energy_wh": float(yhat if yhat is not None else 0.0),
                "expert": expert,
                "model_version": self.model_version,
            }

    def update(self, *, true_energy_wh: float,
               prediction_id: str | None = None,
               features: dict[str, Any] | None = None,
               expert: str | None = None) -> dict:
        y = self._to_float(true_energy_wh)
        with self._lock:
            # Resolve features + expert: prefer a pending prediction_id.
            if prediction_id and prediction_id in self._pending:
                fd, routed = self._pending.pop(prediction_id)
            elif features is not None:
                fd = self._feat_dict(features)
                routed = expert if expert in self.experts else self._route(fd)
            else:
                raise ValueError("update needs a known prediction_id or features")
            if expert in self.experts:   # explicit override always wins
                routed = expert

            self.experts[routed].learn_one(fd, y)
            self.num_updates += 1
            self.model_version += 1
            self._save()
            return {
                "updated_expert": routed,
                "true_energy_wh": y,
                "num_updates": self.num_updates,
                "model_version": self.model_version,
            }

    def predict_then_update(self, features: dict[str, Any], true_energy_wh: float) -> dict:
        pred = self.predict(features)
        upd = self.update(prediction_id=pred["prediction_id"], true_energy_wh=true_energy_wh)
        return {**pred, **upd}

    def info(self) -> dict:
        return {
            # Public API name. Internally the artifact/state still carries
            # "moe-online" (see _load/_save) for backward state compatibility.
            "model_type": "online_moe",
            "online_expert": self.online_expert,
            "n_features": len(self.features),
            "features": self.features,
            "labels": self.labels,
            "num_updates": self.num_updates,
            "model_version": self.model_version,
            "loaded_from": self._loaded_from,
            "pending_predictions": len(self._pending),
        }
