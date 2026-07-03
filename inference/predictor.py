"""Self-contained energy predictor for the MoE inference service.

Loads a MoE artifact (``models/moe_linear.pkl``) that was exported as raw
scikit-learn objects (a gate classifier + per-workload expert regressors). It
depends ONLY on scikit-learn / numpy — not on the training-side ``moe`` package —
so it loads cleanly inside a slim container.

Artifact schema (produced by ``moe_export/export_moe.py``):
    model_type : "moe"
    features   : ordered feature names the gate/experts expect
    labels     : workload labels (one expert each)
    gate       : sklearn classifier  features -> workload label
    experts    : {label: sklearn regressor}
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np


class EnergyPredictor:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        with self.model_path.open("rb") as fh:
            art = pickle.load(fh)
        if art.get("model_type") != "moe":
            raise ValueError(f"Expected a MoE artifact, got model_type={art.get('model_type')!r}")
        self.features: list[str] = list(art["features"])
        self.labels: list[str] = list(art["labels"])
        self.gate = art["gate"]
        self.experts: dict[str, Any] = dict(art["experts"])
        self.expert_class = art.get("expert_class", "?")

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _to_float(v: Any) -> float:
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _matrix(self, samples: list[dict[str, Any]]) -> np.ndarray:
        return np.array(
            [[self._to_float(s.get(f, 0.0)) for f in self.features] for s in samples],
            dtype=float,
        )

    # --- prediction --------------------------------------------------------
    def predict_batch(self, samples: list[dict[str, Any]]) -> list[float]:
        """Gate-route each sample to its expert, then predict. Batched per expert."""
        if not samples:
            return []
        rows = self._matrix(samples)
        routed = self.gate.predict(rows)
        preds = np.empty(len(samples), dtype=float)
        for label in self.labels:
            mask = routed == label
            if mask.any():
                preds[mask] = self.experts[label].predict(rows[mask])
        return [float(p) for p in preds]

    def predict_sample(self, sample: dict[str, Any]) -> float:
        return self.predict_batch([sample])[0]

    def route(self, sample: dict[str, Any]) -> str:
        return str(self.gate.predict(self._matrix([sample]))[0])

    def info(self) -> dict:
        return {
            "model_type": "moe",
            "expert_class": self.expert_class,
            "n_features": len(self.features),
            "features": self.features,
            "labels": self.labels,
        }
