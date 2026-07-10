"""Approved-model predictor for the safe online-learning serving path.

Loads ONLY approved candidate models, selected via ``models/latest_approved.json``.
If no approved model exists, it falls back to the legacy base MoE artifact
(``models/moe_linear.pkl``). It NEVER loads anything from ``models/pending/``.

This wrapper supports two artifact shapes:
  * approved ``CandidateModel`` objects (auto_expert pipeline) — exposes
    ``gate_mode`` and per-sample gate weights when available.
  * legacy ``EnergyPredictor``-style dict artifacts (``model_type == "moe"``).

The /predict response always carries ``model_version`` (and ``gate_mode`` when
the loaded model exposes it).
"""

from __future__ import annotations

import json
import pickle
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from .predictor import EnergyPredictor

LATEST_FILE = Path("models/latest_approved.json")
DEFAULT_BASE_PATH = "models/moe_linear.pkl"


class ApprovedPredictor:
    """Serve only approved models, with a safe base-model fallback."""

    def __init__(self, latest_file: str | Path = LATEST_FILE, base_path: str = DEFAULT_BASE_PATH):
        self.latest_file = Path(latest_file)
        self.base_path = base_path
        self._load()

    def _load(self) -> None:
        # 1) Prefer the latest approved model, if the registry exists and is valid.
        if self.latest_file.exists():
            latest = json.loads(self.latest_file.read_text())
            approved_path = Path(latest["model_path"])
            if approved_path.exists():
                # Hard guard: never serve anything under models/pending/.
                if "models/pending" in str(approved_path):
                    raise RuntimeError(
                        f"latest_approved.json points into pending space: {approved_path}"
                    )
                with approved_path.open("rb") as fh:
                    model = pickle.load(fh)
                self._model = model
                self._kind = "approved"
                self._version = latest.get("approved_version") or getattr(
                    getattr(model, "spec", None), "candidate_version", None
                )
                self._model_path = str(approved_path)
                self._gate_mode = getattr(getattr(model, "spec", None), "gate_mode", None)
                self._features = getattr(model, "feature_names", None)
                return
            # Registry exists but the approved artifact is missing — fall through
            # to the base model rather than serving nothing.

        # 2) Safe fallback: legacy base MoE artifact.
        base = Path(self.base_path)
        if not base.exists():
            raise FileNotFoundError(f"No approved model and no base artifact at {base}")
        self._model = EnergyPredictor(str(base))
        self._kind = "base_fallback"
        self._version = "base"
        self._model_path = str(base)
        self._gate_mode = None
        self._features = getattr(self._model, "features", None)

    # --- helpers -----------------------------------------------------------
    @staticmethod
    def _to_float(v: Any) -> float:
        try:
            return float(v) if v is not None else 0.0
        except (TypeError, ValueError):
            return 0.0

    def _feature_columns(self) -> list[str]:
        if self._features:
            return list(self._features)
        # Legacy EnergyPredictor carries its own feature list.
        return list(getattr(self._model, "features", []))

    def _to_frame(self, features: dict[str, Any]):
        import pandas as pd
        cols = self._feature_columns()
        row = {c: self._to_float(features.get(c, 0.0)) for c in cols}
        # Approved CandidateModel.predict expects a DataFrame with its columns;
        # include any extra request keys harmlessly is not needed, just use cols.
        return pd.DataFrame([row], columns=cols)

    # --- API ---------------------------------------------------------------
    def predict(self, features: dict[str, Any]) -> dict:
        gate_weights = None
        if self._kind == "approved":
            X = self._to_frame(features)
            # CandidateModel exposes predict_with_gate -> (pred, weights, groups)
            if hasattr(self._model, "predict_with_gate"):
                pred, weights, groups = self._model.predict_with_gate(X)
                energy = float(np.asarray(pred).ravel()[0])
                if weights is not None and np.asarray(weights).size:
                    w = np.asarray(weights)
                    row = w[0] if w.ndim > 1 else w
                    gate_weights = {g: float(row[i]) for i, g in enumerate(groups)}
                expert = max(gate_weights, key=gate_weights.get) if gate_weights else None
            else:
                energy = float(np.asarray(self._model.predict(X)).ravel()[0])
                expert = None
        else:
            # Legacy base MoE artifact.
            sample = {f: self._to_float(features.get(f, 0.0)) for f in self._model.features}
            expert = self._model.route(sample)
            energy = float(self._model.predict_sample(sample))

        return {
            "prediction_id": uuid.uuid4().hex,
            "energy_wh": energy,
            "expert": expert,
            "model_version": self._version,
            "model_kind": self._kind,
            "model_path": self._model_path,
            "gate_mode": self._gate_mode,
            "gate_weights": gate_weights,
        }

    def info(self) -> dict:
        return {
            "model_kind": self._kind,
            "model_version": self._version,
            "model_path": self._model_path,
            "gate_mode": self._gate_mode,
            "n_features": len(self._feature_columns()),
            "features": self._feature_columns(),
        }
