"""Mixture-of-Experts (MoE) energy estimator (Phase-2, Task 3 / Task 4).

Architecture (per the Phase-2 plan):

    Interval features
            |
          Gate  (classifier: which workload regime?)
            |
      Expert selection
            |
       Expert model  (one regressor specialised per workload)
            |
      Energy prediction

The motivation is Finding 3 from Phase 1: different workloads have different
resource-behaviour patterns and very different predictability, so a single
"one model fits all" regressor leaves accuracy on the table. Each expert is a
RandomForest (Finding 2/4: tree models clearly beat linear, and RF ~= LightGBM,
so RF is the representative tree expert here). The gate is a RandomForest
classifier that routes each interval to an expert from its features alone — i.e.
at inference time we do NOT need to know the workload label, the gate infers it.

A single global RandomForest trained on all workloads is provided as the
baseline to compare against (MoE vs Single Model).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from .registry import make_batch_model


@dataclass
class MoEConfig:
    n_estimators: int = 200
    max_depth: int | None = None
    random_state: int = 0
    n_jobs: int = -1
    # Expert model class — any key in registry.BATCH_MODELS (e.g. "rf", "linear",
    # "hgb", "extra_trees", "knn", "mlp", "svr"). Default "rf" is the strong tree
    # expert; "linear" (the paper's class) best shows MoE's benefit over a single
    # global model.
    expert: str = "rf"


def _make_regressor(cfg: MoEConfig):
    return make_batch_model(cfg.expert, cfg)


class MixtureOfExperts:
    """Gate + per-workload experts.

    ``fit`` expects features ``X``, target ``y`` and a per-row workload label
    ``w``. The gate learns ``X -> w``; each expert learns ``X -> y`` on its own
    workload slice. ``predict`` routes each row through the gate to its expert.
    """

    def __init__(self, config: MoEConfig | None = None):
        self.config = config or MoEConfig()
        self.gate: RandomForestClassifier | None = None
        self.experts: dict[str, RandomForestRegressor] = {}
        self.labels_: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, w: pd.Series) -> "MixtureOfExperts":
        cfg = self.config
        self.labels_ = sorted(w.unique())

        self.gate = RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth,
            random_state=cfg.random_state,
            n_jobs=cfg.n_jobs,
        )
        self.gate.fit(X.values, w.values)

        self.experts = {}
        for label in self.labels_:
            mask = (w == label).values
            expert = _make_regressor(cfg)
            expert.fit(X.values[mask], y.values[mask])
            self.experts[label] = expert
        return self

    def route(self, X: pd.DataFrame) -> np.ndarray:
        """Return the expert label chosen by the gate for each row."""
        assert self.gate is not None, "Model not fitted"
        return self.gate.predict(X.values)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Gate each row to an expert, then predict with that expert."""
        assert self.gate is not None, "Model not fitted"
        chosen = self.route(X)
        preds = np.empty(len(X), dtype=float)
        # Batch rows per expert to avoid per-row Python overhead.
        for label in self.labels_:
            mask = chosen == label
            if mask.any():
                preds[mask] = self.experts[label].predict(X.values[mask])
        return preds

    def gate_accuracy(self, X: pd.DataFrame, w: pd.Series) -> float:
        return float((self.route(X) == w.values).mean())


class SingleModel:
    """Baseline: one global RandomForest regressor over all workloads combined."""

    def __init__(self, config: MoEConfig | None = None):
        self.config = config or MoEConfig()
        self.model: RandomForestRegressor | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SingleModel":
        self.model = _make_regressor(self.config)
        self.model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "Model not fitted"
        return self.model.predict(X.values)
