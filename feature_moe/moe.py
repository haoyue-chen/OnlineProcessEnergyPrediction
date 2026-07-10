"""Feature-based (resource-grouped) MoE with a learned soft gate (Module 5).

Architecture (per pipeline_design_architecture.md §5):
  - 4 experts, one per resource group (CPU/Memory/IO/Network). Each expert is a
    regressor trained ONLY on its group's features.
  - a learned gate produces per-resource fusion weights; the prediction is the
    weighted sum of expert outputs (soft routing), not a hard pick.

Gate design — "learned gating" (the team's chosen answer to the doc's open
question #2). Concretely we fit non-negative least-squares weights over the
expert-prediction matrix (stacking with non-negative weights, normalized). This
is a principled *learned soft gate*: the weights are data-driven (not hand-set or
static feature-importance), sum to 1, and are non-negative, exactly the weighted
fusion the doc specifies. An importance-weighted gate is available as an ablation
via ``gate="importance"``.

For online/streaming use the per-group experts can be River models (future work);
this module is the offline core.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from moe.moe import MoEConfig
from moe.registry import make_batch_model

from .groups import GROUP_ORDER, RESOURCE_GROUPS


@dataclass
class ResourceMoEConfig:
    expert: str = "rf"            # any registry batch model
    n_estimators: int = 150
    random_state: int = 0
    n_jobs: int = -1
    gate: str = "learned"         # "learned" (nnls) | "importance" (permutation)
    eps: float = 1e-9


class ResourceMoE:
    """Resource-grouped experts + learned soft gate."""

    def __init__(self, cfg: ResourceMoEConfig | None = None):
        self.cfg = cfg or ResourceMoEConfig()
        self.experts: dict[str, object] = {}
        self.gate_weights: np.ndarray | None = None  # len(GROUP_ORDER), sums to 1
        self._group_cols = {g: RESOURCE_GROUPS[g] for g in GROUP_ORDER}

    def _expert_cfg(self) -> MoEConfig:
        return MoEConfig(expert=self.cfg.expert, n_estimators=self.cfg.n_estimators,
                         random_state=self.cfg.random_state, n_jobs=self.cfg.n_jobs)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ResourceMoE":
        # 1) train one expert per resource group on that group's columns
        self.experts = {}
        for g in GROUP_ORDER:
            cols = [c for c in self._group_cols[g] if c in X.columns]
            exp = make_batch_model(self.cfg.expert, self._expert_cfg())
            exp.fit(X[cols].values, y.values)
            self.experts[g] = (exp, cols)

        # 2) build expert-prediction matrix on the training data
        E = self._expert_matrix(X)

        # 3) learn the gate weights
        if self.cfg.gate == "importance":
            from .importance import importance_weights
            w = importance_weights(X, y, seed=self.cfg.random_state)
            if np.isnan(w).any():
                # Degenerate permutation importance (model couldn't find signal).
                # Do NOT emit a fake uniform vector; fall back to the learned nnls
                # gate so prediction stays valid, and record the degeneracy.
                self._importance_degenerate = True
                w, _ = nnls(E, y.values)
                s = w.sum()
                self.gate_weights = w / s if s > self.cfg.eps else np.ones(len(GROUP_ORDER)) / len(GROUP_ORDER)
            else:
                self._importance_degenerate = False
                self.gate_weights = w
        else:  # learned: non-negative least squares, then normalize
            w, _ = nnls(E, y.values)
            s = w.sum()
            self.gate_weights = w / s if s > self.cfg.eps else np.ones(len(GROUP_ORDER)) / len(GROUP_ORDER)
        return self

    def _expert_matrix(self, X: pd.DataFrame) -> np.ndarray:
        """Columns = per-group expert predictions (one column per group)."""
        cols = []
        for g in GROUP_ORDER:
            exp, feat_cols = self.experts[g]
            cols.append(exp.predict(X[feat_cols].values))
        return np.column_stack(cols)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        E = self._expert_matrix(X)
        return E @ self.gate_weights

    def info(self) -> dict:
        return {
            "model_type": "resource_moe",
            "expert": self.cfg.expert,
            "gate": self.cfg.gate,
            "groups": {g: self._group_cols[g] for g in GROUP_ORDER},
            "gate_weights": {g: float(w) for g, w in zip(GROUP_ORDER, self.gate_weights)},
            "gate_weights_sum": float(self.gate_weights.sum()),
        }


class SingleGlobalModel:
    """Baseline: one global regressor on all features (for comparison)."""

    def __init__(self, cfg: ResourceMoEConfig | None = None):
        self.cfg = cfg or ResourceMoEConfig()
        self.model = None

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SingleGlobalModel":
        self.model = make_batch_model(self.cfg.expert, MoEConfig(
            expert=self.cfg.expert, n_estimators=self.cfg.n_estimators,
            random_state=self.cfg.random_state, n_jobs=self.cfg.n_jobs))
        self.model.fit(X.values, y.values)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X.values)
