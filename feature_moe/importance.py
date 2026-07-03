"""Resource importance (architecture Module 4.3).

Computes each resource group's share of predictive contribution, via permutation
importance on a baseline RandomForest. Used for:
  * the "Resource Importance" table (doc §4.3),
  * an ablation gate (importance-weighted) vs the learned gate.

This is NOT the production gate — the production gate is learned (see moe.py).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .groups import GROUP_ORDER, RESOURCE_GROUPS


def resource_importance(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> dict[str, float]:
    """Per-feature permutation importance summed per resource group, normalized to sum=1."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    rf = RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    perm = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=seed, n_jobs=-1)
    fi = pd.Series(perm.importances_mean, index=X.columns)

    group_imp = {}
    for g in GROUP_ORDER:
        cols = [c for c in RESOURCE_GROUPS[g] if c in X.columns]
        group_imp[g] = float(fi[cols].sum()) if cols else 0.0
    total = sum(group_imp.values())
    if total <= 0:
        # uniform fallback if importance is degenerate
        return {g: 1.0 / len(GROUP_ORDER) for g in GROUP_ORDER}
    return {g: max(0.0, v) / total for g, v in group_imp.items()}


def importance_weights(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> np.ndarray:
    """Return the 4-vector of resource importance weights (sums to 1, GROUP_ORDER)."""
    imp = resource_importance(X, y, seed=seed)
    return np.array([imp[g] for g in GROUP_ORDER], dtype=float)
