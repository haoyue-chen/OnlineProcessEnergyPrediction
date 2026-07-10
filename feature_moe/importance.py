"""Resource importance (architecture Module 4.3).

Computes each resource group's share of predictive contribution, via permutation
importance on a baseline RandomForest. Used for:
  * the "Resource Importance" table (doc §4.3),
  * an ablation gate (importance-weighted) vs the learned gate.

This is NOT the production gate — the production gate is learned (see moe.py /
learned_moe.py).

Degenerate handling: when the summed permutation importance is <= 0 (the model
failed to find signal, e.g. a workload it can barely predict), we do NOT return
a misleading uniform 0.25 vector. We return NaN weights with status="degenerate"
so reports must say so explicitly. A uniform-looking result here means "unreliable",
never "balanced routing".
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split

from .groups import GROUP_ORDER, RESOURCE_GROUPS


def _group_importance_raw(X: pd.DataFrame, y: pd.Series, seed: int):
    """Return (group_imp dict, total, baseline RF test R2)."""
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    rf = RandomForestRegressor(n_estimators=120, n_jobs=-1, random_state=seed).fit(Xtr, ytr)
    perm = permutation_importance(rf, Xte, yte, n_repeats=5, random_state=seed, n_jobs=-1)
    fi = pd.Series(perm.importances_mean, index=X.columns)
    group_imp = {}
    for g in GROUP_ORDER:
        cols = [c for c in RESOURCE_GROUPS[g] if c in X.columns]
        group_imp[g] = float(fi[cols].sum()) if cols else 0.0
    total = sum(group_imp.values())
    from sklearn.metrics import r2_score
    r2 = float(r2_score(yte, rf.predict(Xte)))
    return group_imp, total, r2


def resource_importance_detail(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> dict:
    """Full report: weights, raw group sums, total, baseline R2, and a status flag.

    status == "degenerate"  -> summed importance <= 0; weights are NaN and MUST be
                               reported as unreliable, not "balanced".
    status == "ok"          -> weights are the normalized (>=0, sum=1) shares.
    """
    group_imp, total, r2 = _group_importance_raw(X, y, seed)
    if total <= 0:
        return {
            "status": "degenerate",
            "weights": {g: float("nan") for g in GROUP_ORDER},
            "raw": group_imp,
            "total": total,
            "baseline_r2": r2,
        }
    weights = {g: max(0.0, v) / total for g, v in group_imp.items()}
    return {
        "status": "ok",
        "weights": weights,
        "raw": group_imp,
        "total": total,
        "baseline_r2": r2,
    }


def resource_importance(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> dict[str, float]:
    """Per-feature permutation importance summed per resource group, normalized to sum=1.

    Degenerate (sum<=0) -> returns NaN for every group (NOT a fake uniform vector).
    Callers that need the status flag should use ``resource_importance_detail``.
    """
    return resource_importance_detail(X, y, seed)["weights"]


def importance_weights(X: pd.DataFrame, y: pd.Series, seed: int = 0) -> np.ndarray:
    """Return the 4-vector of resource importance weights (GROUP_ORDER).

    Degenerate -> all-NaN vector (was: misleading uniform). Downstream code should
    check ``np.isnan`` if it needs a hard fallback; the learned gate (learned_moe.py)
    does not use this.
    """
    imp = resource_importance(X, y, seed=seed)
    return np.array([imp[g] for g in GROUP_ORDER], dtype=float)
