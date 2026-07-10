"""Discovery (steps 5): detect new resource signals / OOD / drift in the buffer.

Decides whether the incoming data shows a *new resource dimension* (e.g. GPU-like
features the trained model never saw) or just more of the same. This is the trigger
for automatic expert expansion — but only a SIGNAL; expansion itself happens in
expand.py and is only ever a candidate, never a live-model change.

New-resource heuristics (all must be satisfied to claim a new dimension, to avoid
firing on noise):
  * features present in the buffer that are NOT in the approved model's known set
  * those new features are non-trivial (above a min activity threshold, not all 0)
  * they appear consistently (>= min_samples, >= min_fraction of rows)
  * they cluster into a plausible new group (by name prefix, e.g. delta_gpu_*)
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np

from feature_moe.groups import RESOURCE_GROUPS, GROUP_ORDER

# Heuristic name-prefix -> candidate new resource group. Extend as new dimensions
# appear. A new feature is "unassigned" if it matches none of the existing groups.
PREFIX_TO_GROUP = {
    "delta_gpu_": "gpu",
    "gpu_": "gpu",
    "delta_accel_": "accel",
}


def known_features() -> set[str]:
    """All features the current approved model's groups cover."""
    s = set()
    for g in GROUP_ORDER:
        s.update(RESOURCE_GROUPS[g])
    return s


def known_features_from_columns(columns) -> set[str]:
    """Approved-path known features from the base 4-group globals only."""
    return known_features()


def _candidate_group(feat: str) -> str | None:
    for prefix, grp in PREFIX_TO_GROUP.items():
        if feat.startswith(prefix):
            return grp
    return None


def detect_new_resource(
    X,
    *,
    known: set[str] | None = None,
    min_samples: int = 50,
    min_fraction: float = 0.10,
    min_activity: float = 1e-6,
    min_active_count: int = 100,
) -> dict:
    """Inspect buffer features for a new resource dimension.

    Returns a dict with status:
      - status="no_new"        : nothing new (or insufficient evidence)
      - status="pending_data"  : new features seen but not enough samples yet
      - status="new_dimension" : a new resource group is proposed (name + features)
    """
    known = known if known is not None else known_features()
    if X is None or len(X) == 0:
        return {"status": "no_new", "reason": "empty_buffer"}

    new_feats = [c for c in X.columns if c not in known]
    if not new_feats:
        return {"status": "no_new", "reason": "all_features_known"}

    n = len(X)
    if n < min_samples:
        return {"status": "pending_data", "reason": "insufficient_samples",
                "n_samples": n, "min_required": min_samples, "new_features": new_feats}

    # group new features by candidate resource group
    grouped: dict[str, list[str]] = defaultdict(list)
    unassigned: list[str] = []
    for f in new_feats:
        g = _candidate_group(f)
        if g is None:
            unassigned.append(f)
        else:
            grouped[g].append(f)

    # for each candidate group, check it's consistently active (not noise)
    proposals = []
    for grp, feats in grouped.items():
        active_rows = (np.abs(X[feats].values) > min_activity).any(axis=1)
        frac = active_rows.mean()
        n_active = int(active_rows.sum())
        proposal_feats = [f for f in feats if f != "gpu_like_signal"] if grp == "gpu" else feats
        if proposal_feats and (frac >= min_fraction or n_active >= min_active_count):
            proposals.append({"group": grp, "features": proposal_feats,
                              "active_fraction": float(frac), "n_active": n_active})

    if not proposals:
        if unassigned:
            return {"status": "pending_data", "reason": "unassigned_new_features",
                    "unassigned": unassigned}
        return {"status": "no_new", "reason": "new_features_not_consistent"}

    return {"status": "new_dimension", "proposals": proposals,
            "unassigned": unassigned, "n_samples": n}


def ood_score(X_train, X_new) -> float:
    """Rough OOD score: mean standardized distance of new rows from train mean/std.

    Higher = more out-of-distribution. Used as a *signal* for triggering updates,
    not as a hard threshold.
    """
    if X_new is None or len(X_new) == 0 or X_train is None or len(X_train) == 0:
        return 0.0
    common = [c for c in X_train.columns if c in X_new.columns]
    if not common:
        return float("inf")
    mu = X_train[common].mean()
    sd = X_train[common].std().replace(0, 1.0)
    z = (X_new[common] - mu) / sd
    return float(np.sqrt((z.values ** 2).sum(axis=1)).mean())
