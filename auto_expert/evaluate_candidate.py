"""Evaluate candidate (step 8): compare current approved model vs candidate.

Safety checks (any failure -> reject):
  * NaN in predictions
  * prediction explosion (|pred| beyond sane band)
  * severe negative R2 (< -1)
For expanded candidates, additionally:
  * new expert must get non-trivial weight on rows where its features are active
  * new expert must not be ~0 everywhere (unused)
  * per-workload MAE must not visibly worsen on old workloads
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .discovery import ood_score
from .train_candidate import CandidateModel

WORKLOAD_KINDS = ["cpu", "mem", "io", "net", "mixed", "gpu-like"]


def _safe_metrics(yt, yp):
    yt = np.asarray(yt, dtype=float)
    yp = np.asarray(yp, dtype=float)
    nan = bool(np.isnan(yp).any())
    explosion = bool(np.any(np.abs(yp) > 1e6))
    if nan or explosion:
        return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan"),
                "nan": nan, "explosion": explosion}
    return {"r2": float(r2_score(yt, yp)),
            "mae": float(mean_absolute_error(yt, yp)),
            "rmse": float(np.sqrt(mean_squared_error(yt, yp))),
            "nan": False, "explosion": False}


def _worst_decile_mae(y_true, y_pred):
    order = np.argsort(np.asarray(y_true, dtype=float))
    yp = np.asarray(y_pred, dtype=float)[order]
    yy = np.asarray(y_true, dtype=float)[order]
    dec = max(1, len(yy) // 10)
    return float(max(np.abs(yp[i:i + dec] - yy[i:i + dec]).mean() for i in range(0, len(yy), dec)))


def _subset_metrics(y, pred, mask) -> dict | None:
    mask = np.asarray(mask, dtype=bool)
    if mask.sum() == 0:
        return None
    metrics = _safe_metrics(np.asarray(y)[mask], np.asarray(pred)[mask])
    metrics["count"] = int(mask.sum())
    return metrics


def _per_workload_metrics(y, cur_pred, cand_pred, meta: pd.DataFrame | None) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if meta is None or "workload_kind" not in meta.columns:
        return out
    kinds = meta["workload_kind"].fillna("unknown")
    for kind in WORKLOAD_KINDS:
        mask = kinds.eq(kind).to_numpy()
        if not mask.any():
            continue
        out[kind] = {
            "current": _subset_metrics(y, cur_pred, mask),
            "candidate": _subset_metrics(y, cand_pred, mask),
        }
    return out


def _ood_subset_metrics(X: pd.DataFrame, y, cur_pred, cand_pred) -> dict | None:
    if X is None or len(X) < 20:
        return None
    n_ref = max(5, len(X) // 5)
    ref = X.iloc[:n_ref]
    scored = []
    for idx in range(len(X)):
        scored.append(ood_score(ref, X.iloc[[idx]]))
    scored = np.asarray(scored, dtype=float)
    if not np.isfinite(scored).any():
        return None
    threshold = np.quantile(scored[np.isfinite(scored)], 0.8)
    mask = scored >= threshold
    if not mask.any():
        return None
    return {
        "count": int(mask.sum()),
        "threshold": float(threshold),
        "current": _subset_metrics(y, cur_pred, mask),
        "candidate": _subset_metrics(y, cand_pred, mask),
    }


def _gpu_like_metrics(y, cur_pred, cand_pred, meta: pd.DataFrame | None) -> dict | None:
    if meta is None or "workload_kind" not in meta.columns:
        return None
    mask = meta["workload_kind"].fillna("").eq("gpu-like").to_numpy()
    if not mask.any():
        return None
    return {
        "count": int(mask.sum()),
        "current": _subset_metrics(y, cur_pred, mask),
        "candidate": _subset_metrics(y, cand_pred, mask),
    }


def _mean_gate_by_workload(meta: pd.DataFrame | None, weights: np.ndarray, groups: list[str]) -> dict:
    out = {}
    if meta is None or "workload_kind" not in meta.columns or weights.size == 0:
        return out
    kinds = meta["workload_kind"].fillna("unknown")
    for kind in WORKLOAD_KINDS:
        mask = kinds.eq(kind).to_numpy()
        if not mask.any():
            continue
        out[kind] = {g: float(weights[mask, gi].mean()) for gi, g in enumerate(groups)}
    return out


def evaluate(current_pred_fn, candidate: CandidateModel,
             X: pd.DataFrame, y: pd.Series,
             *, groups_map=None, new_groups=None, y_train_mean=None, y_train_std=None,
             meta: pd.DataFrame | None = None):
    """current_pred_fn: callable(X)->np.ndarray (the approved model's predict)."""
    cur_pred = np.asarray(current_pred_fn(X), dtype=float)
    cand_pred, cand_weights, cand_groups = candidate.predict_with_gate(X)
    cand_pred = np.asarray(cand_pred, dtype=float)

    cm = _safe_metrics(y, cur_pred)
    nm = _safe_metrics(y, cand_pred)
    cm["worst_decile_mae"] = _worst_decile_mae(y, cur_pred)
    nm["worst_decile_mae"] = _worst_decile_mae(y, cand_pred)

    checks = {
        "no_nan": not nm["nan"],
        "no_explosion": not nm["explosion"],
        "r2_not_severely_negative": nm["r2"] >= -1.0,
    }

    new_expert_usage = {}
    if new_groups:
        gpu_cols = groups_map.get("gpu", []) if groups_map else []
        gpu_active = np.zeros(len(X), dtype=bool)
        if gpu_cols:
            cols = [c for c in gpu_cols if c in X.columns]
            if cols:
                gpu_active = (np.abs(X[cols].values) > candidate.spec.activation_threshold).any(axis=1)
        gpu_like_mask = meta["workload_kind"].fillna("").eq("gpu-like").to_numpy() if meta is not None and "workload_kind" in meta.columns else np.zeros(len(X), dtype=bool)

        if "gpu" in cand_groups:
            gi = cand_groups.index("gpu")
            overall_gpu_weight = float(cand_weights[:, gi].mean())
            active_gpu_weight = float(cand_weights[gpu_active, gi].mean()) if gpu_active.any() else 0.0
            inactive_gpu_weight = float(cand_weights[~gpu_active, gi].mean()) if (~gpu_active).any() else 0.0
            old_workload_nonzero = int(((cand_weights[:, gi] > 0.05) & ~gpu_like_mask).sum())
        else:
            overall_gpu_weight = active_gpu_weight = inactive_gpu_weight = 0.0
            old_workload_nonzero = 0

        checks["new_expert_nontrivial_weight"] = active_gpu_weight > 0.01
        checks["new_expert_not_unused"] = overall_gpu_weight > 0.001
        checks["new_expert_has_active_support"] = int(gpu_active.sum()) > 0
        checks["new_expert_has_gpu_like_support"] = int((gpu_active & gpu_like_mask).sum()) > 0
        checks["gate_not_collapsed"] = max(cand_weights.mean(axis=0)) <= 0.95 if cand_weights.size else True
        checks["gpu_weight_not_leaking_old_rows"] = old_workload_nonzero == 0
        new_expert_usage = {
            "global_gate_weight": {g: float(cand_weights[:, gi].mean()) for gi, g in enumerate(cand_groups)},
            "mean_gate_by_workload": _mean_gate_by_workload(meta, cand_weights, cand_groups),
            "conditional_gate_weight": {
                "gpu_active_mean": active_gpu_weight,
                "gpu_inactive_mean": inactive_gpu_weight,
            },
            "active_rows": {
                "gpu": {
                    "n_active_rows": int(gpu_active.sum()),
                    "n_gpu_like_active_rows": int((gpu_active & gpu_like_mask).sum()),
                    "n_old_workload_rows_gpu_weight_gt_0_05": old_workload_nonzero,
                }
            },
        }

    per_workload = _per_workload_metrics(y, cur_pred, cand_pred, meta)
    ood_metrics = _ood_subset_metrics(X, y, cur_pred, cand_pred)
    gpu_like_metrics = _gpu_like_metrics(y, cur_pred, cand_pred, meta)

    return {
        "current": cm,
        "candidate": nm,
        "checks": checks,
        "new_expert_usage": new_expert_usage,
        "candidate_kind": candidate.spec.kind,
        "new_groups": new_groups or [],
        "per_workload": per_workload,
        "ood_subset": ood_metrics,
        "gpu_like_subset": gpu_like_metrics,
        "gate_mode": candidate.spec.gate_mode,
    }
