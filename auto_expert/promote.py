"""Promote (steps 9/10/11): auto approve / reject / rollback, with logging.

Promotion is conservative: candidate must clear all safety checks AND beat the
current approved model by a margin (MAE -5% or R2 +0.02), without worsening RMSE
or worst-case MAE. Otherwise -> rejected, with a recorded reason.

Rollback: after a promotion, if a later evaluation shows the new approved model
underperforming the previous one by >10% MAE, roll back to the previous and mark
the failed one for analysis (not deleted).
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

APPROVED_DIR = Path("models/approved")
LATEST_FILE = Path("models/latest_approved.json")
UPDATE_LOG = Path("results/online_learning_update_log.jsonl")
EXPANSION_LOG = Path("results/auto_expert_expansion_log.jsonl")


def _json_safe(obj):
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, tuple):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _log(path: Path, rec: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as fh:
        fh.write(json.dumps(_json_safe(rec)) + "\n")


def _margin_passes(cur, cand, *, mae_improve=0.05, r2_improve=0.02):
    """True if candidate beats current by the required margin on at least one
    primary metric, without regressing the others."""
    if cand["r2"] >= cur["r2"] + r2_improve and cand["rmse"] <= cur["rmse"] * 1.001 \
       and cand["worst_decile_mae"] <= cur["worst_decile_mae"] * 1.05:
        return True, "r2_improved"
    if cand["mae"] <= cur["mae"] * (1 - mae_improve) and cand["rmse"] <= cur["rmse"] * 1.001 \
       and cand["worst_decile_mae"] <= cur["worst_decile_mae"] * 1.05:
        return True, "mae_improved"
    return False, "no_margin"


def decide(ev: dict, candidate_version: str, prev_version: str | None, *, trigger_evidence: dict | None = None) -> dict:
    """Pure decision function. Returns the verdict record (not yet applied).

    Special case: if there is no approved model yet (prev_version is None / current
    metrics are NaN), a candidate that clears the safety checks becomes the FIRST
    approved baseline (no margin required — there's nothing to beat). This seeds
    the safe loop; every later candidate must still beat it by margin.
    """
    checks = ev["checks"]
    failed = [k for k, v in checks.items() if not v]
    base = {
        "current_metrics": ev.get("current"),
        "candidate_metrics": ev.get("candidate"),
        "per_workload": ev.get("per_workload", {}),
        "ood_subset": ev.get("ood_subset"),
        "gpu_like_subset": ev.get("gpu_like_subset"),
        "checks": ev.get("checks", {}),
        "new_expert_usage": ev.get("new_expert_usage", {}),
        "trigger_evidence": trigger_evidence or {},
    }
    if failed:
        return {"status": "rejected", "candidate_version": candidate_version,
                "prev_version": prev_version, "reason": "safety_check_failed",
                "failed_checks": failed, **base}
    cur, cand = ev["current"], ev["candidate"]
    no_baseline = prev_version is None or np.isnan(cur.get("r2", float("nan")))
    if no_baseline:
        return {"status": "approved", "candidate_version": candidate_version,
                "prev_version": prev_version, "reason": "first_baseline",
                "cur_r2": cur["r2"], "cand_r2": cand["r2"],
                "cur_mae": cur["mae"], "cand_mae": cand["mae"], **base}

    if ev.get("candidate_kind") == "expanded" and "gpu" in ev.get("new_groups", []):
        gpu_subset = ev.get("gpu_like_subset")
        if not gpu_subset or not gpu_subset.get("current") or not gpu_subset.get("candidate"):
            return {"status": "rejected", "candidate_version": candidate_version,
                    "prev_version": prev_version, "reason": "missing_gpu_like_subset",
                    "cur_r2": cur["r2"], "cand_r2": cand["r2"],
                    "cur_mae": cur["mae"], "cand_mae": cand["mae"], **base}
        if gpu_subset["candidate"]["mae"] > gpu_subset["current"]["mae"] * 0.95:
            return {"status": "rejected", "candidate_version": candidate_version,
                    "prev_version": prev_version, "reason": "gpu_subset_not_improved",
                    "cur_r2": cur["r2"], "cand_r2": cand["r2"],
                    "cur_mae": cur["mae"], "cand_mae": cand["mae"], **base}
        if cand["rmse"] > cur["rmse"] * 1.001:
            return {"status": "rejected", "candidate_version": candidate_version,
                    "prev_version": prev_version, "reason": "overall_rmse_worse",
                    "cur_r2": cur["r2"], "cand_r2": cand["r2"],
                    "cur_mae": cur["mae"], "cand_mae": cand["mae"], **base}
        per_workload = ev.get("per_workload", {})
        old_workloads_bad = []
        for kind in ["cpu", "mem", "io", "net", "mixed"]:
            wk = per_workload.get(kind)
            if not wk or not wk.get("current") or not wk.get("candidate"):
                continue
            if wk["candidate"]["mae"] > wk["current"]["mae"] * 1.05:
                old_workloads_bad.append(kind)
        if old_workloads_bad:
            return {"status": "rejected", "candidate_version": candidate_version,
                    "prev_version": prev_version, "reason": "old_workloads_degraded",
                    "degraded_workloads": old_workloads_bad,
                    "cur_r2": cur["r2"], "cand_r2": cand["r2"],
                    "cur_mae": cur["mae"], "cand_mae": cand["mae"], **base}
        gpu_usage = ev.get("new_expert_usage", {}).get("conditional_gate_weight", {}).get("gpu_active_mean")
        if gpu_usage is None:
            gpu_usage = ev.get("new_expert_usage", {}).get("global_gate_weight", {}).get("gpu", 0.0)
        gpu_support = ev.get("new_expert_usage", {}).get("active_rows", {}).get("gpu", {})
        if gpu_usage <= 0.01 or gpu_support.get("n_gpu_like_active_rows", 0) <= 0:
            return {"status": "rejected", "candidate_version": candidate_version,
                    "prev_version": prev_version, "reason": "gpu_expert_unused",
                    "cur_r2": cur["r2"], "cand_r2": cand["r2"],
                    "cur_mae": cur["mae"], "cand_mae": cand["mae"], **base}
        if max(ev.get("new_expert_usage", {}).get("global_gate_weight", {}).values() or [0.0]) > 0.95:
            return {"status": "rejected", "candidate_version": candidate_version,
                    "prev_version": prev_version, "reason": "gate_collapse",
                    "cur_r2": cur["r2"], "cand_r2": cand["r2"],
                    "cur_mae": cur["mae"], "cand_mae": cand["mae"], **base}
        return {"status": "approved", "candidate_version": candidate_version,
                "prev_version": prev_version, "reason": "gpu_expansion_improved",
                "cur_r2": cur["r2"], "cand_r2": cand["r2"],
                "cur_mae": cur["mae"], "cand_mae": cand["mae"], **base}

    ok, how = _margin_passes(cur, cand)
    if not ok:
        return {"status": "rejected", "candidate_version": candidate_version,
                "prev_version": prev_version, "reason": how,
                "cur_r2": cur["r2"], "cand_r2": cand["r2"],
                "cur_mae": cur["mae"], "cand_mae": cand["mae"], **base}
    return {"status": "approved", "candidate_version": candidate_version,
            "prev_version": prev_version, "reason": how,
            "cur_r2": cur["r2"], "cand_r2": cand["r2"],
            "cur_mae": cur["mae"], "cand_mae": cand["mae"], **base}


def apply_promotion(verdict: dict, candidate_model_path: Path):
    """Persist an approved candidate as the latest approved model.

    Copies the candidate pkl into models/approved/<version>/ and writes
    latest_approved.json pointing at it. The previous approved model is NOT
    deleted (kept for rollback).
    """
    if verdict["status"] != "approved":
        _log(UPDATE_LOG, verdict)
        return verdict
    v = verdict["candidate_version"]
    dest_dir = APPROVED_DIR / v
    dest_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_model_path, dest_dir / "model.pkl")
    spec_path = candidate_model_path.parent / "spec.json"
    if spec_path.exists():
        shutil.copy2(spec_path, dest_dir / "spec.json")
    latest = {"approved_version": v, "prev_version": verdict.get("prev_version"),
              "reason": verdict["reason"], "model_path": str(dest_dir / "model.pkl")}
    LATEST_FILE.parent.mkdir(parents=True, exist_ok=True)
    LATEST_FILE.write_text(json.dumps(latest, indent=2))
    _log(UPDATE_LOG, verdict)
    return verdict


def rollback(current_version: str, prev_version: str | None, reason: str) -> dict:
    """Roll back to prev_version. The failed current_version is kept on disk."""
    rec = {"status": "rolled_back", "failed_version": current_version,
           "rolled_back_to": prev_version, "reason": reason}
    if prev_version:
        prev_path = APPROVED_DIR / prev_version / "model.pkl"
        if prev_path.exists():
            latest = {"approved_version": prev_version,
                      "prev_version": None, "reason": f"rollback from {current_version}: {reason}",
                      "model_path": str(prev_path)}
            LATEST_FILE.write_text(json.dumps(latest, indent=2))
    _log(UPDATE_LOG, rec)
    return rec


def log_expansion(rec: dict):
    _log(EXPANSION_LOG, rec)
