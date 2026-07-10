"""Pipeline (step 13): orchestrate the safe online-learning closed loop.

detect  -> (if new dimension) build expanded candidate + a plain retrain candidate
        -> train candidates
        -> evaluate vs current approved model
        -> auto approve / reject / rollback
        -> log every decision

This is the single entry point for "automatic model update". It never edits the
live model directly except via promote.apply_promotion, and only after safety
checks + margin pass. Insufficient data -> pending_data, no training.

Usage:
    PYTHONPATH=. python -m auto_expert.pipeline            # one cycle
    PYTHONPATH=. python -m auto_expert.pipeline --demo      # inject simulated gpu data
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import buffer as buf
from . import discovery as disc
from . import expand as exp
from . import promote as prom
from .evaluate_candidate import evaluate
from .train_candidate import train_candidate, save_candidate

MIN_LABELS_FOR_TRAIN = 100
PENDING_DIR = Path("models/pending")


def _current_approved_predictor():
    """Load the latest approved model (or None if none approved yet)."""
    if not prom.LATEST_FILE.exists():
        return None, None
    latest = json.loads(prom.LATEST_FILE.read_text())
    mp = Path(latest["model_path"])
    if not mp.exists():
        return None, latest["approved_version"]
    import pickle
    with mp.open("rb") as fh:
        m = pickle.load(fh)
    return m, latest["approved_version"]


def _current_predict_fn(model):
    """Wrap an approved CandidateModel (or fallback constant) as predict(X)->array."""
    if model is None:
        return lambda X: np.full(len(X), np.nan)  # no baseline -> can't compare margin
    return lambda X: model.predict(X)


def run_cycle(*, demo: bool = False, buffer_path: str | Path = buf.DEFAULT_BUFFER,
              min_labels: int = MIN_LABELS_FOR_TRAIN) -> dict:
    """One automatic-update cycle. Returns a summary of what happened."""
    cycle_id = f"cycle-{int(time.time())}"
    loaded = buf.load_buffer(buffer_path)
    if loaded is None:
        b = buf.build_buffer(buffer_path=buffer_path)
        if b["status"] != "ok":
            rec = {"cycle_id": cycle_id, "status": "pending_data",
                   "n_samples": b.get("n", 0), "min_required": min_labels}
            prom.log_expansion(rec)
            print(json.dumps(rec, indent=2))
            return rec
        loaded = buf.load_buffer(buffer_path)

    X, y, meta = loaded
    b = {"status": "ok", "n": len(y), "buffer_path": str(buffer_path)}
    if b["n"] < min_labels:
        rec = {"cycle_id": cycle_id, "status": "pending_data",
               "n_samples": b.get("n", 0), "min_required": min_labels}
        prom.log_expansion(rec)
        print(json.dumps(rec, indent=2))
        return rec

    # demo mode: inject a synthetic gpu resource dimension so the loop exercises
    # expanded-candidate creation end to end.
    new_groups_for_demo = []
    if demo:
        rng = np.random.RandomState(0)
        X = X.copy()
        X["delta_gpu_utilization"] = rng.rand(len(X)) * 60 + 10
        X["delta_gpu_memory"] = rng.rand(len(X)) * 1e6
        # make gpu activity correlate with energy so a gpu expert could help
        y = y + 0.0 * X["delta_gpu_utilization"]  # (no real signal; demo of plumbing)

    # 2) current approved model (for comparison)
    cur_model, cur_version = _current_approved_predictor()
    cur_fn = _current_predict_fn(cur_model)

    # 3) discovery + trigger classification
    known = disc.known_features_from_columns(X.columns)
    det = disc.detect_new_resource(X, known=known)
    trigger_status = "retrain_only"
    trigger_reason = "ordinary_retrain"
    trigger_evidence = {
        "discovery_status": det.get("status"),
        "gpu_features_present": [c for c in ["delta_gpu_utilization", "delta_gpu_memory", "delta_gpu_power", "gpu_like_signal"] if c in X.columns],
        "gpu_like_subset_count": int(meta["workload_kind"].fillna("").eq("gpu-like").sum()) if "workload_kind" in meta.columns else 0,
    }
    if det["status"] == "new_dimension":
        gpu_props = [p for p in det.get("proposals", []) if p.get("group") == "gpu"]
        gpu_like_mask = meta["workload_kind"].fillna("").eq("gpu-like").to_numpy() if "workload_kind" in meta.columns else np.zeros(len(meta), dtype=bool)
        cur_pred_for_trigger = np.asarray(cur_fn(X), dtype=float)
        overall_mae = float(np.mean(np.abs(cur_pred_for_trigger - y.values))) if np.isfinite(cur_pred_for_trigger).all() else float("inf")
        gpu_like_mae = float(np.mean(np.abs(cur_pred_for_trigger[gpu_like_mask] - y.values[gpu_like_mask]))) if gpu_like_mask.any() and np.isfinite(cur_pred_for_trigger[gpu_like_mask]).all() else float("inf")
        trigger_evidence.update({
            "gpu_proposals": gpu_props,
            "overall_mae": overall_mae,
            "gpu_like_mae": gpu_like_mae,
            "base_model_version": cur_version,
        })
        if gpu_props and gpu_like_mask.any() and gpu_like_mae > overall_mae * 1.2:
            trigger_status = "retrain_and_expand"
            trigger_reason = "gpu_like_features_persistent_and_gpu_subset_error_high"
        else:
            trigger_reason = "gpu_features_detected_but_expand_threshold_not_met"
    elif det["status"] == "pending_data":
        trigger_status = "pending_data"
        trigger_reason = "gpu_detection_insufficient_data"
        prom.log_expansion({"cycle_id": cycle_id, "status": "pending_data",
                            "stage": "discovery", "detail": det})
    print(f"[{cycle_id}] buffer={b['n']} samples, discovery={det['status']} trigger={trigger_status}")

    # 4) build candidate specs
    specs = []
    version = f"cand-{int(time.time())}"
    specs.append(exp.make_retrain_candidate(
        version + "-retrain",
        list(X.columns),
        base_model_version=cur_version,
        trigger_reason=trigger_reason,
        trigger_evidence=trigger_evidence,
    ))
    if trigger_status == "retrain_and_expand":
        gpu_props = [p for p in det.get("proposals", []) if p.get("group") == "gpu"]
        specs.append(exp.make_expanded_candidate(
            version + "-expanded",
            gpu_props,
            list(X.columns),
            base_model_version=cur_version,
            trigger_reason=trigger_reason,
            trigger_evidence=trigger_evidence,
        ))
        specs[-1].gate_mode = "learned_per_sample"
        new_groups_for_demo = [p["group"] for p in gpu_props]

    # 5) train + evaluate each candidate SEQUENTIALLY: if an earlier candidate
    # is promoted, later ones must beat the freshly-approved model (not the
    # stale baseline). This makes "retrain seeds baseline, expanded must improve"
    # work in one cycle.
    results = []
    for spec in specs:
        cand = train_candidate(spec, X, y)
        cand_path = PENDING_DIR / spec.candidate_version / "model.pkl"
        save_candidate(cand, cand_path)
        new_groups = spec.new_groups if spec.kind == "expanded" else None
        ev = evaluate(cur_fn, cand, X, y,
                      groups_map=spec.groups, new_groups=new_groups, meta=meta)
        verdict = prom.decide(ev, spec.candidate_version, cur_version, trigger_evidence=spec.trigger_evidence)
        prom.apply_promotion(verdict, cand_path)   # logs approval OR rejection
        if verdict["status"] == "approved":
            # refresh the comparison baseline to the just-approved candidate
            cur_model, cur_version = cand, spec.candidate_version
            cur_fn = _current_predict_fn(cur_model)
        results.append({"candidate": spec.candidate_version, "kind": spec.kind,
                        "verdict": verdict, "eval": {
                            "current": ev["current"],
                            "candidate": ev["candidate"],
                            "checks": ev["checks"],
                            "new_expert_usage": ev["new_expert_usage"],
                            "per_workload": ev["per_workload"],
                            "ood_subset": ev["ood_subset"],
                            "gpu_like_subset": ev["gpu_like_subset"],
                        }})
        print(f"  candidate {spec.candidate_version} ({spec.kind}): {verdict['status']} ({verdict['reason']})")

    summary = {"cycle_id": cycle_id, "status": "evaluated", "n_samples": b["n"],
               "discovery": det["status"], "trigger_status": trigger_status,
               "trigger_reason": trigger_reason, "trigger_evidence": trigger_evidence,
               "results": results}
    prom.log_expansion(summary)
    return summary


def main():
    ap = argparse.ArgumentParser(description="Safe online-learning auto-update cycle")
    ap.add_argument("--demo", action="store_true",
                    help="inject a synthetic gpu resource dimension to exercise expansion")
    args = ap.parse_args()
    run_cycle(demo=args.demo)


if __name__ == "__main__":
    main()
