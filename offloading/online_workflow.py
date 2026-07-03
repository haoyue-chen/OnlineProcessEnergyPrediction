"""Online learning feedback loop — automatic /predict + /update per job.

This is the closed loop the live online API was built for, with NO manual curl:

    for each job in the workflow:
        POST /predict  (before dispatch)   -> prediction_id, predicted energy, expert
        store prediction_id alongside job_id
        run the job                        -> observe true_energy_wh
        POST /update   (after finish)      -> model learns incrementally
    print a job-by-job record + summary

The "job" comes from the same real workload segmentation used elsewhere
(`offloading.workflow.build_workflow`): each job's true_energy_wh is the *measured*
node energy of its intervals — i.e. ground truth from collected data, used here to
simulate the energy a real job runner / cluster monitor would report on completion.

The driver is a thin HTTP client (stdlib urllib) against a running `serve-online`
service, so it exercises the deployed API exactly as an external job runner would.

Usage:
    python -m offloading.online_workflow --url http://localhost:8800 \
        --jobs-per-workload 5 --out results/online_workflow/run.json
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import numpy as np

from moe import data
from offloading.workflow import build_workflow


def _post(url: str, path: str, body: dict, timeout: float = 30.0) -> dict:
    req = urllib.request.Request(
        url.rstrip("/") + path,
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def _get(url: str, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def job_features(job) -> dict:
    """One representative feature row per job = mean over its intervals.

    The online expert is trained on interval-level rows, so a job's mean-interval
    feature vector stays in the trained domain (summing would not).
    """
    return {c: float(v) for c, v in job.interval_features.mean().items()}


def job_true_energy_wh(job) -> float:
    """Per-interval mean energy — matches the mean-feature row scale used above."""
    return float(job.true_energy_wh / max(job.n_intervals, 1))


def run(url: str, jobs_per_workload: int, seed: int) -> dict:
    datasets = data.load_all()
    jobs = build_workflow(datasets, jobs_per_workload=jobs_per_workload, seed=seed)

    info0 = _get(url, "/info")
    records = []
    n_predict = n_update = 0
    for job in jobs:
        feats = job_features(job)

        # 1) predict BEFORE dispatch, remember prediction_id with the job
        pred = _post(url, "/predict", {"features": feats})
        n_predict += 1
        ver_before = pred["model_version"]

        # 2) "run" the job — here the true energy comes from measured data
        true_wh = job_true_energy_wh(job)

        # 3) update AFTER finish, using the stored prediction_id
        upd = _post(url, "/update", {
            "prediction_id": pred["prediction_id"],
            "true_energy_wh": true_wh,
        })
        n_update += 1

        records.append({
            "job_id": job.job_id,
            "workload": job.workload,
            "prediction_id": pred["prediction_id"],
            "expert": pred["expert"],
            "predicted_energy_wh": round(pred["energy_wh"], 4),
            "true_energy_wh": round(true_wh, 4),
            "abs_error_wh": round(abs(pred["energy_wh"] - true_wh), 4),
            "model_version_before": ver_before,
            "model_version_after": upd["model_version"],
            "update_success": upd["model_version"] > ver_before,
        })

    info1 = _get(url, "/info")
    errs = np.array([r["abs_error_wh"] for r in records], dtype=float)
    summary = {
        "url": url,
        "n_jobs": len(jobs),
        "n_predict_calls": n_predict,
        "n_update_calls": n_update,
        "all_updates_succeeded": all(r["update_success"] for r in records),
        "num_updates_before": info0["num_updates"],
        "num_updates_after": info1["num_updates"],
        "model_version_before": info0["model_version"],
        "model_version_after": info1["model_version"],
        "mean_abs_error_wh": round(float(errs.mean()), 4) if len(errs) else None,
        # error trend: did online learning reduce error over the run?
        "mae_first_half": round(float(errs[:len(errs)//2].mean()), 4) if len(errs) > 1 else None,
        "mae_second_half": round(float(errs[len(errs)//2:].mean()), 4) if len(errs) > 1 else None,
        "records": records,
    }
    return summary


def main():
    ap = argparse.ArgumentParser(description="Online learning feedback loop (auto /predict + /update)")
    ap.add_argument("--url", default="http://localhost:8800", help="serve-online base URL")
    ap.add_argument("--jobs-per-workload", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wait", type=float, default=0.0,
                    help="Seconds to wait for the service /health before starting.")
    ap.add_argument("--out", default=None, help="Optional path to save the run JSON.")
    args = ap.parse_args()

    if args.wait > 0:
        deadline = time.time() + args.wait
        while time.time() < deadline:
            try:
                if _get(args.url, "/health").get("status") == "ok":
                    break
            except Exception:
                time.sleep(1)

    summary = run(args.url, args.jobs_per_workload, args.seed)

    print("\n── Online workflow feedback loop ─────────────────────────────")
    print(f"  jobs                : {summary['n_jobs']}")
    print(f"  /predict calls      : {summary['n_predict_calls']}")
    print(f"  /update calls       : {summary['n_update_calls']}")
    print(f"  all updates ok      : {summary['all_updates_succeeded']}")
    print(f"  num_updates         : {summary['num_updates_before']} -> {summary['num_updates_after']}")
    print(f"  model_version       : {summary['model_version_before']} -> {summary['model_version_after']}")
    print(f"  mean abs error (Wh) : {summary['mean_abs_error_wh']}")
    print(f"  MAE first/second half: {summary['mae_first_half']} / {summary['mae_second_half']}")
    print("──────────────────────────────────────────────────────────────")
    print(f"{'job':>4} {'workload':>9} {'expert':>9} {'pred':>9} {'true':>9} "
          f"{'verB':>5} {'verA':>5} {'ok':>3}")
    for r in summary["records"][:12]:
        print(f"{r['job_id']:>4} {r['workload']:>9} {r['expert']:>9} "
              f"{r['predicted_energy_wh']:>9.2f} {r['true_energy_wh']:>9.2f} "
              f"{r['model_version_before']:>5} {r['model_version_after']:>5} "
              f"{'Y' if r['update_success'] else 'N':>3}")
    if len(summary["records"]) > 12:
        print(f"  … {len(summary['records']) - 12} more")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
