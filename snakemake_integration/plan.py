"""Planner — turns a workflow into per-job offloading assignments (Objective 3, live).

This is the bridge between the MoE energy predictor and a real Snakemake run. It:
  1. loads the workloads and trains the MoE + single predictors,
  2. segments the intervals into a job workflow,
  3. predicts each job's energy with the MoE,
  4. calls the SAME ``select_offloaded`` policy the simulator uses to assign each
     job to the primary or secondary cluster,
  5. writes a ``plan.json`` that the Snakefile consumes — one entry per job with
     its assigned cluster, predicted energy, and the modelled runtime/energy on
     that cluster.

The Snakefile then materialises one rule per job and "executes" it on its assigned
cluster. Decision logic lives here and in ``offloading.decision`` only; the
Snakefile is pure orchestration.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from moe import data
from moe.moe import MixtureOfExperts, MoEConfig, SingleModel
from offloading.clusters import default_clusters
from offloading.decision import select_offloaded
from offloading.workflow import build_workflow, predict_job_energy


def build_plan(expert: str, jobs_per_workload: int, capacity_frac: float,
               strategy: str, seed: int) -> dict:
    datasets = data.load_all()
    X, y, w = data.combined_frame(datasets)
    cfg = MoEConfig(expert=expert)
    moe = MixtureOfExperts(cfg).fit(X, y, w)
    single = SingleModel(cfg).fit(X, y)

    jobs = build_workflow(datasets, jobs_per_workload=jobs_per_workload, seed=seed)
    pred_moe = predict_job_energy(jobs, moe.predict)
    pred_single = predict_job_energy(jobs, single.predict)
    pred = {"moe": pred_moe, "single": pred_single}.get(strategy, pred_moe)

    primary, secondary = default_clusters()
    offloaded = select_offloaded(jobs, pred, secondary, strategy, capacity_frac, seed)

    clusters = {
        "primary": primary.__dict__,
        "secondary": secondary.__dict__,
    }
    plan_jobs = []
    for i, job in enumerate(jobs):
        cl = secondary if offloaded[i] else primary
        plan_jobs.append({
            "job_id": int(job.job_id),
            "workload": job.workload,
            "cluster": cl.name,
            "pred_energy_wh": float(pred[i]),
            "true_energy_wh": float(job.true_energy_wh),
            "primary_runtime_s": float(job.runtime_s),
            "exec_energy_wh": float(cl.energy_of(job.true_energy_wh)),
            "exec_runtime_s": float(cl.runtime_of(job.runtime_s)),
            "exec_cost": float(cl.cost_of(job.true_energy_wh)),
        })

    return {
        "strategy": strategy,
        "expert": expert,
        "capacity_frac": capacity_frac,
        "clusters": clusters,
        "n_jobs": len(plan_jobs),
        "n_offloaded": int(offloaded.sum()),
        "jobs": plan_jobs,
    }


def main():
    ap = argparse.ArgumentParser(description="Build offloading plan for Snakemake")
    ap.add_argument("--expert", choices=["rf", "linear"], default="linear")
    ap.add_argument("--strategy", choices=["moe", "single", "oracle", "random", "all_primary"],
                    default="moe")
    ap.add_argument("--jobs-per-workload", type=int, default=10)
    ap.add_argument("--capacity-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="plan.json")
    args = ap.parse_args()

    plan = build_plan(args.expert, args.jobs_per_workload, args.capacity_frac,
                      args.strategy, args.seed)
    Path(args.out).write_text(json.dumps(plan, indent=2))
    print(f"Wrote {args.out}: {plan['n_jobs']} jobs, "
          f"{plan['n_offloaded']} offloaded to secondary "
          f"(strategy={args.strategy}, expert={args.expert})")


if __name__ == "__main__":
    main()
