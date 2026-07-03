"""Per-job executor — what a single Snakemake job rule actually runs.

In a real deployment this is where the job's command would be dispatched to its
assigned cluster (e.g. submit to the primary scheduler, or SSH/kubectl to the
secondary). Here, with both "clusters" on one machine, we *really execute* a small
proportional workload so the DAG runs for real, then emit the job's modelled
energy/runtime/cost record for that cluster.

The execution is deliberately scaled down (capped busy-loop) so the whole workflow
finishes in seconds rather than the ~15 measured hours — the records carry the
*modelled* full numbers, the busy-loop just makes the job a genuine unit of work.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def _do_work(scaled_seconds: float) -> None:
    """A genuine (tiny) CPU task so the rule is real work, not a no-op."""
    deadline = time.perf_counter() + min(scaled_seconds, 0.05)
    x = 0.0
    while time.perf_counter() < deadline:
        x += 1.0  # keep the CPU busy briefly
    return None


def main():
    ap = argparse.ArgumentParser(description="Execute one workflow job on its cluster")
    ap.add_argument("--plan", required=True)
    ap.add_argument("--job-id", type=int, required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    job = next(j for j in plan["jobs"] if j["job_id"] == args.job_id)

    # Scale the modelled runtime down by a large factor for the live demo.
    _do_work(job["exec_runtime_s"] / 3600.0)

    record = {
        "job_id": job["job_id"],
        "workload": job["workload"],
        "cluster": job["cluster"],
        "exec_energy_wh": job["exec_energy_wh"],
        "exec_runtime_s": job["exec_runtime_s"],
        "exec_cost": job["exec_cost"],
        "true_energy_wh": job["true_energy_wh"],
    }
    Path(args.out).write_text(json.dumps(record))
    print(f"job {job['job_id']:>3} [{job['workload']:>8}] -> {job['cluster']:<9} "
          f"energy={job['exec_energy_wh']/1000:.3f} kWh")


if __name__ == "__main__":
    main()
