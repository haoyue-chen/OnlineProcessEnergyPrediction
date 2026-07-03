"""Aggregator — collect per-job records into a workflow-level summary.

Computes the realised totals from the *executed* jobs: total energy, cost, and the
makespan (max of the two clusters' busy time, since primary and secondary run in
parallel). Compares against the all-primary baseline to report energy saved.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description="Aggregate executed job records")
    ap.add_argument("--records", nargs="+", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    plan = json.loads(Path(args.plan).read_text())
    records = [json.loads(Path(r).read_text()) for r in args.records]

    total_energy = sum(r["exec_energy_wh"] for r in records)
    total_cost = sum(r["exec_cost"] for r in records)
    primary_busy = sum(r["exec_runtime_s"] for r in records if r["cluster"] == "primary")
    secondary_busy = sum(r["exec_runtime_s"] for r in records if r["cluster"] == "secondary")
    makespan = max(primary_busy, secondary_busy)

    baseline_energy = sum(r["true_energy_wh"] for r in records)  # all on primary
    saved = baseline_energy - total_energy

    summary = {
        "strategy": plan["strategy"],
        "expert": plan["expert"],
        "n_jobs": len(records),
        "n_offloaded": sum(1 for r in records if r["cluster"] == "secondary"),
        "total_energy_kwh": total_energy / 1000.0,
        "baseline_energy_kwh": baseline_energy / 1000.0,
        "energy_saved_kwh": saved / 1000.0,
        "energy_saved_pct": 100.0 * saved / baseline_energy if baseline_energy else 0.0,
        "total_cost": total_cost,
        "makespan_s": makespan,
        "primary_busy_s": primary_busy,
        "secondary_busy_s": secondary_busy,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))

    print("\n── Snakemake live offloading run — summary ───────────────────")
    print(f"  strategy / expert     : {summary['strategy']} / {summary['expert']}")
    print(f"  jobs (offloaded)      : {summary['n_jobs']} ({summary['n_offloaded']})")
    print(f"  energy                : {summary['total_energy_kwh']:.2f} kWh "
          f"(baseline {summary['baseline_energy_kwh']:.2f})")
    print(f"  energy saved          : {summary['energy_saved_kwh']:.2f} kWh "
          f"({summary['energy_saved_pct']:.1f}%)")
    print(f"  cost                  : {summary['total_cost']:.1f}")
    print(f"  makespan              : {summary['makespan_s']:.0f} s "
          f"(primary {primary_busy:.0f}, secondary {secondary_busy:.0f})")
    print("──────────────────────────────────────────────────────────────")


if __name__ == "__main__":
    main()
