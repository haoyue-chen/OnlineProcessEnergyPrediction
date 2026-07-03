# Live Snakemake Offloading Integration (Objective 3)

This wires the MoE energy predictor into a **real Snakemake workflow**. Where the
`offloading/` package *simulates* the decision in one process, this runs an actual
Snakemake DAG: a planning checkpoint, one executed job rule per workflow job, and
an aggregation rule — the shape a real SWMS plugin would take.

> **Honest scope.** Both "clusters" run on this one machine (two execution
> channels), and each job executes a tiny scaled busy-loop rather than the real
> ~15 h workload, so the DAG finishes in seconds. The decision logic, the plan,
> and the energy/cost/makespan accounting are real and identical to the simulator;
> what is *not* real is a second physical cluster. Pointing `run_job.py` at a real
> primary scheduler / secondary (SSH, `kubectl`, cloud API) is the only change
> needed for a true multi-cluster deployment — the DAG and decision layer stay.

## DAG

```text
checkpoint plan   ─► plan.json   (train MoE, predict per-job energy,
      │                           assign each job to primary/secondary)
      ▼
rule run_job × N  ─► records/job_<id>.json   (execute job on its cluster)
      ▼
rule aggregate    ─► summary.json   (realised energy / cost / makespan)
```

Decision logic lives only in `offloading.decision.select_offloaded` and
`snakemake_integration/plan.py`; the `Snakefile` is pure orchestration.

## Files

| File | Role |
|---|---|
| `plan.py` | Train MoE + assign jobs to clusters → `plan.json`. |
| `Snakefile` | The DAG: `plan` checkpoint → `run_job` per job → `aggregate`. |
| `run_job.py` | Execute one job on its assigned cluster, emit its record. |
| `aggregate.py` | Collect records → workflow `summary.json`. |
| `run.sh` | Driver: single strategy or `compare` across all strategies. |

## Run

```sh
cd ProcessEnergyAccounting/modeling/estimation

# single run (strategy=moe, expert=linear)
./snakemake_integration/run.sh

# compare strategies through the real workflow
./snakemake_integration/run.sh compare

# or call snakemake directly
OFFLOAD_PY=/home/hujiao/MPDS/.venv/bin/python OFFLOAD_SRC=$(pwd) \
  /home/hujiao/MPDS/.venv/bin/snakemake -s snakemake_integration/Snakefile \
  --cores 4 -d snakemake_integration/.run \
  --config expert=linear strategy=moe jobs_per_workload=10 capacity_frac=0.15
```

## Result (live workflow, 40 jobs, capacity 15%, linear expert)

```
all_primary  offloaded=  0/40  energy= 8380.15 kWh  saved=  0.0%  makespan=54242s
random       offloaded=  7/40  energy= 7957.13 kWh  saved=  5.0%  makespan=45522s
single       offloaded=  7/40  energy= 7876.87 kWh  saved=  6.0%  makespan=44878s
moe          offloaded=  7/40  energy= 7877.61 kWh  saved=  6.0%  makespan=44878s
oracle       offloaded=  7/40  energy= 7876.87 kWh  saved=  6.0%  makespan=44878s
```

The live numbers match the in-process simulator: MoE-guided offloading beats
`random` and tracks the `oracle` ceiling. (At this job granularity MoE and single
pick the same 7 jobs; the gap widens at tighter budgets / finer granularity, as in
the `offloading/` sweep.)

Run artifacts (`.run*/`, `summary_*.json`, `plan.json`) are gitignored.
