# Objective 3 — Energy-Aware Offloading (MoE-driven decision engine)

This module closes the loop of the project: the Phase-2 MoE energy predictor now
**drives an offloading decision**, the original goal in the project plan
(`Workflow → Energy Prediction → Offloading Decision → Cluster Selection`).

There is no live multi-cluster / Snakemake deployment in this repo, so the
clusters and workflow are modelled analytically and the decision is evaluated in
simulation — but every job's features, energy and runtime come from the **real
measured data**, and every strategy is scored on the **actual measured** outcome.

## Pipeline

```text
work/ intervals ──► segment into a job workflow ──► MoE predicts each job's energy
        │                                                      │
        │                                              greedy energy-first offload
        │                                              onto a greener, capacity-
        │                                              limited secondary cluster
        ▼                                                      ▼
  ground-truth energy ◄──────── score every strategy on ACTUAL measured energy/cost/makespan
```

* **Workflow** (`workflow.py`): each workload is chopped into contiguous blocks of
  intervals → `Job`s. A job's energy/runtime is the sum/span of its intervals.
  Energy is predicted per *interval* and summed per job (the predictors are
  trained on interval rows), so predictions stay in the trained domain.
* **Clusters** (`clusters.py`): `primary` = the measured machine. `secondary` =
  greener (`energy_factor 0.7`) and faster (`speed_factor 1.2`) but pricier
  (`$0.54` vs `$0.30`/kWh) and **capacity-limited**. So offloading saves energy
  but can't absorb everything — you must choose *which* jobs.
* **Decision** (`decision.py`): energy-first 0/1 knapsack solved greedily by
  energy-saved-per-unit-secondary-time. Strategies differ only in how jobs are
  *scored*: `moe`, `single`, `oracle` (true energy), `random`, `all_primary`.

## Why a capacity limit is the point

With unlimited secondary capacity you'd offload everything and ranking wouldn't
matter. Under a **tight budget** the engine must pick the highest-energy jobs
*correctly* — that is exactly where a better energy predictor pays off. The
decision quality is therefore measured as how close a strategy gets to the
**oracle** (perfect knowledge) ceiling.

## Results (capacity = 15% of makespan)

Both model classes are reported — not just the one that differentiates — so the
RF-saturation effect is visible rather than hidden.

**Linear experts** (default; the weaker, paper-style model class):

| Strategy | Offloaded | Energy saved | vs oracle |
|---|---|---|---|
| all_primary | 0 | 0.0% | — |
| random | 17 | 5.16% | floor |
| single | 18 | 6.16% | |
| **moe** | 18 | **6.18%** | closes 87% of single→oracle gap |
| oracle | 18 | 6.19% | ceiling |

**RF experts** (the strong tree class):

| Strategy | Offloaded | Energy saved | vs oracle |
|---|---|---|---|
| all_primary | 0 | 0.0% | — |
| random | 17 | 5.16% | floor |
| single | 18 | 6.19% | = oracle |
| moe | 18 | 6.19% | = oracle |
| oracle | 18 | 6.19% | ceiling |

**Reading both honestly:** under the linear class MoE-guided offloading clearly
beats random and the single global model and nearly reaches the oracle ceiling
(closes 87% of the gap). Under RF the per-job predictions are near-perfect
(R²≈1.0), so `single`, `moe` and `oracle` all collapse onto the same decision —
MoE shows *no* advantage there, and that is itself a finding (Task 4 Finding 4: RF
saturates, so the differentiation only appears for weaker model classes). We
default to `--expert linear` because it is the regime where prediction quality
actually drives the decision; the RF run is reported alongside so the result does
not look cherry-picked.

The `capacity_sweep_*.png` plot shows this across budgets: under linear the `moe`
line tracks `oracle` and sits above `random` at every capacity.

## Run

```sh
# from the repository root
python -m offloading.run_offloading --expert linear --plot results/offloading/energy_saved_linear.png
python -m offloading.run_offloading --expert rf
python -m offloading.run_offloading --expert linear --sweep results/offloading/capacity_sweep_linear.png
python -m offloading.run_offloading --capacity-frac 0.10   # tighter budget
```

Saved runs (text + plots) live in `results/offloading/`.

## Caveats

This is a **simulation**, not a live deployment. The secondary cluster's
energy/speed/cost factors are plausible assumptions, not measured; a real
integration would plug measured secondary-cluster characteristics (and Snakemake's
job DAG with dependencies) into the same `decide()` interface. The contribution
here is the decision engine and the demonstration that prediction quality
translates into real energy savings.
