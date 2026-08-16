# Online Process Energy Prediction

Predicting node-level energy consumption from process-level resource metrics,
with workload-aware and resource-aware Mixture-of-Experts models, online
adaptation under concept drift, and energy-aware job offloading.

## 1. Project Overview

The input is per-process, per-interval resource counters: CPU time,
instructions, cycles, cache misses, branch instructions, RSS memory, I/O bytes,
network bytes, context switches, and a syscall-class breakdown — 16 numeric
features. Energy is measured at node level, so it is available once per
interval rather than once per process.

The problem is therefore framed at interval granularity: all process rows
sharing a timestamp are summed into one feature vector, and the target is that
interval's node energy. Four workload runs are used — DAW1, DAW2, Phoronix and
Stress. A single global regressor trained across all four is the baseline.

Two Mixture-of-Experts variants are evaluated against it. The **workload-aware
MoE** trains one expert per workload plus a gate classifier that routes each
interval from its features alone, so the workload label is not needed at
inference time. The **resource-aware MoE** instead partitions the 16 features
into four resource groups (CPU, memory, I/O, network), trains one expert per
group, and combines them with a learned or importance-derived gate.

Two further components use these models. **Online adaptation** streams the four
workloads end-to-end to induce drift and compares frozen models against
incrementally updated River regressors under a prequential (test-then-train)
protocol. **Energy-aware offloading** uses the predictor as a scheduling input:
jobs are ranked by predicted energy and placed on a more efficient secondary
cluster under a capacity budget.

## 2. Research Problem

A single global regressor treats all workloads as one distribution, but
different workloads have different resource-behaviour patterns and different
predictability, and a model trained once is frozen against later drift. This
project asks three questions:

1. Does conditioning experts on workload — or on resource type — improve
   interval-level energy prediction over one global model?
2. Does incremental online updating recover accuracy when the workload
   distribution shifts mid-stream?
3. Is the resulting predictor accurate enough to drive an energy-aware
   scheduling decision?

Results vary by component and by model class; the report gives the exact
figures, and the reference outputs are described in section 12.

## 3. Repository Structure

| Path | Contents |
|---|---|
| `moe/` | Data loading, interval aggregation, workload-grouped MoE, model registry, and the offline and online experiment entry points. |
| `feature_moe/` | Resource-grouped MoE: feature-to-group mapping, resource importance, learned and importance gates, plus the `LearnedResourceMoE` used in the controlled evaluation. |
| `offloading/` | Workflow segmentation, the two-cluster model, the placement policy, and the offloading entry point. |
| `snakemake_integration/` | Snakemake workflow running the offloading decision as a real DAG. |
| `inference/` | HTTP service exposing a trained model. |
| `moe_export/` | Exports a trained MoE to the artifact the service loads. |
| `experiments/` | The controlled-dataset LORO/LOGO runner behind report component 5. |
| `demo/` | The three demonstration scripts described below. |
| `deploy/` | Docker smoke test and Kubernetes manifest templates. |
| `datasets/demo/` | Small real-data demo subset (see section 6). |
| `models/` | Pre-exported model artifacts used by the inference service. |
| `results/reference/` | Committed reference outputs for the report. |

## 4. Requirements

- Linux or WSL (developed on WSL / Ubuntu 24.04).
- Python 3.12.
- `git`, which the Snakemake workflow tooling expects on `PATH`.

```sh
python3.12 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Run everything from the repository root. The demo scripts set `PYTHONPATH`
themselves; for bare module invocations, prefix with `PYTHONPATH=.`.

PyTorch is **not** in `requirements.txt` and is not needed for any demo or for
report components 1–4, 6 and 7. It is required only by
`feature_moe/learned_moe.py`, used in component 5 (section 11).

## 5. Quick Start

```sh
git clone <repository-url>
cd OnlineProcessEnergyPrediction
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./demo/run_offline_demo.sh       # Demo 1  (~30 s)
./demo/run_online_demo.sh        # Demo 2  (~10 s)
./demo/run_offloading_demo.sh    # Demo 3  (~2 s)
```

No dataset download, no environment variables, and no absolute paths are
required — the demos locate the repository root and the bundled demo dataset
themselves. All generated output goes to `results/demo/`, which is git-ignored.

## 6. Demo Dataset

`datasets/demo/` contains a **small real-data subset**, 0.54 MB total: 2,000
consecutive intervals from each of the four measured workloads, 8,000 intervals
in all. It is stored interval-aggregated (one row per interval) using the
project's own `moe.data.aggregate_intervals`, so the loader reads it unchanged.
No values were altered, no features renamed, no units changed, and nothing was
synthesised.

```
datasets/demo/<run-directory>/runs/demo/datasets/process_interval_data.parquet
```

| Run directory | Label | Intervals | Size |
|---|---|---|---|
| `baseline-daw-4h-01` | DAW1 | 2,000 | 133 KB |
| `baseline-daw-clean-perf-03` | DAW2 | 2,000 | 159 KB |
| `baseline-phoronix-clean-perf-01` | Phoronix | 2,000 | 131 KB |
| `baseline-stress-clean-perf-01` | Stress | 2,000 | 133 KB |

> **This dataset does not reproduce the numbers in the report.** It exists only
> to show that the repository runs end-to-end after cloning. The reported
> results come from the full measurement datasets (~100 MB of per-process
> Parquet, ~9.1 M process rows across 27,121 intervals) and, for component 5,
> from a separate 18-phase controlled dataset. Neither is committed here.

Point the code at the full data with `PEA_DATA_ROOT`:

```sh
PEA_DATA_ROOT=/path/to/full/dataset PYTHONPATH=. python -m moe.run_moe_baseline --expert linear
```

## 7. Demo 1 — Offline Prediction

```sh
./demo/run_offline_demo.sh
```

Runs `moe.run_moe_baseline` (global single model vs workload-aware MoE, 5-fold
CV, with per-workload R²/MAE and gate routing accuracy) followed by
`feature_moe.run_resource_moe` (resource-aware MoE with learned and importance
gates against both baselines, plus the resource importance breakdown and
learned gate weights).

Output: tables and plots in `results/demo/`. Override the model class with
`DEMO_EXPERT=rf`.

## 8. Demo 2 — Online Adaptation

```sh
./demo/run_online_demo.sh
```

Runs `moe.run_online`. The four workloads are streamed end-to-end to induce
drift, and Static-Single, Static-MoE and Online-MoE are compared prequentially.
The first 15% of each workload is warm-up and is excluded from scoring.

Set `WITH_PERIODIC_RETRAIN=1` to add the optional Periodic-Retrain baseline,
which is off by default so the default output matches the three reported rows.
Override the online expert with `DEMO_ONLINE_EXPERT=hatr`.

Output: table and error-over-time plot in `results/demo/`.

## 9. Demo 3 — Energy-aware Offloading

```sh
./demo/run_offloading_demo.sh
```

Runs `offloading.run_offloading`. The MoE predicts per-job energy, and jobs are
placed on a primary or a more efficient secondary cluster under a capacity
budget. Compares all-primary, random, single-model, MoE and oracle placement.

The secondary cluster is an **analytical model** — an energy factor, a speed
factor and a price per kWh relative to the primary (`offloading/clusters.py`) —
not measured hardware.

Output: decision table and plot in `results/demo/`.

## 10. Docker

```sh
docker build -t online-process-energy-prediction .

docker run --rm -v "$PWD/datasets/demo:/data/work:ro" \
  online-process-energy-prediction demo-offline
docker run --rm -v "$PWD/datasets/demo:/data/work:ro" \
  online-process-energy-prediction demo-online
docker run --rm -v "$PWD/datasets/demo:/data/work:ro" \
  online-process-energy-prediction demo-offloading
```

The image sets `PEA_DATA_ROOT=/data/work`; mount any dataset there. No dataset
is baked into the image. Add `-v "$PWD/results:/app/results"` to keep generated
output on the host.

The entry point also accepts `task4`, `task5`, `online`, `compare`, `offload`,
`snakemake`, `export`, `serve`, `all` and `bash`.

`docker compose` mounts `./datasets/demo` by default and starts the inference
service:

```sh
docker compose up --build
curl http://localhost:8800/health
```

The service listens on port 8800 and exposes `GET /health`, `GET /info`,
`POST /predict` and `POST /predict_batch`.

## 11. Reproducing the Main Experiments

These commands read the dataset through `moe/data.py`, so `PEA_DATA_ROOT`
applies to all of them. Run them against the **full** dataset to approach the
reported figures; against `datasets/demo/` they run but produce different
numbers.

| Report component | Command |
|---|---|
| 1. Offline model comparison | `PYTHONPATH=. python -m moe.compare_models` |
| 2. Workload-aware MoE | `PYTHONPATH=. python -m moe.run_moe_baseline --expert linear` |
| 3. Resource-aware MoE | `PYTHONPATH=. python -m feature_moe.run_resource_moe --expert rf` |
| 4. Online adaptation | `PYTHONPATH=. python -m moe.run_online --online-expert arf` |
| 4. Online vs offline families | `PYTHONPATH=. python -m moe.online_baseline_comparison` |
| 5. Controlled LORO/LOGO | `PYTHONPATH=. python -m experiments.run_learned_feature_moe_final` |
| 6. Energy-aware offloading | `PYTHONPATH=. python -m offloading.run_offloading --expert linear` |
| 7. Snakemake integration | `./snakemake_integration/run.sh compare` |

Component 5 additionally requires PyTorch and the controlled 18-phase dataset
(`data/controlled_workload_labeled_final.parquet`), neither of which is part of
this repository. It cannot be run from the demo dataset — its phase and group
labels do not exist there.

## 12. Results / Reference Outputs

`results/reference/` holds the committed reference material and a `README.md`
explaining what each file corresponds to. It currently contains the
controlled-dataset LORO/LOGO summary for component 5.

The reported figures for the other components are recorded in the per-package
documentation: `moe/README.md` (components 1, 2, 4), `feature_moe/README.md`
(component 3), `offloading/README.md` (component 6) and
`snakemake_integration/README.md` (component 7).

Everything the scripts generate at runtime under `results/` — `results/demo/`,
`results/moe/`, `results/feature_moe/` — is git-ignored and recreated by running
the commands above. Only `results/reference/` is committed.

## 13. Limitations

- The offloading evaluation is an analytical simulation; the secondary cluster
  is a scalar model of energy, speed and price, not measured hardware.
- No multi-cluster deployment was carried out. All measurements come from a
  single primary machine.
- The Kubernetes manifests in `deploy/k8s/` are templates and were never
  applied to a cluster.
- Raw measurement and data collection are handled by a separate project and are
  outside this submission.
- The resource-grouped MoE partitions features by resource type, but several
  system-level counters (context switches, syscall counts) have no single
  natural group and are assigned to the CPU expert. The groups are fixed by
  hand rather than discovered from the data.
- The controlled dataset has no network-dominant phase, so network-routing
  conclusions in component 5 are caveated rather than validated.
- The measurement dataset covers four workload runs on one machine. Results
  should not be assumed to transfer to other hardware or workload mixes.
- The demo dataset is a short slice of each workload. Drift behaviour in Demo 2
  is weaker than in the reported experiment, and all demo numbers differ from
  the report.

## 14. Relation to the Final Report

The repository contains the implementation behind all seven reported
components:

| # | Report component | Implementation |
|---|---|---|
| 1 | Process-level energy prediction | `moe/data.py`, `moe/registry.py`, `moe/compare_models.py` |
| 2 | Workload-aware MoE | `moe/moe.py`, `moe/run_moe_baseline.py` |
| 3 | Resource-aware MoE | `feature_moe/moe.py`, `feature_moe/run_resource_moe.py` |
| 4 | Online adaptation | `moe/run_online.py`, `moe/online_baseline_comparison.py` |
| 5 | Controlled LORO/LOGO | `feature_moe/learned_moe.py`, `experiments/run_learned_feature_moe_final.py` |
| 6 | Energy-aware offloading | `offloading/` |
| 7 | System integration | `snakemake_integration/`, `inference/` |

Two distinct things must not be confused:

- **A. Quick reproducibility demos** (sections 7–9) use the 0.54 MB demo
  dataset. They verify the repository executes; their numbers are not the
  report's.
- **B. Full experiments** (section 11) use the complete measurement datasets and
  are what produced the reported results. Those datasets are not committed
  here.
