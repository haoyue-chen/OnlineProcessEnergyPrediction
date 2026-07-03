# Mixture-of-Experts + Online Learning (Phase 2)

This package implements the Phase-2 objective from `Phase1_Summary_Phase2_Plan.md`:
a **Mixture-of-Experts (MoE)** energy estimator and an **online-learning** variant,
together with the experiments that validate them against the Phase-1 baselines.

It builds directly on the Phase-1 data flow in `work/`: four workload runs (DAW1,
DAW2, Phoronix, Stress), each a per-process / per-interval `process_interval_data.parquet`.

## Layout

| File | Purpose |
|---|---|
| `data.py` | Loads the four parquet runs, aggregates per-process rows into one row per interval (features summed, `interval_energy` as target), and labels each interval with its workload. |
| `registry.py` | Central model registry. Batch (sklearn): `linear`, `rf`, `extra_trees`, `hgb`, `knn`, `mlp`, `svr`. Online (river): `linear`, `pa`, `htr`, `hatr`, `arf`, `knn`. |
| `moe.py` | `MixtureOfExperts` (gate + per-workload experts) and `SingleModel` (one global regressor). Expert class is any batch-registry key. |
| `run_moe_baseline.py` | **Task 4** — MoE vs Single Model, 5-fold CV (or `--time-split`). |
| `run_online.py` | **Task 5** — Static-Single vs Static-MoE vs Online-MoE under induced workload drift, prequential, with any online-registry expert. |
| `compare_models.py` | **Baseline survey** — every batch + online model family in one run, with CSV/plot output. |
| `online_baseline_comparison.py` | **Unified table** — the 7 screenshot families (Linear, SGD, RF, LightGBM/HGB, Hoeffding, Hoeffding-Adaptive, Adaptive-RF) with overall + per-workload R²/MAE. |

## Reproduce

```sh
cd energy-offloading   # run with PYTHONPATH=. so `moe` is importable

# Task 4 — MoE vs single model (default: 5-fold CV, matches the mid-term report)
PYTHONPATH=. python -m moe.run_moe_baseline --expert rf      --plot results/moe/task4_cv_rf.png
PYTHONPATH=. python -m moe.run_moe_baseline --expert linear  --plot results/moe/task4_cv_linear.png
PYTHONPATH=. python -m moe.run_moe_baseline --expert rf --time-split   # chronological hold-out

# Task 5 — static vs online MoE under drift (arf = Adaptive Random Forest, strongest)
PYTHONPATH=. python -m moe.run_online --online-expert arf --plot results/moe/task5_online_arf.png
PYTHONPATH=. python -m moe.run_online --online-expert linear

# Baseline survey — compare all model families
PYTHONPATH=. python -m moe.compare_models --plot results/moe/model_comparison.png

# Unified online-vs-offline table (7 families, per-workload R²/MAE)
PYTHONPATH=. python -m moe.online_baseline_comparison
```

Saved runs (text + plots + CSVs) live in `results/moe/`.


## Architecture

```text
Interval features  (process metrics summed over the interval)
        |
      Gate        RandomForest classifier — infers the workload regime
        |         from features alone (no label needed at inference)
   Expert select
        |
   Expert model   one regressor specialised per workload
        |
   Energy prediction
```

The gate is intentionally a *separate* classifier rather than a soft weighting, so
each expert trains only on its own regime and routing is inspectable (we report
gate accuracy). This is the transferable idea from **Sizey** (Task 1): pick the
estimator that fits the current task profile instead of forcing one global model.

The online variant (Task 5) keeps the gate frozen but replaces each expert with a
**River** online regressor that updates sample-by-sample, addressing Phase-1
problem #1 (static, frozen models that cannot follow workload drift) — the
**River** reading from Task 2.

## Results

### Baseline survey — all model families (`compare_models.py`)

Beyond the original linear-vs-RF and SGD-only setup, every model *family* is
compared, so the choice of expert is evidence-based. Batch = 5-fold CV
(subsampled to 8k rows so SVR/MLP are tractable); online = prequential over the
drifting workload stream.

**Online-learning capability** is the axis that separates the two registries. A
model is "online" if it can update incrementally — `predict → receive true value →
update` — without retraining on all past data. Only online-capable models can
adapt to workload drift (`DAW1 → DAW2 → Phoronix → Stress`):

| family | online? | here | why it matters |
|---|---|---|---|
| Linear (OLS) | No | batch `linear` | closed-form fit, no incremental update |
| SGD-linear | **Yes** | online `linear` (river + Adam) | per-sample gradient step |
| Random Forest | No | batch `rf` | bagging needs the full dataset |
| LightGBM / boosting | No | batch `hgb` | offline boosting rounds |
| Hoeffding tree | **Yes** | online `htr` / `hatr` | stream decision tree |
| Adaptive RF | **Yes** | online `arf` | streaming forest, adapts to drift |

The batch registry holds the offline families (used as MoE experts + offline
accuracy survey); the online registry holds the river models that genuinely
support `learn_one` / `predict_one`. `registry.is_online_capable(name, kind)`
encodes this map.

**Batch (single global model vs MoE), R²:**

| model | single R² | MoE R² | family |
|---|---|---|---|
| extra_trees | 0.975 | 0.976 | bagged trees |
| rf | 0.972 | 0.973 | bagged trees |
| hgb | 0.971 | 0.973 | boosted trees (LightGBM stand-in) |
| knn | 0.963 | 0.961 | instance-based |
| mlp | 0.942 | 0.855 | neural net |
| svr | 0.922 | 0.950 | kernel |
| linear | 0.802 | 0.843 | linear (paper class) |

**Online (prequential over drift), R²** — extra models beyond the 7-family set
(SGD/Hoeffding/Adaptive-RF are in the unified table below):

| model | R² | MAE | family |
|---|---|---|---|
| **arf** | **0.931** | **6.33** | Adaptive Random Forest (plan Task 2) |
| knn | 0.902 | 6.41 | instance-based |
| htr | 0.831 | 11.29 | Hoeffding tree |
| hatr | 0.745 | 16.60 | Hoeffding adaptive tree |
| pa | 0.129 | 16.88 | passive-aggressive |

Takeaways: all tree ensembles cluster at ~0.97 (the accuracy ceiling, so MoE adds
little there — RF saturates); MoE's lift is largest for the weaker linear class
(+0.04) and SVR (+0.027). Online, **ARF is the clear winner** and is exactly the
"Adaptive Random Forest" the Phase-2 plan named in Task 2. CSVs:
`results/moe/{batch,online}_comparison.csv`.

### Final unified comparison — the screenshot's 7 families, one table

`online_baseline_comparison.py` trains the exact families from the project's
online-learning table and reports overall **and per-workload** R²/MAE together.
Protocol differs by type and is labelled: offline = 5-fold CV; online =
prequential over the drift stream `DAW1 → DAW2 → Phoronix → Stress` (warm-up
excluded). Full output: `results/moe/online_baseline_comparison.{md,csv,txt}`.

**Overall:**

| Model | Online | Protocol | R² | MAE (Wh) |
|---|---|---|---|---|
| Linear (OLS) | no | 5-fold CV | 0.792 | 15.08 |
| SGD (linear+Adam) | yes | prequential | 0.887 | 9.64 |
| RF | no | 5-fold CV | 0.964 | 3.67 |
| LightGBM/HGB | no | 5-fold CV | 0.964 | 4.21 |
| Hoeffding | yes | prequential | 0.845 | 10.57 |
| Hoeffding Adaptive | yes | prequential | 0.776 | 15.52 |
| **Adaptive RF** | yes | prequential | **0.936** | **6.07** |

**Per-workload R²:**

| Model | Online | DAW1 | DAW2 | Phoronix | Stress |
|---|---|---|---|---|---|
| Linear (OLS) | no | 0.587 | 0.770 | 0.657 | 0.738 |
| SGD (linear+Adam) | yes | 0.803 | 0.835 | 0.700 | 0.867 |
| RF | no | 0.943 | 0.960 | 0.897 | 0.968 |
| LightGBM/HGB | no | 0.939 | 0.956 | 0.889 | 0.976 |
| Hoeffding | yes | 0.686 | 0.782 | 0.430 | 0.900 |
| Hoeffding Adaptive | yes | 0.716 | 0.741 | 0.555 | 0.543 |
| Adaptive RF | yes | 0.882 | 0.935 | 0.691 | 0.954 |

**Per-workload MAE (Wh):**

| Model | Online | DAW1 | DAW2 | Phoronix | Stress |
|---|---|---|---|---|---|
| Linear (OLS) | no | 19.13 | 15.57 | 10.48 | 15.59 |
| SGD (linear+Adam) | yes | 12.62 | 11.04 | 6.45 | 9.11 |
| RF | no | 5.72 | 4.12 | 2.39 | 2.95 |
| LightGBM/HGB | no | 6.26 | 4.64 | 3.27 | 3.23 |
| Hoeffding | yes | 15.16 | 10.59 | 10.53 | 7.45 |
| Hoeffding Adaptive | yes | 15.29 | 15.51 | 9.02 | 20.78 |
| Adaptive RF | yes | 9.12 | 5.78 | 5.65 | 4.52 |

Reading it: among **online** models that can follow drift, **Adaptive RF is best
by a wide margin** (R² 0.936, within ~0.03 of the offline RF/LightGBM ceiling) —
the streaming forest gets most of the tree-ensemble accuracy *while* adapting
online. SGD-linear is the next-best online option and far above batch OLS. The
offline RF/LightGBM are the overall accuracy ceiling but cannot update on a stream.

### Task 4 — MoE vs Single Model (5-fold cross-validation, R²)

Evaluation uses **5-fold CV** by default. A single random/chronological hold-out is
sensitive to which "hard" rows land in the test split (Phoronix especially), so CV
gives the stable numbers that line up with the mid-term report's baseline (see
"Consistency" below). Add `--time-split` for the chronological-hold-out framing.

With **RandomForest** experts, a single global RF already saturates (it partitions
workloads internally), so explicit MoE adds little — the tree ceiling is high:

| Workload | Single (RF) | MoE (RF) | Gate acc. |
|---|---|---|---|
| DAW1 | 0.944 | 0.946 | 1.00 |
| DAW2 | 0.960 | 0.962 | 1.00 |
| Phoronix | 0.898 | 0.903 | 0.99 |
| Stress | 0.967 | 0.968 | 1.00 |
| **Overall** | **0.964** | **0.966** | — |

With **linear** experts (the paper's model class), MoE's benefit is clear and large
— specialising the linear map per workload recovers most of the gap to the tree
ceiling, and the gate routes near-perfectly:

| Workload | Single (linear) | MoE (linear) | R² gain |
|---|---|---|---|
| DAW1 | 0.587 | 0.758 | **+0.17** |
| DAW2 | 0.770 | 0.875 | **+0.10** |
| Phoronix | 0.657 | 0.808 | **+0.15** |
| Stress | 0.738 | 0.882 | **+0.14** |
| **Overall** | **0.792** | **0.888** | **+0.10** |

**Takeaway:** MoE's value is in lifting a weaker, interpretable model class up
toward the tree ceiling, while keeping per-expert transparency and a >99%-accurate
gate. (LightGBM is not installed; per Phase-1 Finding 4, RF ≈ LightGBM, so RF is
the representative tree expert.)

### Consistency with the mid-term report

Under the same 5-fold CV methodology the report used, our per-workload baselines
reproduce its numbers (`TeamGreen-MidTerm-Presentation.pdf`, p.19):

| Workload | Report Linear / RF | Ours (5-fold) Linear / RF |
|---|---|---|
| DAW1 | 0.707 / 0.932 | 0.587* / 0.944 |
| DAW2 | 0.846 / 0.964 | 0.770* / 0.960 |
| Phoronix | 0.730 / 0.874 | 0.657* / 0.898 |
| Stress | 0.864 / 0.963 | 0.738* / 0.967 |

The **RF** column matches closely. The *linear* column is lower here only because
this table's "Single (linear)" is one global model trained across *all* workloads
at once (the MoE comparison baseline), whereas the report fits a separate linear
model per workload. Per-workload linear fits reproduce the report (≈0.74/0.88/0.68/0.88);
see `results/moe/consistency_with_midterm.md` for the full check.

### Task 5 — Static vs Online MoE under workload drift (prequential, R²)

The four workloads are streamed end-to-end (`DAW1 → DAW2 → Phoronix → Stress`) to
induce drift; every interval is predicted before the model learns it. The default
online expert is now **ARF (Adaptive Random Forest)** — the strongest online model
in the baseline survey and the one the Phase-2 plan named in Task 2.

| Model | DAW1 | DAW2 | Phoronix | Stress | Overall R² | Overall MAE |
|---|---|---|---|---|---|---|
| Static-Single (frozen) | 0.50 | 0.71 | -0.17 | 0.70 | 0.71 | 18.3 |
| Static-MoE (frozen) | 0.59 | 0.87 | -3.99 | -2.97 | -0.32 | 18.4 |
| **Online-MoE (ARF)** | **0.88** | **0.94** | **0.68** | **0.93** | **0.93** | **6.2** |

For reference the Adam-linear online expert (`--online-expert linear`) reaches
overall R² 0.78 / MAE 10.1 — already beating both static baselines; ARF pushes it
much further.

**Takeaway:** a frozen MoE warmed only on early data actually *degrades* under
drift (its specialised experts overfit the warm-up regime, Phoronix → -3.99). The
online MoE adapts as each workload arrives and wins decisively — **R² 0.93 vs 0.71,
MAE 6.2 vs 18.3 Wh (-66%)**, and even lifts Phoronix from -0.17 to +0.68 —
validating `MoE + Online Learning` over both a single static model and a static
MoE. Plot: `results/moe/task5_online_arf.png`.

## Mapping to the Phase-2 plan

| Plan task | Status |
|---|---|
| Task 1 — Sizey expert/gating ideas | Applied: hard gate selecting a per-regime expert. |
| Task 2 — River online-learning survey | Applied: River linear (Adam) and Hoeffding adaptive tree experts. |
| Task 3 — MoE architecture | `moe.py` (gate + experts), diagram above. |
| Task 4 — MoE vs Single Model | `run_moe_baseline.py`, results above. |
| Task 5 — MoE vs MoE + Online | `run_online.py`, results above. |
