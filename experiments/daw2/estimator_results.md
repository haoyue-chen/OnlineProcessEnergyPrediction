# DAW Dataset 2 - Baseline Estimator Results

## Dataset Overview

Dataset:

* DAW Scientific Workflow Dataset 2

Workload Type:

* Data-intensive scientific workflow
* Communication-heavy workload
* Repeated pipeline execution pattern

Data Statistics:

| Metric             | Value     |
| ------------------ | --------- |
| Intervals          | 6131      |
| Process-level Rows | 2,663,509 |
| Train/Test Split   | 80% / 20% |

---

# Selected Features

Selected through Sequential Forward Selection (SFS):

1. delta_net_send_bytes
2. syscall_class_other
3. delta_instructions

These features achieved the best validation performance before diminishing returns appeared.

---

# Baseline Model

Model:

* Linear Energy Estimator
* CVXPY Optimization
* L1-Regularized Least Squares

Target:

```text
interval_energy
```

---

# Final Prediction Performance

| Metric                    | Value  |
| ------------------------- | ------ |
| R²                        | 0.8455 |
| MAE (Wh)                  | 11.51  |
| Mean Interval Energy (Wh) | 303.56 |
| MAE (%)                   | 3.79%  |

---

# Learned Weights

| Feature              | Weight |
| -------------------- | ------ |
| delta_net_send_bytes | 95.93  |
| syscall_class_other  | 96.70  |
| delta_instructions   | 41.69  |

Static Energy Component:

```text
227.68 Wh
```

---

# Interpretation

## Network Activity

delta_net_send_bytes receives a large positive weight.

This indicates that communication intensity is a major contributor to energy consumption.

---

## System-call Activity

syscall_class_other receives the largest weight.

The model relies heavily on operating-system interactions and process behavior to explain energy variations.

---

## CPU Execution

delta_instructions remains important but contributes less than communication-related features.

This suggests that the workflow is not purely CPU-bound.

---

## Static Power

The learned static energy term:

```text
227.68 Wh
```

represents baseline node energy consumption independent of workload activity.

A large fraction of total energy is therefore explained by platform idle power.

---

# Process-Level Observations

Dominant processes observed during execution:

* java
* python
* influxd
* docker-proxy
* STAR-avx2

These processes correspond to workflow orchestration, monitoring, container management, and bioinformatics pipeline execution.

No negative per-process energy estimates were observed.

---

# Prediction Quality

The predicted interval-energy curve closely follows the measured smart-meter energy.

Observed characteristics:

* Workload phases are captured accurately.
* Idle periods are identified correctly.
* Major workload transitions are tracked.
* Prediction errors remain small throughout execution.

---

# Comparison with Previous DAW Dataset

| Metric            | DAW1          | DAW2                             |
| ----------------- | ------------- | -------------------------------- |
| R²                | 0.707         | 0.846                            |
| MAE (%)           | 6.07%         | 3.79%                            |
| Dominant Features | syscall + CPU | network + syscall + instructions |
| Predictability    | Moderate      | High                             |

DAW2 exhibits substantially stronger feature-energy relationships and is easier to model using a linear estimator.

---

# Key Finding

A small subset of workload-aware metrics is sufficient to explain most energy variation:

```text
delta_net_send_bytes
+
syscall_class_other
+
delta_instructions
```

These three features achieve:

```text
R² = 0.8455
MAE = 3.79%
```

which represents strong baseline performance for a static linear model.

---

# Limitations

The current estimator remains:

* static
* offline-trained
* globally optimized

Potential challenges include:

* workload phase transitions
* concept drift
* unseen workloads
* changing system behavior

---

# Next Steps

1. Train SGDRegressor (online learning baseline)

2. Train Random Forest Regressor

3. Train LightGBM / XGBoost

4. Compare all models against the linear baseline

5. Evaluate workload-dependent performance

6. Investigate Mixture-of-Experts (MoE) routing strategies
