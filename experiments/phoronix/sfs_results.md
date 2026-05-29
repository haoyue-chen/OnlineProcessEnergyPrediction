# Phoronix Workload - Sequential Forward Selection Results

## Dataset Overview

Dataset:

* Phoronix Benchmark Suite

Workload Type:

* Synthetic benchmark workload
* CPU-intensive and memory-intensive benchmark execution
* Includes:

  * 7-Zip Compression
  * OpenSSL Cryptography
  * STREAM Memory Benchmark

Statistics:

| Metric          | Value |
| --------------- | ----- |
| Intervals       | 6689  |
| Train Intervals | 5351  |
| Test Intervals  | 1338  |

---

# Sequential Forward Selection (SFS)

Method:

* Greedy Sequential Forward Selection
* CVXPY baseline estimator
* Evaluation on held-out test intervals

Stopping Criterion:

```text
min_gain = 0.01
```

Feature selection stops when additional features improve R² by less than 0.01.

---

# Feature Selection Progress

| Step | Selected Feature     | R²     | MAE (%) |
| ---- | -------------------- | ------ | ------- |
| 1    | delta_net_send_bytes | 0.3902 | 2.87    |
| 2    | delta_cycles         | 0.5073 | 2.53    |
| 3    | delta_cache_misses   | 0.6442 | 1.85    |

Selection stopped after Step 3.

Additional features produced negligible improvements.

---

# Best Feature Combination

Final selected feature set:

```text
delta_net_send_bytes
delta_cycles
delta_cache_misses
```

Performance:

| Metric  | Value  |
| ------- | ------ |
| R²      | 0.6442 |
| MAE (%) | 1.85   |

---

# Learned Feature Weights

| Feature              | Weight |
| -------------------- | ------ |
| delta_cycles         | 155.53 |
| delta_net_send_bytes | 108.77 |
| delta_cache_misses   | 22.47  |

---

# Interpretation

## Step 1: Network Activity

The strongest single-feature predictor was:

```text
delta_net_send_bytes
```

Performance:

```text
R² = 0.3902
```

Although Phoronix is primarily a benchmark suite, network traffic appears to capture benchmark phase transitions and workload execution periods.

---

## Step 2: CPU Hardware Activity

Adding:

```text
delta_cycles
```

improved performance significantly:

```text
0.3902 → 0.5073
```

CPU cycle counts provide direct information about processor activity and computational intensity.

---

## Step 3: Cache Behavior

Adding:

```text
delta_cache_misses
```

further improved performance:

```text
0.5073 → 0.6442
```

Cache misses capture memory hierarchy effects and workload efficiency.

This feature appears particularly important for memory-intensive benchmarks such as STREAM.

---

# Feature Redundancy Analysis

Several features showed strong correlation with energy but were not selected:

* delta_cpu_time_psutil
* delta_cpu_ns
* delta_instructions

This suggests:

```text
High Correlation
≠
High Additional Predictive Value
```

These metrics likely contain information already represented by:

* delta_cycles
* delta_cache_misses

Therefore SFS excluded them.

---

# Comparison with DAW Workloads

| Property                 | DAW1          | DAW2              | Phoronix          |
| ------------------------ | ------------- | ----------------- | ----------------- |
| Dominant Features        | Syscall + CPU | Network + Syscall | Hardware Counters |
| Best R²                  | ~0.71         | ~0.85             | 0.64              |
| Best MAE (%)             | ~6.1          | ~3.8              | 1.85              |
| Communication Importance | Medium        | High              | Medium            |
| CPU Importance           | Medium        | Medium            | High              |
| Cache Importance         | Low           | Low               | High              |

---

# Key Findings

The Phoronix benchmark suite is primarily driven by:

1. CPU execution activity
2. Cache behavior
3. Benchmark phase transitions

The selected feature set:

```text
delta_net_send_bytes
delta_cycles
delta_cache_misses
```

captures these characteristics effectively.

---

# Research Implications

Unlike the DAW scientific workflows:

* system-call behavior contributes less information
* hardware counters contribute substantially more information

This supports the hypothesis that:

```text
Different workload classes require different feature sets for accurate energy modeling.
```

A single global feature set may therefore be insufficient for workload-aware energy estimation.

---

# Next Step

Use the selected feature set as input to:

* CVXPY baseline estimator
* SGDRegressor
* Random Forest Regressor
* LightGBM / XGBoost

and compare prediction performance across models.
