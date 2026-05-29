# Phoronix Workload - Baseline Estimator Results

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

| Metric               | Value     |
| -------------------- | --------- |
| Intervals            | 6131      |
| Process-level Rows   | 2,663,509 |
| Mean Interval Energy | 303.56 Wh |

---

# Selected Features

Selected through Sequential Forward Selection (SFS):

1. delta_net_send_bytes
2. delta_cycles
3. delta_cache_misses

These features achieved the best prediction performance during feature selection.

---

# Baseline Model

Model:

* CVXPY Energy Estimator
* Linear Regression
* L1-Regularized Least Squares

Target:

```text
interval_energy
```

---

# Final Prediction Performance

| Metric                    | Value  |
| ------------------------- | ------ |
| R²                        | 0.7299 |
| MAE (Wh)                  | 15.57  |
| MAE (%)                   | 5.13   |
| Mean Interval Energy (Wh) | 303.56 |

---

# Learned Weights

| Feature              | Weight |
| -------------------- | ------ |
| delta_net_send_bytes | 161.68 |
| delta_cache_misses   | 36.11  |
| delta_cycles         | 7.56   |

Static Energy Component:

```text
216.51 Wh
```

---

# Feature Interpretation

## Network Activity

delta_net_send_bytes receives the largest weight.

This suggests that benchmark execution phases and workload transitions generate network activity that strongly correlates with energy consumption.

---

## Cache Behavior

delta_cache_misses contributes significantly to prediction quality.

Memory hierarchy effects appear important for workloads such as:

* STREAM
* Compression benchmarks
* Cryptographic benchmarks

---

## CPU Execution Activity

delta_cycles remains useful but contributes less than expected.

This suggests that cycle counts alone cannot fully explain benchmark energy behavior.

Additional information from memory access patterns is required.

---

# Process-Level Observations

Frequently observed processes include:

* java
* python
* influxd
* docker-proxy
* R

These processes represent:

* benchmark orchestration
* monitoring infrastructure
* container management
* workload execution

No negative process-energy estimates were observed.

---

# Prediction Quality

The model achieves:

```text
R² = 0.7299
MAE = 5.13%
```

This indicates moderate-to-strong agreement between:

* measured interval energy
* predicted interval energy

The model successfully captures:

* benchmark execution phases
* high-load periods
* idle periods

However, prediction quality remains lower than DAW Dataset 2.

---

# Comparison with DAW Dataset 2

| Metric            | DAW2                             | Phoronix                 |
| ----------------- | -------------------------------- | ------------------------ |
| R²                | 0.8455                           | 0.7299                   |
| MAE (%)           | 3.79                             | 5.13                     |
| Dominant Features | Network + Syscall + Instructions | Network + Cycles + Cache |
| Predictability    | High                             | Moderate                 |

Observation:

The scientific workflow exhibits stronger feature-energy relationships than the synthetic benchmark workload.

---

# Key Findings

The Phoronix workload energy profile is primarily influenced by:

1. Network activity
2. Cache behavior
3. CPU execution activity

Unlike DAW workflows:

* system-call activity contributes less information
* hardware performance counters contribute more information

---

# Limitations

The current estimator remains:

* static
* offline-trained
* globally optimized

Potential challenges include:

* workload phase transitions
* benchmark switching
* unseen workloads
* concept drift

---

# Research Implications

Different workload classes exhibit different dominant energy predictors.

For Phoronix:

```text
Network Activity
+
Cache Behavior
+
CPU Cycles
```

provide the strongest predictive information.

This differs substantially from DAW workloads, supporting the hypothesis that workload-aware feature selection and model adaptation may improve energy estimation performance.

---

# Next Steps

1. Train SGDRegressor

2. Train Random Forest Regressor

3. Train LightGBM / XGBoost

4. Compare all models against the baseline estimator

5. Evaluate workload-dependent performance

6. Investigate online learning and Mixture-of-Experts approaches

---

# Conclusion

A compact three-feature linear model achieves:

```text
R² = 0.7299
MAE = 5.13%
```

on the Phoronix benchmark workload.

The results demonstrate that hardware performance counters and memory-related behavior play a larger role than operating-system activity in benchmark-oriented energy estimation.
