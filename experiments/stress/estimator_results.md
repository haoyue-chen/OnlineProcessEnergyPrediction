# Stress Workload - Baseline Estimator Results

## Dataset Overview

Dataset:

* Stress Benchmark

Workload Type:

* Synthetic stress workload
* CPU-intensive benchmark
* Includes stress-ng components:

  * stress-ng-cpu
  * stress-ng-vm
  * stress-ng-hdd
  * stress-ng-matrix
  * stress-ng-syscall

Statistics:

| Metric               | Value     |
| -------------------- | --------- |
| Intervals            | 8501      |
| Process-level Rows   | 1,726,700 |
| Mean Interval Energy | 290.13 Wh |

---

# Selected Features

Selected through Sequential Forward Selection (SFS):

1. delta_instructions
2. syscall_class_file

These features achieved the highest predictive performance while maintaining a compact model.

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
| R²                        | 0.8640 |
| MAE (Wh)                  | 11.78  |
| MAE (%)                   | 4.06   |
| Mean Interval Energy (Wh) | 290.13 |

---

# Learned Weights

| Feature            | Weight |
| ------------------ | ------ |
| delta_instructions | 290.71 |
| syscall_class_file | 32.79  |

Static Energy Component:

```text
264.94 Wh
```

---

# Feature Interpretation

## CPU Instructions

delta_instructions receives the dominant weight.

This confirms that:

* executed instructions are the primary energy driver
* workload behavior is strongly compute-bound
* CPU activity explains most energy variation

---

## File-System Activity

syscall_class_file contributes a smaller but measurable improvement.

Possible explanations:

* file-system access correlates with workload scheduling
* disk-related stress-ng components introduce additional energy variation
* I/O behavior complements CPU execution information

---

# Process-Level Analysis

Frequently observed processes:

* stress-ng-cpu
* stress-ng-vm
* stress-ng-hdd
* stress-ng-syscall
* stress-ng-matrix
* influxd
* python

These processes correspond directly to the workload generation framework and monitoring infrastructure.

No negative energy allocations were observed.

---

# Prediction Quality

The model achieves:

```text
R² = 0.8640
MAE = 4.06%
```

This indicates strong agreement between:

* measured interval energy
* predicted interval energy

The estimator captures:

* workload execution phases
* CPU-intensive activity
* I/O-intensive activity
* idle periods

with high accuracy.

---

# Comparison with Other Datasets

| Dataset  | R²    | MAE (%) |
| -------- | ----- | ------- |
| DAW1     | 0.707 | 6.07    |
| Phoronix | 0.730 | 5.13    |
| DAW2     | 0.846 | 3.79    |
| Stress   | 0.864 | 4.06    |

Observation:

Stress is the most predictable workload among all evaluated datasets.

---

# Key Findings

Energy consumption is dominated by:

1. CPU instruction execution
2. File-system activity

Only two features are required to achieve high prediction accuracy.

This suggests that:

* compute-heavy workloads are easier to model
* hardware counters provide highly informative signals
* complex system-call behavior is less important than in workflow-oriented datasets

---

# Research Implications

Stress represents a workload with:

* stable execution behavior
* low workload diversity
* minimal phase complexity

Consequently, a simple linear model is sufficient to achieve strong performance.

This contrasts with DAW workflows, where communication and system-call behavior play a much larger role.

---

# Limitations

The model remains:

* static
* offline-trained
* workload-specific

Potential challenges include:

* unseen workloads
* changing workload mixtures
* concept drift
* heterogeneous execution environments

---

# Next Steps

1. Train SGDRegressor

2. Train Random Forest Regressor

3. Train LightGBM / XGBoost

4. Compare against the baseline estimator

5. Evaluate online adaptation capability

6. Investigate Mixture-of-Experts approaches

---

# Conclusion

A compact two-feature linear model achieves:

```text
R² = 0.8640
MAE = 4.06%
```

on the Stress workload dataset.

The results demonstrate that CPU instruction activity alone explains the majority of energy variation, making Stress the most predictable workload among the evaluated baselines.
