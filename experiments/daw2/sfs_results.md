# DAW Dataset 2 - Sequential Forward Selection Results

## Dataset Overview

Dataset:

* DAW Scientific Workflow Dataset 2

Workload Characteristics:

* Data-intensive scientific workflow
* Significant network communication
* Multi-stage execution pipeline
* Strong temporal continuity

Intervals:

* 6131

Process Records:

* 2,663,509

---

# Sequential Forward Selection (SFS)

Method:

* Greedy Sequential Forward Selection
* CVXPY linear baseline estimator
* Feature added only if performance gain exceeds threshold

Evaluation Metrics:

* Interval-level R²
* MAE (%)

---

# Feature Selection Progress

| Step | Selected Feature     | R²     | MAE (%) |
| ---- | -------------------- | ------ | ------- |
| 1    | delta_net_send_bytes | 0.7125 | 5.20    |
| 2    | syscall_class_other  | 0.8337 | 3.88    |
| 3    | delta_instructions   | 0.8455 | 3.79    |

Stopping criterion reached:

* Additional features improve R² by less than 0.01
* Diminishing returns observed

---

# Best Feature Combination

```text
delta_net_send_bytes
syscall_class_other
delta_instructions
```

Performance:

| Metric  | Value  |
| ------- | ------ |
| R²      | 0.8455 |
| MAE (%) | 3.79   |

---

# Learned Weights

| Feature              | Weight |
| -------------------- | ------ |
| delta_net_send_bytes | 95.93  |
| syscall_class_other  | 96.70  |
| delta_instructions   | 41.69  |

---

# Observations

## 1. Network Activity Dominates

The strongest single predictor is:

```text
delta_net_send_bytes
```

Using only network traffic already achieves:

```text
R² = 0.7125
```

This indicates that workflow communication behavior is strongly linked to energy consumption.

---

## 2. System-call Behavior Adds Major Improvement

Adding:

```text
syscall_class_other
```

increases performance:

```text
0.7125 → 0.8337
```

This is the largest performance gain observed during feature selection.

The workflow energy profile is therefore highly influenced by operating-system interactions.

---

## 3. Hardware Activity Provides Additional Refinement

Adding:

```text
delta_instructions
```

further improves performance:

```text
0.8337 → 0.8455
```

CPU execution information contributes additional predictive power but is less important than communication behavior.

---

## 4. Strong Feature Redundancy

After three selected features:

```text
R² improvement < 0.01
```

Most remaining features contain redundant information.

This suggests that only a small subset of workload-aware metrics is necessary for accurate prediction.

---

# Comparison with DAW Dataset 1

| Property          | DAW1             | DAW2                |
| ----------------- | ---------------- | ------------------- |
| Dominant Feature  | syscall activity | network traffic     |
| Best R²           | ~0.71            | 0.85                |
| Best MAE          | ~6%              | 3.79%               |
| Workload Behavior | syscall-heavy    | communication-heavy |
| Predictability    | moderate         | high                |

---

# Conclusions

Key findings:

* Energy consumption is primarily driven by network communication and workflow activity.
* Only three features are required to achieve strong prediction performance.
* Hardware counters contribute less than communication metrics.
* Significant feature redundancy exists after the first three selected features.

These selected features will be used for:

* Baseline linear regression
* SGDRegressor
* Random Forest
* LightGBM / XGBoost
* Online learning experiments
* Mixture-of-Experts evaluation

---

# Research Implication

Different workloads exhibit different dominant energy predictors.

For DAW Dataset 2:

```text
Network traffic
+
System-call behavior
+
CPU execution activity
```

are the most informative metrics.

This supports the hypothesis that workload-specific modeling and Mixture-of-Experts approaches may outperform a single global model.
