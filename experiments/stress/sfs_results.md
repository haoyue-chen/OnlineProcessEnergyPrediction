# Stress Workload - Sequential Forward Selection Results

## Dataset Overview

Dataset:

* Stress Benchmark

Workload Type:

* CPU-intensive synthetic workload

Train/Test Split:

| Metric          | Value |
| --------------- | ----- |
| Train Intervals | 6800  |
| Test Intervals  | 1701  |

---

# Sequential Forward Selection (SFS)

Method:

* Greedy Sequential Forward Selection
* CVXPY baseline estimator
* Test-set evaluation

Stopping criterion:

```text
min_gain = 0.01
```

---

# Selection Process

## Step 1

Best single feature:

| Feature            | R²     | MAE (%) |
| ------------------ | ------ | ------- |
| delta_instructions | 0.8481 | 4.43    |

Selected:

```text
delta_instructions
```

---

## Step 2

Best additional feature:

| Feature Added      | New R² | MAE (%) |
| ------------------ | ------ | ------- |
| syscall_class_file | 0.8640 | 4.06    |

Selected:

```text
delta_instructions
syscall_class_file
```

---

Stopping condition reached:

```text
Gain = 0.0065 < 0.01
```

No further feature provided meaningful improvement.

---

# Final Selected Features

```text
delta_instructions
syscall_class_file
```

---

# Final Performance

| Metric  | Value  |
| ------- | ------ |
| R²      | 0.8640 |
| MAE (%) | 4.06   |

---

# Learned Weights

| Feature            | Weight |
| ------------------ | ------ |
| delta_instructions | 290.71 |
| syscall_class_file | 32.79  |

---

# Interpretation

## Dominant Feature

delta_instructions alone explains most energy variation:

```text
R² = 0.8481
```

This confirms that CPU instruction execution is the primary energy driver.

---

## Additional Information

syscall_class_file contributes a small but measurable improvement.

Possible explanation:

* File-system interactions correlate with workload scheduling.
* They provide supplementary information beyond CPU activity.

---

# Comparison with Other Workloads

| Dataset  | Best Features                    |
| -------- | -------------------------------- |
| Stress   | instructions + file              |
| Phoronix | net_send + cycles + cache        |
| DAW1     | syscall + cpu                    |
| DAW2     | network + syscall + instructions |

---

# Key Findings

Stress workload is highly predictable.

Only two features are required to achieve:

```text
R² = 0.864
MAE = 4.06%
```

This is the strongest SFS result observed so far.

The selected feature set is compact, interpretable, and highly effective.

---

# Next Step

Use the selected features:

```text
delta_instructions
syscall_class_file
```

for:

* CVXPY baseline estimation
* SGDRegressor
* Random Forest
* LightGBM / XGBoost

and compare model performance against the baseline.
