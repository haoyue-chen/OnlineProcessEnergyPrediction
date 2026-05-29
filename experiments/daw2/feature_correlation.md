# DAW Dataset 2 - Feature Correlation Analysis

## Dataset Overview

Dataset: DAW Scientific Workflow (Dataset 2)

Workload Type:

* Data-intensive scientific workflow
* nf-core/rnaseq pipeline
* Workflow-based execution with alternating compute and communication phases

Statistics:

| Metric               | Value     |
| -------------------- | --------- |
| Intervals            | 6131      |
| Process Rows         | 2,663,509 |
| Mean Interval Energy | 302.11 Wh |
| Std Energy           | 38.92 Wh  |
| Max Energy           | 714.77 Wh |

---

# Pearson Correlation with Interval Energy

| Feature                   | Correlation |
| ------------------------- | ----------- |
| delta_net_send_bytes      | 0.837       |
| syscall_class_other       | 0.768       |
| delta_instructions        | 0.706       |
| syscall_count             | 0.705       |
| delta_cycles              | 0.705       |
| delta_cpu_ns              | 0.692       |
| delta_cpu_time_psutil     | 0.679       |
| syscall_class_process     | 0.658       |
| delta_branch_instructions | 0.651       |
| context_switches          | 0.625       |

## Observations

* Network activity is the strongest predictor of energy consumption.
* System-call activity remains highly important.
* CPU execution metrics show strong linear relationships.
* Hardware counters become much more informative than in the previous DAW dataset.
* This workload appears more communication-intensive than the first DAW workload.

---

# Spearman Correlation

| Feature                   | Correlation |
| ------------------------- | ----------- |
| delta_cycles              | 0.867       |
| delta_cpu_ns              | 0.852       |
| syscall_class_other       | 0.846       |
| context_switches          | 0.846       |
| syscall_class_file        | 0.846       |
| syscall_count             | 0.846       |
| delta_cpu_time_psutil     | 0.845       |
| delta_branch_instructions | 0.845       |
| delta_instructions        | 0.842       |

## Observations

* Strong monotonic relationships exist between workload activity and energy usage.
* CPU-related metrics dominate energy behavior.
* Correlation values are consistently higher than Pearson values.
* This suggests nonlinear workload-energy relationships.
* Adaptive or nonlinear models may outperform static linear regression.

---

# Lag Correlation Analysis

Top predictors of future energy consumption:

| Feature              | Future Correlation |
| -------------------- | ------------------ |
| delta_net_send_bytes | 0.82 - 0.85        |
| syscall_class_other  | 0.70 - 0.77        |
| delta_instructions   | 0.65 - 0.69        |
| syscall_count        | 0.65 - 0.71        |

## Interpretation

The workflow exhibits strong temporal persistence:

* Network traffic predicts future energy demand.
* Workload phases persist across multiple intervals.
* Concept drift occurs gradually rather than abruptly.
* Online learning methods should benefit from this temporal structure.

---

# Conclusions

This workload is characterized by:

* Strong network activity
* High system-call intensity
* Significant CPU utilization
* Persistent workload phases

Compared with the previous DAW dataset:

* Network behavior plays a much larger role.
* Hardware counters provide more useful information.
* Future energy is easier to predict due to stronger temporal continuity.

These observations motivate:

* Online learning
* Adaptive regression
* Workload-aware model selection
* Mixture-of-Experts approaches
