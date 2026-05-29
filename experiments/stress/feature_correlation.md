# Stress Workload - Feature Correlation Analysis

## Dataset Overview

Dataset:

* Stress Benchmark

Workload Type:

* Synthetic CPU-intensive stress workload
* High CPU utilization
* Continuous computational pressure

Statistics:

| Metric               | Value     |
| -------------------- | --------- |
| Intervals            | 8501      |
| Process-level Rows   | 1,726,700 |
| Mean Interval Energy | 292.63 Wh |
| Std Energy           | 35.82 Wh  |

---

# Pearson Correlation with Interval Energy

Top correlated features:

| Feature                   | Correlation |
| ------------------------- | ----------- |
| delta_cycles              | 0.9010      |
| delta_instructions        | 0.8977      |
| delta_cpu_time_psutil     | 0.8891      |
| delta_cpu_time_proc       | 0.8505      |
| delta_cpu_ns              | 0.8033      |
| delta_branch_instructions | 0.7823      |
| delta_cache_misses        | 0.4176      |
| delta_io_bytes            | 0.3364      |

Weak correlations:

| Feature             | Correlation |
| ------------------- | ----------- |
| syscall_count       | 0.0760      |
| syscall_class_file  | 0.0898      |
| syscall_class_other | 0.0737      |
| context_switches    | 0.0527      |

Negative correlations:

| Feature          | Correlation |
| ---------------- | ----------- |
| delta_rss_memory | -0.0047     |

---

# Observations

The workload is overwhelmingly CPU-dominated.

The strongest predictors are:

* CPU cycles
* Retired instructions
* CPU execution time
* Branch instructions

System-call activity contributes almost no predictive information.

This differs significantly from DAW workflows where syscall behavior strongly influences energy consumption.

---

# Spearman Correlation

Top monotonic relationships:

| Feature                   | Correlation |
| ------------------------- | ----------- |
| delta_instructions        | 0.9309      |
| delta_cycles              | 0.9309      |
| delta_cpu_time_psutil     | 0.9037      |
| delta_cpu_ns              | 0.8854      |
| delta_cpu_time_proc       | 0.8807      |
| delta_branch_instructions | 0.8544      |

---

# Interpretation

The extremely high Spearman coefficients indicate:

* Stable workload behavior
* Strong monotonic relationship between CPU activity and energy consumption
* Minimal workload phase variation

Stress behaves much more predictably than DAW or Phoronix workloads.

---

# Lag Correlation Analysis

Top predictors of future energy consumption:

## Lag 1

| Feature               | Correlation |
| --------------------- | ----------- |
| delta_cycles          | 0.9008      |
| delta_instructions    | 0.8974      |
| delta_cpu_time_psutil | 0.8891      |
| delta_cpu_time_proc   | 0.8517      |

## Lag 5

| Feature               | Correlation |
| --------------------- | ----------- |
| delta_cycles          | 0.8036      |
| delta_instructions    | 0.7983      |
| delta_cpu_time_psutil | 0.7935      |
| delta_cpu_time_proc   | 0.7640      |

## Lag 10

| Feature               | Correlation |
| --------------------- | ----------- |
| delta_cycles          | 0.6781      |
| delta_instructions    | 0.6731      |
| delta_cpu_time_psutil | 0.6696      |
| delta_cpu_time_proc   | 0.6448      |

---

# Key Findings

Stress workload energy consumption is dominated by:

1. CPU cycles
2. Executed instructions
3. CPU execution time

System-call behavior contributes very little additional information.

This workload represents the simplest energy-modeling scenario among all evaluated datasets.

---

# Research Implications

Stress workload demonstrates that:

* Pure compute workloads are highly predictable.
* Hardware performance counters alone explain most energy variation.
* Complex workload-aware modeling may provide limited additional benefit.

This workload serves as an ideal baseline for evaluating energy estimation methods.
