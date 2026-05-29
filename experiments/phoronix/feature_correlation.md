# Phoronix Workload - Feature Correlation Analysis

## Dataset Overview

Dataset:

* Phoronix Benchmark Suite

Workload Type:

* Synthetic benchmark workload
* CPU-intensive and memory-intensive benchmarks
* Includes workloads such as:

  * 7-Zip Compression
  * OpenSSL Cryptography
  * STREAM Memory Benchmark

Statistics:

| Metric               | Value     |
| -------------------- | --------- |
| Intervals            | 6689      |
| Process-level Rows   | 1,412,818 |
| Mean Interval Energy | 350.98 Wh |
| Std Energy           | 29.26 Wh  |
| Max Energy           | 993.54 Wh |

---

# Pearson Correlation with Interval Energy

| Feature               | Correlation |
| --------------------- | ----------- |
| delta_cpu_time_psutil | 0.845       |
| delta_cycles          | 0.794       |
| delta_cpu_ns          | 0.777       |
| delta_net_send_bytes  | 0.730       |
| syscall_class_file    | 0.675       |
| delta_instructions    | 0.557       |
| delta_cache_misses    | 0.400       |
| context_switches      | 0.399       |
| syscall_class_other   | 0.390       |

Negative correlations:

| Feature               | Correlation |
| --------------------- | ----------- |
| syscall_class_network | -0.101      |
| syscall_class_time    | -0.047      |
| syscall_class_signal  | -0.033      |

## Observations

* CPU execution metrics dominate energy prediction.
* Hardware performance counters show strong linear relationships with energy consumption.
* Network-related system calls exhibit weak negative correlation.
* Benchmark behavior is primarily compute-driven rather than communication-driven.
* Compared to DAW workloads, operating-system interactions contribute less predictive information.

---

# Spearman Correlation

| Feature               | Correlation |
| --------------------- | ----------- |
| delta_net_send_bytes  | 0.598       |
| context_switches      | 0.579       |
| syscall_class_file    | 0.577       |
| delta_cache_misses    | 0.568       |
| syscall_class_other   | 0.540       |
| delta_cpu_time_psutil | 0.499       |
| delta_cpu_time_proc   | 0.487       |
| syscall_count         | 0.436       |

Negative correlations:

| Feature               | Correlation |
| --------------------- | ----------- |
| syscall_class_network | -0.273      |
| delta_instructions    | -0.130      |
| syscall_class_process | -0.034      |

## Observations

* Several features exhibit strong monotonic relationships with energy usage.
* Spearman values are generally lower than those observed in DAW workflows.
* This suggests a more deterministic workload structure dominated by benchmark execution phases.
* Some CPU-related features remain important, but their relationships are not perfectly monotonic.

---

# Lag Correlation Analysis

Top predictors of future energy consumption:

## Lag 1

| Feature               | Correlation |
| --------------------- | ----------- |
| delta_cpu_time_psutil | 0.814       |
| delta_cycles          | 0.762       |
| delta_cpu_ns          | 0.755       |
| delta_net_send_bytes  | 0.693       |

## Lag 5

| Feature               | Correlation |
| --------------------- | ----------- |
| delta_net_send_bytes  | 0.638       |
| syscall_class_file    | 0.576       |
| delta_cpu_time_psutil | 0.561       |
| delta_cycles          | 0.529       |

## Lag 10

| Feature               | Correlation |
| --------------------- | ----------- |
| delta_net_send_bytes  | 0.614       |
| syscall_class_file    | 0.561       |
| delta_cpu_time_psutil | 0.540       |
| delta_cpu_ns          | 0.512       |

## Interpretation

The benchmark exhibits strong temporal persistence:

* CPU activity remains predictive across multiple future intervals.
* Benchmark execution phases persist over time.
* Energy consumption evolves gradually rather than abruptly.
* Online learning approaches may benefit from exploiting this temporal continuity.

---

# Comparison with DAW Workloads

| Property                | DAW1          | DAW2              | Phoronix                |
| ----------------------- | ------------- | ----------------- | ----------------------- |
| Dominant Features       | syscall + CPU | network + syscall | CPU + hardware counters |
| Communication Impact    | Moderate      | High              | Low                     |
| System-call Impact      | High          | High              | Moderate                |
| CPU Impact              | Moderate      | Moderate          | Very High               |
| Hardware Counter Impact | Low           | Moderate          | High                    |

---

# Conclusions

The Phoronix benchmark suite is fundamentally different from the DAW scientific workflows.

Key findings:

* Energy consumption is primarily driven by CPU execution activity.
* Hardware performance counters provide valuable predictive information.
* Communication-related features contribute little to energy prediction.
* Workload behavior is more stable and deterministic than workflow-based workloads.
* Strong temporal continuity exists across benchmark execution intervals.

Most informative feature categories:

1. CPU execution metrics

   * delta_cpu_time_psutil
   * delta_cpu_ns
   * delta_cycles

2. Hardware performance counters

   * delta_instructions
   * delta_cache_misses

3. File-system activity

   * syscall_class_file

These observations suggest that workload-aware feature selection is necessary and support the hypothesis that different workload classes require different energy modeling strategies.
