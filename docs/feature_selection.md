# Feature Selection Analysis

## Objective

Identify which process-level metrics are most useful for predicting interval energy consumption.

Methods:

1. Pearson Correlation
2. Spearman Correlation
3. Sequential Forward Selection (SFS)

---

## DAW Workload

Top Features:

- syscall_class_other
- delta_cpu_ns
- syscall_class_signal

Observation:

- System-call activity is highly correlated with energy consumption.
- CPU scheduling behavior contributes significantly.
- Workflow communication introduces nonlinear effects.

---

## Phoronix Workload

Top Features:

- delta_net_send_bytes
- delta_cycles
- delta_cache_misses

Observation:

- Hardware performance counters dominate.
- Benchmark phases exhibit strong compute behavior.
- Network activity correlates with workload execution.

---

## Stress Workload

Top Features:

- delta_instructions
- syscall_class_file

Observation:

- CPU instruction count explains most energy variation.
- Workload behavior is highly compute-driven.
- Only a small number of features are required.

---

## Cross-Workload Findings

Feature importance differs significantly across workloads.

| Dataset | Dominant Features |
|----------|----------|
| Stress | CPU Instructions |
| Phoronix | Hardware Counters |
| DAW | Syscalls + CPU Activity |

---

## Conclusion

The relationship between process metrics and energy consumption is workload-dependent.

This suggests that adaptive or workload-aware modeling approaches may outperform static models.