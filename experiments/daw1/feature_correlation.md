# DAW Workload - Feature Correlation Analysis

## Dataset Overview

- Dataset: DAW scientific workflow
- Workload type: Data-intensive workflow
- Intervals: 5800
- Process-level rows: 3,307,621

---

# Pearson Correlation with Interval Energy

| Feature | Pearson Correlation |
|---|---|
| syscall_class_other | 0.7277 |
| delta_cpu_ns | 0.6809 |
| syscall_count | 0.6779 |
| context_switches | 0.6096 |
| syscall_class_signal | 0.5752 |
| syscall_class_file | 0.5500 |
| delta_cpu_time_proc | 0.5494 |

## Observations

- CPU-related metrics show strong correlation with interval energy.
- System-call related metrics are highly important for DAW workflows.
- Hardware counter features (`delta_instructions`, `delta_cycles`) show weak correlation.
- DAW workload behavior appears syscall-heavy rather than purely compute-heavy.

---

# Spearman Correlation

| Feature | Spearman Correlation |
|---|---|
| delta_cpu_ns | 0.8631 |
| context_switches | 0.8069 |
| syscall_count | 0.8034 |
| syscall_class_other | 0.7999 |
| syscall_class_file | 0.7949 |

## Observations

- Strong monotonic relationship exists between CPU activity and energy usage.
- Nonlinear behavior may exist in workload transitions.
- Spearman correlation is consistently higher than Pearson correlation, suggesting nonlinear effects.

---

# Conclusion

The DAW scientific workflow is dominated by:

- CPU scheduling activity
- Process communication
- System-call behavior

This suggests that nonlinear or adaptive models may outperform static linear regression for workflow energy estimation.