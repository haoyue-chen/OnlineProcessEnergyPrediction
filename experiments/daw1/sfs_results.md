# Sequential Forward Selection (SFS) Results

## Dataset

- Dataset: DAW scientific workflow
- Train intervals: 4640
- Test intervals: 1160

---

# Candidate Features

- delta_cpu_ns
- delta_cycles
- delta_instructions
- delta_cache_misses
- delta_branch_instructions
- delta_io_bytes
- delta_net_send_bytes
- context_switches
- syscall_count
- delta_rss_memory
- syscall_class_file
- syscall_class_network
- syscall_class_memory
- syscall_class_process
- syscall_class_other
- syscall_class_sched
- syscall_class_signal
- syscall_class_time

---

# SFS Selection Process

| Step | Added Feature | R² | MAE (%) |
|---|---|---|---|
| 1 | syscall_class_other | 0.5560 | 7.22 |
| 2 | delta_cpu_ns | 0.6965 | 6.16 |
| 3 | syscall_class_signal | 0.7074 | 6.07 |

---

# Best Feature Combination

```python
[
    "syscall_class_other",
    "delta_cpu_ns",
    "syscall_class_signal"
]

