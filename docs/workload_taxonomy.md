# Workload Taxonomy

## Objective

Construct a multi-level workload dataset for process-level energy estimation.

The selected workloads cover three levels of complexity:

```text
Synthetic
    ↓
Benchmark
    ↓
Scientific Workflow
```

---

## Workload Overview

| Dataset | Category | Description | Characteristics |
|----------|----------|----------|----------|
| Stress | Synthetic Workload | stress-ng benchmark | CPU / Memory / Disk intensive |
| Phoronix | Benchmark Suite | Phoronix Test Suite | Mixed benchmark workload |
| DAW | Scientific Workflow | Bioinformatics workflow (RNA-seq) | Real-world data-intensive workflow |

---

## Workload Characteristics

### Stress

Focus:

- CPU stress
- Memory stress
- Disk stress
- System call stress

Characteristics:

- Highly controlled
- Repeatable
- Low workload diversity

Expected Modeling Difficulty:

Low

---

### Phoronix

Focus:

- 7zip compression
- OpenSSL encryption
- STREAM benchmark

Characteristics:

- CPU-intensive
- Memory-intensive
- Mixed benchmark phases

Expected Modeling Difficulty:

Medium

---

### DAW

Focus:

- RNA-seq workflow execution
- Scientific data processing

Characteristics:

- Workflow phases
- Idle periods
- Communication overhead
- Dynamic resource utilization

Expected Modeling Difficulty:

High

---

## Research Motivation

Different workload categories exhibit different relationships between:

- process metrics
- system behavior
- energy consumption

Therefore, workload-aware energy modeling is required.