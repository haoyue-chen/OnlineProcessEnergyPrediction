# Baseline Estimator Results

## Objective

Reproduce the original CVXPY-based energy estimator provided by the project repository.

Model:

- Linear Regression
- CVXPY Optimization
- L1-Regularized Least Squares

Target:

interval_energy

---

## Results Summary

| Dataset | Best Features | R² | MAE (%) |
|----------|----------|----------|----------|
| DAW1 | syscall + cpu + signal | 0.707 | 6.07 |
| DAW2 | network + syscall + instruction | 0.846 | 3.79 |
| Phoronix | network + cycles + cache | 0.730 | 5.13 |
| Stress | instructions + file | 0.864 | 4.06 |

---

## Observations

### Stress

Highest prediction quality.

Reasons:

- Stable workload
- Strong CPU dominance
- Low behavioral complexity

---

### Phoronix

Moderate prediction quality.

Reasons:

- Mixed benchmark phases
- Multiple resource types
- Hardware counter dependence

---

### DAW

Most challenging workload.

Reasons:

- Workflow transitions
- Communication overhead
- Dynamic execution phases

---

## Key Findings

1. Baseline estimator successfully reproduces repository results.

2. Prediction quality varies significantly across workload types.

3. Different workloads require different feature sets.

4. Linear modeling has limitations when workload behavior changes over time.

---

## Conclusion

The baseline establishes a strong reference point for evaluating machine-learning approaches.