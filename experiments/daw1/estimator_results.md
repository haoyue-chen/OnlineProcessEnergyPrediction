
# Baseline Estimator Results

## Configuration

Model:
- Linear regression baseline
- CVXPY optimization
- L1-regularized least squares

Dataset:
- DAW scientific workflow

Selected Features:
- syscall_class_other
- delta_cpu_ns
- syscall_class_signal

---

# Final Metrics

| Metric | Value |
|---|---|
| R² | 0.7074 |
| MAE (%) | 6.07 |
| Static Energy | 250.556 |

---

# Learned Feature Weights

| Feature | Weight |
|---|---|
| syscall_class_other | 127.95 |
| delta_cpu_ns | 62.28 |
| syscall_class_signal | 42.79 |

---

# Interpretation

- Baseline linear regression achieves moderate-to-good performance.
- System-call activity dominates energy prediction.
- CPU time remains an important predictor.
- Hardware performance counters contribute little additional information.

---

# Limitations

The linear baseline still struggles with:

- workload transitions
- dynamic behavior
- phase changes
- concept drift

This motivates exploration of:

- online learning
- adaptive models
- Mixture-of-Experts (MoE)
- nonlinear regressors

---

# Next Steps

Planned experiments:

1. Compare multiple ML regressors
   - SGDRegressor
   - Random Forest
   - XGBoost / LightGBM

2. Evaluate online learning behavior

3. Analyze workload-dependent prediction quality

4. Explore Mixture-of-Experts routing strategies