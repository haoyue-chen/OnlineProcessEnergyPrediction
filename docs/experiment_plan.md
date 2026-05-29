# Experiment Plan

## Research Objectives

### MO1

Improve process-level energy estimation accuracy using machine-learning models.

### MO2

Determine under which workloads online learning provides the largest benefits.

### MO3

Investigate whether Mixture-of-Experts (MoE) can adapt to workload changes.

---

# Phase 1: Baseline Reproduction

Status:

Completed

Tasks:

- Dataset generation
- Feature analysis
- SFS feature selection
- CVXPY estimator evaluation

---

# Phase 2: Machine Learning Models

Objective:

Compare standard machine-learning regressors against the baseline.

Models:

- SGDRegressor
- RandomForestRegressor
- LightGBM

Evaluation:

- R²
- MAE
- Training Time

Results Table:

| Dataset | Baseline | SGD | RF | LightGBM |
|----------|----------|----------|----------|----------|
| Stress | 0.864 | | | |
| Phoronix | 0.730 | | | |
| DAW1 | 0.707 | | | |
| DAW2 | 0.846 | | | |

---

# Phase 3: Online Learning

Objective:

Evaluate model adaptation under changing workloads.

Candidate:

- SGDRegressor (partial_fit)

Metrics:

- Online MAE
- Online R²
- Adaptation Speed

---

# Phase 4: Mixture of Experts

Objective:

Select specialized models for different workload phases.

Architecture:

Workload Classifier
    ↓
Expert Selection
    ↓
Prediction

Experts:

- SGDRegressor
- Hoeffding Tree
- Passive-Aggressive Regressor

Strategies:

1. Winner-Take-All
2. Softmax Interpolation

---

# Expected Outcome

Determine:

1. Which ML model performs best.

2. Which workload benefits most from online learning.

3. Whether workload-aware expert routing improves energy estimation accuracy.