## ML Model Evaluation (LightGBM)

| Dataset | Linear R² | LightGBM R² | Improvement |
|----------|----------|----------|----------|
| DAW1 | 0.707 | 0.931 | +31.7% |
| DAW2 | 0.846 | 0.965 | +14.1% |
| Phoronix | 0.730 | 0.862 | +18.1% |
| Stress | 0.864 | 0.966 | +11.8% |

### Observation

- LightGBM consistently outperforms the linear baseline.
- Improvement is largest for DAW1, suggesting stronger nonlinear relationships.
- Scientific workflows appear to benefit more from nonlinear modeling than synthetic workloads.