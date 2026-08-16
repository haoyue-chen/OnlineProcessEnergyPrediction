# MoE → Inference Artifact Export

Bridges the Phase-2 MoE predictor into the monitor's inference service.

The monitor's `InferenceRequest` (`monitor/inference/api.py`) historically loaded a
flat linear artifact (`features` + `weights` + `scaler` + `static_energy`). It now
also accepts a **MoE artifact** — detected by `model_type == "moe"` — and routes
each sample through the gate to its per-workload expert. A MoE model is therefore a
drop-in replacement: pass it via `--model-pkl`.

## Why a separate export step

The monitor inference container must **not** depend on the `moe` training package.
`export_moe.py` unpacks a fitted `MixtureOfExperts` into raw scikit-learn objects
(the gate classifier, the per-label expert regressors, the feature order, the
labels) and pickles just those. The artifact loads with **only scikit-learn**
present — verified by loading it with the `moe` package removed from `sys.path`.

## Artifact schema (`model_type="moe"`)

| Key | Meaning |
|---|---|
| `model_type` | `"moe"` — tells `InferenceRequest` to use gate routing |
| `features` | feature order the gate/experts expect |
| `labels` | workload labels (one expert each) |
| `gate` | sklearn classifier: features → workload label |
| `experts` | `{label: sklearn regressor}` |
| `expert_class` | `"linear"` or `"rf"` |
| `source_workloads` | workloads the model was trained on |

## Build

```sh
# from the repository root
python -m moe_export.export_moe --expert linear --out models/moe_linear.pkl
python -m moe_export.export_moe --expert rf     --out models/moe_rf.pkl
```

## Use in the monitor

```sh
# anywhere the monitor takes --model-pkl, point it at the MoE artifact:
python -m monitor ... --model-pkl monitor/models/moe_linear.pkl
```

`InferenceRequest.predict_sample` / `predict_many` then gate-route automatically;
the legacy linear artifacts keep working unchanged.
