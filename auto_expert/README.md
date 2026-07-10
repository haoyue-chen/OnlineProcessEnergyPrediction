# auto_expert — Safe Online Learning with Automatic Expert Expansion

A safe, fully-automatic closed loop for online model updates. It never edits the
live model directly — every change is a *candidate* that must pass safety checks
and beat the current approved model by a margin before promotion, with rollback
if it later degrades.

```
online prediction logging
  → label ingestion (true_energy back-fill)
  → training buffer
  → trigger (enough labels / drift / new feature)
  → train candidate (retrain OR expanded-expert)
  → evaluate candidate vs approved (R²/MAE/RMSE/worst-case + NaN/explosion/neg-R² checks)
  → auto promote / reject
  → rollback if degraded
```

Automatic expert expansion is one *candidate type*: when a new resource dimension
(e.g. GPU-like features) is detected, a candidate with a new expert is generated,
trained, and only promoted if it actually helps.

## Files (step 13)

| File | Role |
|---|---|
| `logging.py` | append every prediction to `data/online_logs/predictions.jsonl` (request_id-keyed) |
| `label_ingestion.py` | match `true_energy` to request_id → `labeled_samples.jsonl`; unknown ids logged, never crash |
| `buffer.py` | labeled samples → `data/training_buffer/buffer.parquet` (X, y, meta) |
| `discovery.py` | detect new features / new resource group / OOD / drift; `pending_data` if insufficient |
| `expand.py` | build expanded `RESOURCE_GROUPS` + `CandidateSpec` (retrain or expanded) |
| `train_candidate.py` | train experts + learned gate per spec; save under `models/pending/<version>/` |
| `evaluate_candidate.py` | compare candidate vs approved; safety + expanded-expert usage checks |
| `promote.py` | decide approve/reject/rollback; persist approved to `models/approved/`, log to `results/` |
| `pipeline.py` | orchestrate one cycle: detect → train → evaluate → promote/reject |
| `demo.py` | end-to-end demo (no live server needed) |

## How to run (answers 2–8)

### 2. Online logging demo
The serve-online server logs predictions when `ONLINE_PRED_LOG` is set:
```sh
docker run -d -p 8800:8800 -e ONLINE_PRED_LOG=/app/data/online_logs/predictions.jsonl \
  -v "$PWD/data:/app/data" energy-offloading:v1.2 serve-online
curl -X POST localhost:8800/predict -H 'Content-Type: application/json' \
  -d '{"features":{"delta_cpu_ns":5e8}}'
# -> logged to data/online_logs/predictions.jsonl
```
Without the env var, behavior is unchanged (no logging).

### 3. Back-fill labels
```sh
PYTHONPATH=. python -m auto_expert.label_ingestion \
  # (programmatic API — see demo.py; pass [{request_id, true_energy}, ...])
```
Or via the `ingest_labels([...])` function. Output: `data/online_logs/labeled_samples.jsonl`; unmatched ids → `label_errors.jsonl`.

### 4. Trigger automatic model update
```sh
PYTHONPATH=. python -m auto_expert.pipeline          # one cycle, real buffer
```
Triggers when ≥100 labeled samples exist; otherwise `status=pending_data`, no training.

### 5. Simulate a new resource group
```sh
PYTHONPATH=. python -m auto_expert.pipeline --demo   # injects synthetic delta_gpu_* features
```
`discovery` detects the `gpu` group → an expanded candidate (5 experts) is generated.

### 6. View candidate evaluation
```sh
cat results/online_learning_update_log.jsonl   # every approve/reject/rollback decision
cat results/auto_expert_expansion_log.jsonl    # cycle summaries + discovery
ls models/pending/                             # candidate artifacts (rejected ones kept)
```

### 7. Confirm auto approve / reject
```sh
cat models/latest_approved.json   # points at the current approved model (or absent = none promoted)
```
Each decision in `online_learning_update_log.jsonl` has `status` = approved/rejected and a `reason` (first_baseline / mae_improved / r2_improved / no_margin / safety_check_failed / ...).

### 8. Rollback
Rollback is automatic on degradation (`promote.rollback`). To simulate/inspect:
```sh
PYTHONPATH=. python -c "from auto_expert import promote as p; print(p.rollback('v_bad','v_good','mae_degraded_15pct'))"
```
The failed model is NOT deleted (kept under `models/approved/<failed_version>/` for analysis).

## One-shot end-to-end demo
```sh
PYTHONPATH=. python -m auto_expert.demo
```
Runs the full loop on simulated data: log 120 predictions → back-fill labels → build buffer → detect synthetic GPU dimension → train retrain + expanded candidates → auto-promote/reject → print decision log + artifacts. Verified flow: retrain approved as first baseline, expanded (gpu expert, weight 0.45) approved on MAE improvement.

## Safety design (honest)
- A candidate is promoted ONLY if: no NaN, no explosion, R²≥−1, AND (MAE −5% or R² +0.02) without worsening RMSE/worst-case. Expanded candidates must additionally show the new expert gets non-trivial weight.
- The first cycle has no baseline to beat → a safe candidate becomes the seed baseline (`first_baseline`); every later candidate must then win by margin.
- Insufficient data → `pending_data`, no training, no promotion.
- Rollback restores the previous approved model; the failed one is retained.

## MVP scope (honest)
- The expanded-expert demo uses a *synthetic* GPU feature group to exercise the plumbing end-to-end. It proves the loop (detect → candidate → train → evaluate → auto decide → log) works. It does NOT prove a GPU expert improves real accuracy — that needs real GPU metrics, which aren't in the current dataset.
- Candidates use the robust ResourceMoE (RF experts + global learned gate), not the unstable torch per-sample gate, so evaluation is stable.
- `serve-online` still serves the workload-grouped online MoE; wiring it to load `latest_approved.json` instead of the baked-in artifact is the remaining integration step.
