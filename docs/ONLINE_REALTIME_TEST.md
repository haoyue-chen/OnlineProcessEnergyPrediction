# Online real-time testing harness

This harness lets you run the **currently approved** model as a live server and
test it with real-time requests streamed from the controlled parquet dataset.

Serving safety is the core guarantee:

- `/predict` serves **only** the model selected by `models/latest_approved.json`
  via `ApprovedPredictor`.
- Pending or rejected candidates under `models/pending/` are **never** loaded.
- If `latest_approved.json` is absent, `/predict` falls back safely to the
  legacy base MoE artifact (`models/moe_linear.pkl`).
- The stream client only calls `/predict`. It never calls `/update`, never loads
  `models/pending/`, and never trains anything.
- Candidate training/evaluation (the `auto_expert` pipeline) remains fully
  separate from serving. Promotion is the **only** way to change the approved
  serving model.
- Labels exported by the stream client can be ingested later by the
  `auto_expert` pipeline for safe offline candidate training/evaluation.

## A. Start the server

```bash
bash scripts/start_approved_server.sh
```

This sets `PORT=8801`, `ONLINE_PRED_LOG=data/online_logs/predictions_live.jsonl`,
and runs `python -m inference.online_server`. It prints the server URL, the
`/info` command, a sample `/predict` curl command, and the prediction log path.

## B. Check the server

```bash
curl -s localhost:8801/info
```

The response reports `model_kind` (should be `approved`), `model_version`,
`model_path`, and `gate_mode` of the served model.

## C. Run the real-time stream test

In a separate terminal:

```bash
python scripts/stream_parquet_to_server.py \
  --data-root data/controlled_feature_moe_more_runs/runs \
  --max-requests 100 \
  --sleep 0.1 \
  --workload-kind all \
  --write-labels
```

Supported flags:

- `--max-requests N` — cap the number of `/predict` requests.
- `--sleep SECONDS` — delay between requests.
- `--workload-kind {cpu,mem,io,net,mixed,gpu-like,all}` — restrict streamed rows.
- `--write-labels` — also write true-energy labels to
  `data/online_logs/stream_labels.jsonl` for later `auto_expert` ingestion.

The client prints per-request progress (index, workload kind, prediction id,
energy, model_version, gate_mode, latency) and saves responses to
`data/online_logs/stream_responses.jsonl`.

## D. Check the logs

```bash
python scripts/check_online_test_logs.py
```

It reads the prediction log, stream responses, and (optionally) labels, then
prints counts, unique `model_version`/`gate_mode` values, average latency,
per-workload counts, and sample entries. It **fails loudly** if any served
`model_path` is under `models/pending/`, if `model_kind` is `pending`, or if
`model_version`/`prediction_id` is missing.

## E. Safety summary

- `/predict` serves only the approved model.
- Pending/rejected candidates are never served.
- Labels can be used later by the `auto_expert` pipeline.
- Candidate training/evaluation remains separate from serving.
- Promotion is the only way to change the approved serving model.
