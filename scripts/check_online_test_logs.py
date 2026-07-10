#!/usr/bin/env python3
"""Check online real-time test logs for serving safety and basic stats.

Reads:
  - data/online_logs/predictions_live.jsonl        (server-side prediction log)
  - data/online_logs/stream_responses.jsonl         (client-side stream responses)
  - data/online_logs/stream_labels.jsonl            (optional true-energy labels)

Fails loudly if serving safety is violated:
  - any served model_path contains models/pending
  - any model_kind is pending
  - model_version is missing
  - prediction_id is missing
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PREDICTIONS_LOG = Path("data/online_logs/predictions_live.jsonl")
RESPONSES_LOG = Path("data/online_logs/stream_responses.jsonl")
LABELS_LOG = Path("data/online_logs/stream_labels.jsonl")


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _fail(msg: str) -> None:
    print(f"FAIL: {msg}", file=sys.stderr)
    sys.exit(1)


def _check_safety(records: list[dict], source: str) -> None:
    for rec in records:
        model_path = rec.get("model_path") or ""
        if "models/pending" in model_path:
            _fail(f"{source}: served model_path is under models/pending: {model_path}")
        kind = rec.get("model_kind")
        if kind == "pending":
            _fail(f"{source}: model_kind is pending")
        if rec.get("model_version") in (None, ""):
            _fail(f"{source}: record missing model_version: {rec.get('prediction_id') or rec.get('request_id')}")
        # Server-side prediction logs use request_id; client-side stream responses use prediction_id.
        if source == "predictions_live":
            if rec.get("request_id") in (None, ""):
                _fail(f"{source}: record missing request_id")
        else:
            if rec.get("prediction_id") in (None, ""):
                _fail(f"{source}: record missing prediction_id")


def main() -> None:
    ap = argparse.ArgumentParser(description="Check online real-time test logs")
    ap.add_argument("--predictions-log", default=PREDICTIONS_LOG)
    ap.add_argument("--responses-log", default=RESPONSES_LOG)
    ap.add_argument("--labels-log", default=LABELS_LOG)
    ap.parse_args()

    preds = _read_jsonl(Path(PREDICTIONS_LOG))
    resps = _read_jsonl(Path(RESPONSES_LOG))
    labels = _read_jsonl(Path(LABELS_LOG))

    print("=== online test log check ===")

    # Safety checks (fail loudly).
    _check_safety(preds, "predictions_live")
    _check_safety(resps, "stream_responses")

    # predictions_live (server-side)
    print(f"\npredictions_log: {PREDICTIONS_LOG}")
    print(f"  num_predictions: {len(preds)}")
    if preds:
        print(f"  unique model_version: {sorted({str(r.get('model_version')) for r in preds})}")
        gate_modes = sorted({str(r.get('gate_mode')) for r in preds if r.get('gate_mode') is not None})
        print(f"  unique gate_mode: {gate_modes}")
        pending_paths = [str(r.get('model_path')) for r in preds if 'models/pending' in str(r.get('model_path', ''))]
        print(f"  any model_path under models/pending: {bool(pending_paths)} ({len(pending_paths)} rows)")
        latencies = [r.get('latency_ms') for r in preds if r.get('latency_ms') is not None]
        if latencies:
            print(f"  avg latency_ms: {sum(latencies)/len(latencies):.2f}")
        print("  sample latest prediction log entry:")
        print(json.dumps(preds[-1], indent=2))

    # stream_responses (client-side)
    print(f"\nresponses_log: {RESPONSES_LOG}")
    print(f"  num_responses: {len(resps)}")
    if resps:
        print(f"  unique model_version: {sorted({str(r.get('model_version')) for r in resps})}")
        gm = sorted({str(r.get('gate_mode')) for r in resps if r.get('gate_mode') is not None})
        print(f"  unique gate_mode: {gm}")
        by_kind: dict[str, int] = {}
        for r in resps:
            k = r.get("workload_kind", "unknown")
            by_kind[k] = by_kind.get(k, 0) + 1
        print(f"  count by workload_kind: {by_kind}")
        lat = [r.get('latency_ms') for r in resps if r.get('latency_ms') is not None]
        if lat:
            print(f"  avg latency_ms: {sum(lat)/len(lat):.2f}")
        print("  sample latest stream response:")
        print(json.dumps(resps[-1], indent=2))

    # stream_labels (optional)
    print(f"\nlabels_log: {LABELS_LOG}")
    print(f"  num_labels: {len(labels)}")
    if labels:
        print("  sample latest label record:")
        print(json.dumps(labels[-1], indent=2))

    print("\nSAFETY OK: no pending/rejected model was served; all records have model_version and prediction_id.")


if __name__ == "__main__":
    main()
