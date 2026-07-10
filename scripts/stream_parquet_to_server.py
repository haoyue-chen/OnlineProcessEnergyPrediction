#!/usr/bin/env python3
"""Real-time stream test client for the approved-model online server.

Reads controlled process_interval_data.parquet rows and sends each one as a
POST /predict request to the running online server. It ONLY calls /predict — it
never calls /update, never loads models/pending/, and never trains anything.

Optionally writes true-energy labels (data/online_logs/stream_labels.jsonl) so
the auto_expert pipeline can ingest them later for safe offline candidate
training/evaluation. This script itself does not train or promote.

Usage:
    python scripts/stream_parquet_to_server.py \
        --data-root data/controlled_feature_moe_more_runs/runs \
        --max-requests 100 \
        --sleep 0.1 \
        --workload-kind all \
        --write-labels
"""
from __future__ import annotations

import argparse
import json
import time
import urllib.request
from pathlib import Path

import pandas as pd

from moe.data import FEATURES, TARGET

WORKLOAD_KINDS = ["cpu", "mem", "io", "net", "mixed", "gpu-like"]
DEFAULT_DATA_ROOT = "data/controlled_feature_moe_more_runs/runs"
DEFAULT_SERVER_URL = "http://localhost:8801"
RESPONSES_LOG = Path("data/online_logs/stream_responses.jsonl")
LABELS_LOG = Path("data/online_logs/stream_labels.jsonl")


def _run_kind(run_name: str) -> str:
    if run_name.startswith("controlled-gpu-like-"):
        return "gpu-like"
    parts = run_name.split("-")
    return parts[1] if len(parts) >= 3 else "unknown"


def _collect_rows(data_root: Path, workload_kind: str) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(p for p in data_root.iterdir() if p.is_dir()):
        kind = _run_kind(run_dir.name)
        if workload_kind != "all" and kind != workload_kind:
            continue
        parquet_path = run_dir / "datasets" / "process_interval_data.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        for idx, row in enumerate(df.itertuples(index=False)):
            rec = row._asdict()
            features = {}
            for f in FEATURES:
                value = rec.get(f, 0.0)
                if value is None or pd.isna(value):
                    value = 0.0
                features[f] = float(value)
            true_energy = rec.get(TARGET, None)
            if true_energy is not None and pd.isna(true_energy):
                true_energy = None
            rows.append({
                "features": features,
                "workload_kind": kind,
                "source_run": run_dir.name,
                "source_row_index": idx,
                "true_energy": float(true_energy) if true_energy is not None else None,
            })
    return rows


def _round_robin_rows(rows: list[dict], max_requests: int) -> list[dict]:
    """When workload_kind=all, interleave available kinds roughly evenly.

    A 60-request run over 6 available kinds should produce roughly 10 requests per
    kind by taking one row from each non-empty kind in turn until max_requests is
    reached or all buckets are exhausted.
    """
    buckets: dict[str, list[dict]] = {k: [] for k in WORKLOAD_KINDS}
    for row in rows:
        kind = row.get("workload_kind")
        if kind in buckets:
            buckets[kind].append(row)

    ordered: list[dict] = []
    idx = 0
    while len(ordered) < max_requests:
        progressed = False
        for kind in WORKLOAD_KINDS:
            bucket = buckets[kind]
            if idx < len(bucket):
                ordered.append(bucket[idx])
                progressed = True
                if len(ordered) >= max_requests:
                    break
        if not progressed:
            break
        idx += 1
    return ordered


def _post_predict(server_url: str, features: dict) -> tuple[dict, float]:
    payload = json.dumps({"features": features}).encode()
    req = urllib.request.Request(
        f"{server_url.rstrip('/')}/predict",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=30) as resp:
        body = json.loads(resp.read().decode())
    return body, (time.time() - t0) * 1000.0


def main() -> None:
    ap = argparse.ArgumentParser(description="Stream controlled parquet rows to the online /predict API")
    ap.add_argument("--data-root", default=DEFAULT_DATA_ROOT, help="Controlled runs root directory")
    ap.add_argument("--server-url", default=DEFAULT_SERVER_URL, help="Online server base URL")
    ap.add_argument("--max-requests", type=int, default=100, help="Max number of /predict requests to send")
    ap.add_argument("--sleep", type=float, default=0.1, help="Seconds to sleep between requests")
    ap.add_argument("--workload-kind", default="all",
                    choices=["all", *WORKLOAD_KINDS], help="Restrict streamed rows to one workload kind")
    ap.add_argument("--write-labels", action="store_true",
                    help="Write true_energy labels to stream_labels.jsonl for later auto_expert ingestion")
    args = ap.parse_args()

    data_root = Path(args.data_root)
    if not data_root.exists():
        raise SystemExit(f"data root not found: {data_root}")

    rows = _collect_rows(data_root, args.workload_kind)
    if not rows:
        raise SystemExit(f"no rows found for workload_kind={args.workload_kind} under {data_root}")
    if args.workload_kind == "all":
        rows = _round_robin_rows(rows, args.max_requests)
    else:
        rows = rows[: args.max_requests]

    RESPONSES_LOG.parent.mkdir(parents=True, exist_ok=True)
    if args.write_labels:
        LABELS_LOG.parent.mkdir(parents=True, exist_ok=True)

    n_sent = 0
    latency_ms_list: list[float] = []
    with RESPONSES_LOG.open("a") as rf:
        lf = LABELS_LOG.open("a") if args.write_labels else None
        try:
            for i, row in enumerate(rows):
                resp, latency_ms = _post_predict(args.server_url, row["features"])
                latency_ms_list.append(latency_ms)
                n_sent += 1
                rec = {
                    "request_index": i,
                    "workload_kind": row["workload_kind"],
                    "prediction_id": resp.get("prediction_id"),
                    "energy_wh": resp.get("energy_wh"),
                    "model_version": resp.get("model_version"),
                    "model_kind": resp.get("model_kind"),
                    "model_path": resp.get("model_path"),
                    "gate_mode": resp.get("gate_mode"),
                    "expert": resp.get("expert"),
                    "latency_ms": round(latency_ms, 3),
                    "source_run": row["source_run"],
                    "source_row_index": row["source_row_index"],
                }
                rf.write(json.dumps(rec) + "\n")
                rf.flush()
                if lf is not None and row["true_energy"] is not None:
                    label_rec = {
                        "request_id": resp.get("prediction_id"),
                        "prediction_id": resp.get("prediction_id"),
                        "true_energy": row["true_energy"],
                        "workload_kind": row["workload_kind"],
                        "source_run": row["source_run"],
                        "source_row_index": row["source_row_index"],
                        "model_version": resp.get("model_version"),
                    }
                    lf.write(json.dumps(label_rec) + "\n")
                    lf.flush()
                print(
                    f"[{i+1}/{len(rows)}] kind={row['workload_kind']:8s} "
                    f"pid={resp.get('prediction_id','')[:8]} "
                    f"energy_wh={resp.get('energy_wh')} "
                    f"ver={resp.get('model_version')} "
                    f"gate={resp.get('gate_mode')} "
                    f"latency={latency_ms:.1f}ms"
                )
                if args.sleep > 0:
                    time.sleep(args.sleep)
        finally:
            if lf is not None:
                lf.close()

    print(f"\nstreamed {n_sent} requests")
    print(f"avg latency = {sum(latency_ms_list)/len(latency_ms_list):.1f} ms")
    print(f"responses log: {RESPONSES_LOG}")
    if args.write_labels:
        print(f"labels log:    {LABELS_LOG}")


if __name__ == "__main__":
    main()
