from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from auto_expert import buffer as train_buffer
from auto_expert import pipeline
from auto_expert import promote
from auto_expert.label_ingestion import DEFAULT_ERRORS, DEFAULT_LABELED, ingest_labels
from auto_expert.logging import DEFAULT_LOG, log_prediction
from inference.predictor import EnergyPredictor
from moe.data import FEATURES, TARGET

GPU_OPTIONAL_FEATURES = [
    "delta_gpu_utilization",
    "delta_gpu_memory",
    "delta_gpu_power",
    "gpu_like_signal",
]

RUNS_ROOT = Path(__file__).resolve().parents[1] / "data" / "controlled_feature_moe_more_runs" / "runs"
LABELS_LOG = Path("data/online_logs/labels.jsonl")


def _run_kind(run_name: str) -> str:
    if run_name.startswith("controlled-gpu-like-"):
        return "gpu-like"
    parts = run_name.split("-")
    return parts[1] if len(parts) >= 3 else "unknown"


def _iter_rows(base_dir: Path) -> list[dict]:
    rows: list[dict] = []
    for run_dir in sorted(p for p in base_dir.iterdir() if p.is_dir()):
        parquet_path = run_dir / "datasets" / "process_interval_data.parquet"
        if not parquet_path.exists():
            continue
        df = pd.read_parquet(parquet_path)
        workload_kind = _run_kind(run_dir.name)
        replay_features = list(FEATURES) + [f for f in GPU_OPTIONAL_FEATURES if f in df.columns]
        for row in df.itertuples(index=False):
            rec = row._asdict()
            features = {}
            for f in replay_features:
                value = rec.get(f, 0.0)
                if value is None or pd.isna(value):
                    value = 0.0
                features[f] = float(value)
            rows.append(
                {
                    "run_name": run_dir.name,
                    "workload_kind": workload_kind,
                    "features": features,
                    "true_energy": float(rec[TARGET]),
                }
            )
    return rows


def _select_rows(rows: list[dict], *, limit: int, stratified: bool, samples_per_kind: int) -> list[dict]:
    if not stratified:
        return rows[:limit]

    selected: list[dict] = []
    for kind in ["cpu", "mem", "io", "net", "mixed", "gpu-like"]:
        kind_rows = [row for row in rows if row["workload_kind"] == kind]
        selected.extend(kind_rows[:samples_per_kind])
    return selected


def _reset_outputs() -> None:
    targets = [
        Path("data/online_logs"),
        Path("data/training_buffer"),
        Path("models/pending"),
        Path("models/approved"),
        Path("models/latest_approved.json"),
        Path("results/online_learning_update_log.jsonl"),
        Path("results/auto_expert_expansion_log.jsonl"),
    ]
    for target in targets:
        if target.is_dir():
            shutil.rmtree(target)
        elif target.exists():
            target.unlink()


def _append_labels_log(labels: list[dict]) -> None:
    LABELS_LOG.parent.mkdir(parents=True, exist_ok=True)
    with LABELS_LOG.open("a") as fh:
        for label in labels:
            fh.write(json.dumps(label) + "\n")


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


def run_replay(*, base_dir: Path, limit: int, batch_size: int, reset: bool,
               stratified: bool = False, samples_per_kind: int = 300) -> dict:
    if reset:
        _reset_outputs()

    predictor = EnergyPredictor("models/moe_linear.pkl")
    all_rows = _iter_rows(base_dir)
    rows = _select_rows(all_rows, limit=limit, stratified=stratified, samples_per_kind=samples_per_kind)

    logged_request_ids: list[str] = []
    for row in rows:
        routed = predictor.route(row["features"])
        pred = predictor.predict_sample(row["features"])
        request_id = log_prediction(
            log_path=DEFAULT_LOG,
            features=row["features"],
            prediction=pred,
            model_version=0,
            expert=routed,
        )
        row["request_id"] = request_id
        logged_request_ids.append(request_id)

    labels = [
        {
            "request_id": row["request_id"],
            "true_energy": row["true_energy"],
            "run_name": row["run_name"],
            "workload_kind": row["workload_kind"],
            "source": "controlled_replay",
        }
        for row in rows
    ]
    _append_labels_log(labels)
    ingest_summary = ingest_labels(labels, log_path=DEFAULT_LOG, labeled_path=DEFAULT_LABELED, errors_path=DEFAULT_ERRORS)

    buffer_summary = train_buffer.build_buffer(labeled_path=DEFAULT_LABELED)
    pending_summary = pipeline.run_cycle(buffer_path=train_buffer.DEFAULT_BUFFER)

    trained_cycle = None
    if len(rows) >= batch_size:
        trained_cycle = pipeline.run_cycle(buffer_path=train_buffer.DEFAULT_BUFFER, min_labels=batch_size)

    workload_counts: dict[str, int] = {}
    for row in rows:
        workload_counts[row["workload_kind"]] = workload_counts.get(row["workload_kind"], 0) + 1

    return {
        "logged_requests": len(logged_request_ids),
        "ingest_summary": ingest_summary,
        "buffer_summary": buffer_summary,
        "pending_summary": pending_summary,
        "trained_cycle": trained_cycle,
        "workload_counts": workload_counts,
        "prediction_log": str(DEFAULT_LOG),
        "labels_log": str(LABELS_LOG),
        "labeled_log": str(DEFAULT_LABELED),
        "buffer_path": str(train_buffer.DEFAULT_BUFFER),
        "update_log": str(promote.UPDATE_LOG),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Replay controlled rows through the safe online learning MVP")
    ap.add_argument("--limit", type=int, default=500, help="Number of replay rows to process")
    ap.add_argument("--batch-size", type=int, default=500, help="Labeled-sample threshold for retraining")
    ap.add_argument("--base-dir", type=Path, default=RUNS_ROOT, help="Controlled runs root")
    ap.add_argument("--no-reset", action="store_true", help="Do not clear prior outputs before replay")
    ap.add_argument("--stratified", action="store_true", help="Sample rows from each workload kind evenly")
    ap.add_argument("--samples-per-kind", type=int, default=300, help="Rows to sample per workload kind when --stratified is used")
    args = ap.parse_args()

    summary = run_replay(
        base_dir=args.base_dir.resolve(),
        limit=args.limit,
        batch_size=args.batch_size,
        reset=not args.no_reset,
        stratified=args.stratified,
        samples_per_kind=args.samples_per_kind,
    )
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
