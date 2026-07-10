#!/usr/bin/env python3
"""End-to-end demo of the safe online-learning + automatic expert-expansion loop.

Simulates the full closed loop WITHOUT a live server (so it runs anywhere):
  1. generate fake online predictions (logged to predictions.jsonl)
  2. back-fill true_energy labels (labeled_samples.jsonl)
  3. build training buffer
  4. run one pipeline cycle WITH a synthetic gpu resource dimension injected
     -> discovery detects 'gpu' -> expanded candidate trained -> evaluated ->
        auto approve/reject logged
  5. print the decision log + show where artifacts landed

Usage:
    PYTHONPATH=. python -m auto_expert.demo
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np

from auto_expert.logging import log_prediction
from auto_expert.label_ingestion import ingest_labels
from auto_expert import buffer as buf
from auto_expert import pipeline as pipe
from auto_expert import promote as prom


def _reset(paths):
    for p in paths:
        p = Path(p)
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()


def main():
    print("=== Safe Online Learning + Auto Expert Expansion — DEMO ===\n")
    # clean slate so the demo is reproducible
    _reset(["data/online_logs", "data/training_buffer", "models/pending",
            "models/approved", "models/latest_approved.json",
            "results/online_learning_update_log.jsonl", "results/auto_expert_expansion_log.jsonl"])

    # 1) simulate online predictions (>=100 so training triggers)
    rng = np.random.RandomState(0)
    print("[1] logging 120 simulated online predictions ...")
    rids = []
    for i in range(120):
        feats = {
            "delta_cpu_ns": float(rng.rand() * 1e8),
            "delta_instructions": float(rng.rand() * 1e9),
            "delta_io_bytes": float(rng.rand() * 1e5),
            "delta_net_send_bytes": float(rng.rand() * 1e5),
            "delta_rss_memory": float(rng.rand() * 1e6),
            "context_switches": float(rng.rand() * 1e3),
            "syscall_count": float(rng.rand() * 1e4),
        }
        rid = log_prediction(features=feats, prediction=250.0 + rng.randn() * 20,
                             model_version=0, expert="DAW1")
        rids.append(rid)

    # 2) back-fill labels (true_energy from a "measurement")
    print("[2] back-filling true_energy labels for 120 predictions ...")
    labels = [{"request_id": rid, "true_energy": 250.0 + rng.randn() * 15} for rid in rids]
    summ = ingest_labels(labels)
    print("    ", summ)

    # 3) build buffer
    print("[3] building training buffer ...")
    b = buf.build_buffer()
    print("    ", {k: v for k, v in b.items() if k != "feature_keys"})

    # 4) run one auto-update cycle WITH a synthetic gpu dimension (--demo)
    print("\n[4] running auto-update cycle (with synthetic gpu dimension) ...")
    summary = pipe.run_cycle(demo=True)
    print("\n=== cycle summary ===")
    print(json.dumps(summary, indent=2, default=str))

    # 5) show decision log
    print("\n=== online_learning_update_log.jsonl ===")
    logp = Path("results/online_learning_update_log.jsonl")
    if logp.exists():
        for line in logp.read_text().splitlines():
            if line.strip():
                print("  " + line)
    print("\n=== latest approved (if any) ===")
    if prom.LATEST_FILE.exists():
        print(prom.LATEST_FILE.read_text())
    else:
        print("  (no promotion — candidate was rejected or pending, as designed)")

    print("\n=== artifacts ===")
    for d in ["data/online_logs", "data/training_buffer", "models/pending",
              "models/approved", "results"]:
        p = Path(d)
        if p.exists():
            files = [str(f.relative_to(".")) for f in p.rglob("*") if f.is_file()]
            print(f"  {d}/: {files}")


if __name__ == "__main__":
    main()
