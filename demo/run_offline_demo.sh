#!/usr/bin/env bash
# Demo 1 — Offline energy prediction.
#
# Runs the existing offline entry points against the bundled demo dataset:
#   * global single model vs workload-aware MoE   (moe.run_moe_baseline)
#   * resource-aware MoE vs both baselines        (feature_moe.run_resource_moe)
#
# The demo dataset is a small real-data subset. It demonstrates that the
# pipeline runs; it does NOT reproduce the numbers in the report.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
export PEA_DATA_ROOT="${PEA_DATA_ROOT:-$REPO_ROOT/datasets/demo}"
export PYTHONPATH="$REPO_ROOT"

EXPERT="${DEMO_EXPERT:-linear}"
OUT="$REPO_ROOT/results/demo"
mkdir -p "$OUT"

echo "── Demo 1: offline energy prediction ─────────────────────────"
echo "  data root : $PEA_DATA_ROOT"
echo "  expert    : $EXPERT"
echo "  output    : $OUT  (git-ignored)"
echo

echo "[1/2] Global single model vs workload-aware MoE (5-fold CV)"
"$PYTHON" -m moe.run_moe_baseline --expert "$EXPERT" \
    --plot "$OUT/offline_workload_moe.png" 2>&1 | tee "$OUT/offline_workload_moe.txt"

echo
echo "[2/2] Resource-aware MoE vs single global and workload MoE (5-fold CV)"
"$PYTHON" -m feature_moe.run_resource_moe --expert "$EXPERT" \
    --plot "$OUT/offline_resource_moe.png" 2>&1 | tee "$OUT/offline_resource_moe.txt"

echo
echo "── Demo 1 complete ───────────────────────────────────────────"
echo "  Tables and plots: $OUT"
echo "  Note: results come from the small demo dataset and are not the"
echo "        values reported in the final report."
