#!/usr/bin/env bash
# Demo 3 — Energy-aware offloading.
#
# Runs the existing offloading entry point (offloading.run_offloading) against
# the bundled demo dataset: the MoE predicts per-job energy, and jobs are placed
# on a primary or a more efficient secondary cluster under a capacity budget.
# Placement strategies compared: all_primary, random, single, moe, oracle.
#
# The secondary cluster is an analytical model, not measured hardware.
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
JOBS="${DEMO_JOBS_PER_WORKLOAD:-10}"
OUT="$REPO_ROOT/results/demo"
mkdir -p "$OUT"

echo "── Demo 3: energy-aware offloading ───────────────────────────"
echo "  data root         : $PEA_DATA_ROOT"
echo "  expert            : $EXPERT"
echo "  jobs per workload : $JOBS"
echo "  output            : $OUT  (git-ignored)"
echo

"$PYTHON" -m offloading.run_offloading --expert "$EXPERT" \
    --jobs-per-workload "$JOBS" \
    --plot "$OUT/offloading_energy_saved.png" 2>&1 | tee "$OUT/offloading.txt"

echo
echo "── Demo 3 complete ───────────────────────────────────────────"
echo "  Decision table and plot: $OUT"
echo "  Note: the secondary cluster is an analytical model and the demo"
echo "        dataset is a small subset, so these savings are not the"
echo "        values reported in the final report."
