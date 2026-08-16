#!/usr/bin/env bash
# Demo 2 — Online adaptation under concept drift.
#
# Runs the existing online entry point (moe.run_online) against the bundled
# demo dataset. The four workloads are streamed end-to-end to induce drift and
# three strategies are compared prequentially:
#   Static-Single, Static-MoE, Online-MoE
#
# Set WITH_PERIODIC_RETRAIN=1 to add the optional Periodic-Retrain baseline
# (off by default, matching the reported table).
#
# The demo dataset is a small real-data subset. It demonstrates that the
# pipeline runs; it does NOT reproduce the numbers in the report.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python3}"
export PEA_DATA_ROOT="${PEA_DATA_ROOT:-$REPO_ROOT/datasets/demo}"
export PYTHONPATH="$REPO_ROOT"

ONLINE_EXPERT="${DEMO_ONLINE_EXPERT:-arf}"
OUT="$REPO_ROOT/results/demo"
mkdir -p "$OUT"

EXTRA=()
if [[ "${WITH_PERIODIC_RETRAIN:-0}" == "1" ]]; then
  EXTRA+=(--with-periodic-retrain)
  echo "  (periodic-retrain baseline enabled)"
fi

echo "── Demo 2: online adaptation under drift ─────────────────────"
echo "  data root     : $PEA_DATA_ROOT"
echo "  online expert : $ONLINE_EXPERT"
echo "  output        : $OUT  (git-ignored)"
echo "  stream order  : DAW1 -> DAW2 -> Phoronix -> Stress"
echo "  warm-up       : first 15% of each workload, excluded from scoring"
echo

"$PYTHON" -m moe.run_online --online-expert "$ONLINE_EXPERT" "${EXTRA[@]}" \
    --plot "$OUT/online_adaptation.png" 2>&1 | tee "$OUT/online_adaptation.txt"

echo
echo "── Demo 2 complete ───────────────────────────────────────────"
echo "  Table and error-over-time plot: $OUT"
echo "  Note: the demo dataset is a short slice of each workload, so the"
echo "        drift behaviour is weaker than in the reported experiment and"
echo "        the values are not the ones in the final report."
