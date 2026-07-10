#!/usr/bin/env bash
# Start the safe approved-model online server.
#
# /predict serves ONLY the model selected by models/latest_approved.json
# (ApprovedPredictor). Pending/rejected candidates under models/pending/ are
# never loaded. If no approved model is registered, it falls back to the legacy
# base MoE artifact (models/moe_linear.pkl).

set -euo pipefail

# cd to project root (this script lives in <root>/scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

mkdir -p data/online_logs

export PYTHONPATH=.
export ONLINE_BASE_PATH=models/moe_linear.pkl
export ONLINE_STATE_PATH=/tmp/mpds_online_state.pkl
export ONLINE_PRED_LOG=data/online_logs/predictions_live.jsonl
export PORT=8801

if [[ -x "/home/hujiao/.CodeBuddy/tmp/gpu-data-venv/bin/python" ]]; then
  PYTHON_BIN="/home/hujiao/.CodeBuddy/tmp/gpu-data-venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
else
  echo "ERROR: neither the repo test venv nor python3 is available" >&2
  exit 127
fi

echo "=== Approved-model online server ==="
echo "Python     : ${PYTHON_BIN}"
echo "Server URL : http://localhost:${PORT}"
echo "/info      : curl -s localhost:${PORT}/info"
echo "Predict    : curl -s -X POST localhost:${PORT}/predict -H 'Content-Type: application/json' -d '{\"features\":{...}}'"
echo "Pred log   : ${ONLINE_PRED_LOG}"
echo "-----------------------------------"
exec "${PYTHON_BIN}" -m inference.online_server
