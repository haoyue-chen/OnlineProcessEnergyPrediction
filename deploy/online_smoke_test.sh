#!/usr/bin/env bash
# Smoke test for the LIVE online-learning service (serve-online).
# Verifies: health, predict, update (model changes), state persists across restart.
#
# Usage:
#   ./deploy/online_smoke_test.sh
#   PORT=8821 IMAGE=energy-offloading:latest ./deploy/online_smoke_test.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-energy-offloading:latest}"
PORT="${PORT:-8820}"
NAME="energy-online-smoke"
STATE_DIR="$(mktemp -d)"

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; exit 1; }
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$STATE_DIR"; }
trap cleanup EXIT

jget() { python3 -c "import sys,json;print(json.load(sys.stdin)['$1'])"; }

echo "== online smoke test =="
echo "image=$IMAGE  port=$PORT  state_dir=$STATE_DIR"
docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "image not found: $IMAGE (build it first)"

start() {
  docker rm -f "$NAME" >/dev/null 2>&1 || true
  docker run -d --name "$NAME" -p "$PORT:8800" \
    -e ONLINE_STATE_PATH=/app/models/state/online_state.pkl \
    -v "$STATE_DIR:/app/models/state" "$IMAGE" serve-online >/dev/null
  for i in $(seq 1 15); do
    sleep 1; curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && return 0
  done
  fail "service did not become healthy"
}

echo "[1/7] start serve-online ..."; start; pass "service up"

echo "[2/7] GET /health ..."
curl -sf "localhost:$PORT/health" | grep -q '"ok"' && pass "/health ok" || fail "/health"

echo "[3/7] GET /info (baseline num_updates) ..."
U0=$(curl -sf "localhost:$PORT/info" | jget num_updates)
pass "num_updates=$U0"

echo "[4/7] POST /predict ..."
PRED=$(curl -sf -X POST "localhost:$PORT/predict" -H 'Content-Type: application/json' \
  -d '{"features":{"delta_cpu_ns":5e8,"delta_instructions":2e9}}')
echo "$PRED" | grep -q '"prediction_id"' && pass "/predict -> $PRED" || fail "/predict"
PID=$(echo "$PRED" | jget prediction_id)

echo "[5/7] POST /update (true_energy_wh=250) ..."
UPD=$(curl -sf -X POST "localhost:$PORT/update" -H 'Content-Type: application/json' \
  -d "{\"prediction_id\":\"$PID\",\"true_energy_wh\":250.0}")
U1=$(echo "$UPD" | jget num_updates)
[ "$U1" -gt "$U0" ] && pass "num_updates $U0 -> $U1 (model updated)" || fail "num_updates did not increase"

echo "[6/7] predict again (model_version advanced) ..."
V=$(curl -sf -X POST "localhost:$PORT/predict" -H 'Content-Type: application/json' \
  -d '{"features":{"delta_cpu_ns":5e8}}' | jget model_version)
[ "$V" -ge "$U1" ] && pass "model_version=$V" || fail "version did not advance"

echo "[7/7] restart container, state must persist ..."
docker rm -f "$NAME" >/dev/null 2>&1
start
U2=$(curl -sf "localhost:$PORT/info" | jget num_updates)
[ "$U2" -eq "$U1" ] && pass "after restart num_updates=$U2 (persisted)" \
  || fail "state not persisted: expected $U1 got $U2"

echo "== all online smoke checks passed =="
