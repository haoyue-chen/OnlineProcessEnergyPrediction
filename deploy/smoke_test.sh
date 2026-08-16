#!/usr/bin/env bash
# One-shot smoke test for the energy-offloading image: exercises the batch tasks,
# docker compose, and the inference HTTP endpoints. Fails fast on any error.
#
# Usage:
#   ./deploy/smoke_test.sh                 # uses datasets/demo, port 8800
#   DATA=/path/to/dataset PORT=8801 ./deploy/smoke_test.sh
set -euo pipefail

# Resolve paths relative to this script (deploy/ is one level under the project).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-energy-offloading:latest}"
DATA="${DATA:-$PROJ/datasets/demo}"
PORT="${PORT:-8800}"
NAME="energy-smoke"

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; exit 1; }
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; }
trap cleanup EXIT

echo "== smoke test =="
echo "image=$IMAGE  data=$DATA  port=$PORT"
[ -d "$DATA" ] || fail "data dir not found: $DATA (set DATA=...)"
docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "image not found: $IMAGE (build it first)"

echo "[1/5] docker run task4 ..."
docker run --rm -v "$DATA:/data/work:ro" "$IMAGE" task4 --expert linear 2>&1 \
  | grep -q "OVERALL" && pass "task4 produced OVERALL row" || fail "task4 no output"

echo "[2/5] docker run offload ..."
docker run --rm -v "$DATA:/data/work:ro" "$IMAGE" offload --expert linear 2>&1 \
  | grep -q "MoE closes" && pass "offload produced decision summary" || fail "offload no output"

echo "[3/5] start service (docker run -d) ..."
cleanup
docker run -d --name "$NAME" -p "$PORT:8800" -v "$DATA:/data/work:ro" "$IMAGE" serve >/dev/null
# wait for /health
for i in $(seq 1 15); do
  sleep 1
  if curl -sf "localhost:$PORT/health" >/dev/null 2>&1; then break; fi
  [ "$i" = 15 ] && fail "service did not become healthy"
done

echo "[4/5] curl /health ..."
curl -sf "localhost:$PORT/health" | grep -q '"ok"' && pass "/health ok" || fail "/health bad response"

echo "[5/5] curl /predict ..."
RESP=$(curl -sf -X POST "localhost:$PORT/predict" -H 'Content-Type: application/json' \
  -d '{"features":{"delta_cpu_ns":5e8,"delta_instructions":2e9}}')
echo "$RESP" | grep -q '"energy_wh"' && pass "/predict -> $RESP" || fail "/predict bad response: $RESP"

echo "== all smoke checks passed =="
