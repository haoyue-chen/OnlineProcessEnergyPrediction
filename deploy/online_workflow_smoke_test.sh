#!/usr/bin/env bash
# Smoke test for the online-learning FEEDBACK LOOP (online-workflow).
# Verifies the closed loop runs automatically (no manual curl) and that the model
# state it produces persists and reloads in a fresh serve-online container.
#
# Usage:
#   ./deploy/online_workflow_smoke_test.sh
#   DATA=/path/to/work PORT=8841 ./deploy/online_workflow_smoke_test.sh
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJ="$(cd "$HERE/.." && pwd)"
IMAGE="${IMAGE:-energy-offloading:latest}"
PORT="${PORT:-8842}"
DATA="${DATA:-$PROJ/datasets/demo}"
NAME="energy-ow-smoke"
STATE_DIR="$(mktemp -d)"
RES_DIR="$(mktemp -d)"
JPW=3   # jobs per workload (small workflow)

pass() { echo "  PASS: $1"; }
fail() { echo "  FAIL: $1"; exit 1; }
cleanup() { docker rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$STATE_DIR" "$RES_DIR"; }
trap cleanup EXIT
jget() { python3 -c "import sys,json;print(json.load(sys.stdin)['$1'])"; }

echo "== online workflow smoke test =="
echo "image=$IMAGE  data=$DATA  state=$STATE_DIR"
[ -d "$DATA" ] || fail "data dir not found: $DATA (set DATA=...)"
docker image inspect "$IMAGE" >/dev/null 2>&1 || fail "image not found: $IMAGE"

echo "[1/6] run online-workflow (auto /predict + /update per job) ..."
docker run --rm \
  -e ONLINE_STATE_PATH=/app/models/state/online_state.pkl \
  -v "$DATA:/data/work:ro" \
  -v "$STATE_DIR:/app/models/state" \
  -v "$RES_DIR:/app/results" \
  "$IMAGE" online-workflow --jobs-per-workload $JPW \
    --out /app/results/run.json >/dev/null 2>&1 \
  && pass "workflow completed" || fail "workflow run failed"

RUN="$RES_DIR/run.json"
[ -f "$RUN" ] || fail "run.json not written"

echo "[2/6] every job called /predict ..."
NJOBS=$(cat "$RUN" | jget n_jobs)
NPRED=$(cat "$RUN" | jget n_predict_calls)
[ "$NPRED" = "$NJOBS" ] && pass "/predict calls = jobs = $NPRED" || fail "predict $NPRED != jobs $NJOBS"

echo "[3/6] every job called /update ..."
NUPD=$(cat "$RUN" | jget n_update_calls)
ALLOK=$(cat "$RUN" | jget all_updates_succeeded)
[ "$NUPD" = "$NJOBS" ] && [ "$ALLOK" = "True" ] \
  && pass "/update calls = jobs = $NUPD, all succeeded" || fail "updates $NUPD/$NJOBS ok=$ALLOK"

echo "[4/6] num_updates increased ..."
NB=$(cat "$RUN" | jget num_updates_before)
NA=$(cat "$RUN" | jget num_updates_after)
[ "$NA" -gt "$NB" ] && pass "num_updates $NB -> $NA" || fail "num_updates did not increase"

echo "[5/6] online_state.pkl persisted to volume ..."
[ -f "$STATE_DIR/online_state.pkl" ] && pass "state file present" || fail "no state file"

echo "[6/6] restart as serve-online, state reloads ..."
docker run -d --name "$NAME" -p "$PORT:8800" \
  -e ONLINE_STATE_PATH=/app/models/state/online_state.pkl \
  -v "$STATE_DIR:/app/models/state" "$IMAGE" serve-online >/dev/null
for i in $(seq 1 15); do sleep 1; curl -sf "localhost:$PORT/health" >/dev/null 2>&1 && break; \
  [ "$i" = 15 ] && fail "service not healthy after restart"; done
RELOADED=$(curl -sf "localhost:$PORT/info" | jget num_updates)
[ "$RELOADED" = "$NA" ] && pass "reloaded num_updates=$RELOADED (persisted across restart)" \
  || fail "expected $NA after reload, got $RELOADED"

echo "== all online workflow smoke checks passed =="
