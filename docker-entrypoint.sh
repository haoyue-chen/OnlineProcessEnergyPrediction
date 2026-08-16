#!/usr/bin/env bash
# Entry point for the full-project image. Dispatches to each deliverable so the
# whole project can be reproduced from one container.
#
#   demo-offline    — Demo 1: offline prediction (global vs workload vs resource MoE)
#   demo-online     — Demo 2: online adaptation under drift
#   demo-offloading — Demo 3: energy-aware offloading
#
#   serve       — start the MoE inference HTTP service (default; port 8800)
#   serve-online— start the LIVE online-learning service (/predict + /update; port 8800)
#   online-workflow — closed loop: start serve-online, run a workflow that auto-calls
#                     /predict before each job and /update after (no manual curl)
#   task4       — MoE vs Single Model (5-fold CV)
#   task5       — Static vs Online MoE under drift (ARF expert)
#   online      — online/offline baseline comparison (the 7-family table)
#   compare     — full batch+online model survey
#   offload     — energy-aware offloading simulation
#   snakemake   — live Snakemake offloading DAG (strategy comparison)
#   export      — (re)export the MoE artifact used by `serve`
#   all         — run task4 + task5 + online + offload (writes results/)
#   bash        — drop into a shell
#
# Data is mounted read-only at /data/work; results land in /app/results.
set -euo pipefail
export PYTHONPATH=/app
PY=python
CMD="${1:-serve}"; shift || true

case "$CMD" in
  # Demo wrappers. PYTHON=python so they use the image interpreter, and
  # PEA_DATA_ROOT stays at its image default (/data/work) unless overridden.
  demo-offline)    PYTHON=$PY exec /app/demo/run_offline_demo.sh ;;
  demo-online)     PYTHON=$PY exec /app/demo/run_online_demo.sh ;;
  demo-offloading) PYTHON=$PY exec /app/demo/run_offloading_demo.sh ;;
  serve)     exec $PY -m inference.server ;;
  serve-online) exec $PY -m inference.online_server ;;
  online-workflow)
    # Closed loop in one container: launch the online service in the background,
    # then drive it job-by-job (auto /predict + /update). State persists to
    # ONLINE_STATE_PATH (mount a volume to keep it across runs).
    $PY -m inference.online_server &
    SRV_PID=$!
    trap 'kill $SRV_PID 2>/dev/null || true' EXIT
    $PY -m offloading.online_workflow --url "http://localhost:${PORT:-8800}" \
        --wait 20 "$@"
    rc=$?
    kill $SRV_PID 2>/dev/null || true
    exit $rc
    ;;
  export-online) exec $PY -m moe_export.export_online --online-expert "${1:-arf}" --out models/online_base.pkl ;;
  task4)     exec $PY -m moe.run_moe_baseline "$@" ;;
  task5)     exec $PY -m moe.run_online --online-expert arf "$@" ;;
  online)    exec $PY -m moe.online_baseline_comparison "$@" ;;
  compare)   exec $PY -m moe.compare_models "$@" ;;
  offload)   exec $PY -m offloading.run_offloading "$@" ;;
  snakemake) OFFLOAD_PY=python OFFLOAD_SNAKEMAKE=snakemake exec ./snakemake_integration/run.sh "${1:-compare}" ;;
  export)    exec $PY -m moe_export.export_moe --expert "${1:-linear}" --out models/moe_linear.pkl ;;
  all)
    $PY -m moe.run_moe_baseline --expert linear
    $PY -m moe.run_online --online-expert arf
    $PY -m moe.online_baseline_comparison
    $PY -m offloading.run_offloading --expert linear
    echo "Done. Results in /app/results/."
    ;;
  bash)      exec bash ;;
  *) echo "Unknown command: $CMD"; sed -n '3,20p' "$0"; exit 2 ;;
esac
