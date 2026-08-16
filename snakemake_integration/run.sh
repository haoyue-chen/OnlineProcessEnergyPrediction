#!/usr/bin/env bash
# Driver: run the live Snakemake offloading workflow, optionally across strategies.
#
# Usage:
#   ./run.sh                      # default: strategy=moe, expert=linear
#   ./run.sh single               # one strategy
#   ./run.sh compare              # run moe / single / oracle / all_primary and diff
#
# Uses the ``python3`` / ``snakemake`` found on PATH (activate the project venv
# first). Override either with OFFLOAD_PY / OFFLOAD_SNAKEMAKE if needed.
set -euo pipefail

# estimation/ is two levels up from this script (snakemake_integration/run.sh).
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EST_DIR="$(cd "$HERE/.." && pwd)"
PY="${OFFLOAD_PY:-python3}"
SNAKE="${OFFLOAD_SNAKEMAKE:-snakemake}"
EXPERT="${EXPERT:-linear}"
JOBS_PER_WL="${JOBS_PER_WL:-10}"
CAPACITY="${CAPACITY:-0.15}"

cd "$EST_DIR"

run_one () {
  local strat="$1"
  local rundir="snakemake_integration/.run_${strat}"
  rm -rf "$rundir"
  OFFLOAD_PY="$PY" OFFLOAD_SRC="$EST_DIR" "$SNAKE" \
    -s snakemake_integration/Snakefile --cores 4 -d "$rundir" --quiet all \
    --config expert="$EXPERT" strategy="$strat" \
             jobs_per_workload="$JOBS_PER_WL" capacity_frac="$CAPACITY" \
    >/dev/null 2>&1
  cp "$rundir/summary.json" "snakemake_integration/summary_${strat}.json"
  "$PY" - "$strat" "snakemake_integration/summary_${strat}.json" <<'PYEOF'
import json, sys
strat, path = sys.argv[1], sys.argv[2]
s = json.load(open(path))
print(f"{strat:12s} offloaded={s['n_offloaded']:>3}/{s['n_jobs']:<3} "
      f"energy={s['total_energy_kwh']:8.2f} kWh  saved={s['energy_saved_pct']:5.1f}%  "
      f"cost={s['total_cost']:8.1f}  makespan={s['makespan_s']:.0f}s")
PYEOF
}

case "${1:-moe}" in
  compare)
    echo "── Live Snakemake offloading — strategy comparison (expert=$EXPERT) ──"
    for s in all_primary random single moe oracle; do run_one "$s"; done
    ;;
  *)
    run_one "${1:-moe}"
    ;;
esac
