#!/usr/bin/env bash
# Publication reference first; screening budgets for all other final checks.
set -euo pipefail
cd -P "$(dirname "$0")"
export BUCEX_BATCH=final_paper
export BUCEX_TIER=paper
export BUCEX_RESULTS_ROOT="${BUCEX_RESULTS_ROOT:-$PWD/results/serra_1983_final}"
exec bash RUN_OVERNIGHT_BIOBOT.sh --tier paper --batch final_paper \
  --cpus "${BUCEX_CPUS:-48}" --max-jobs "${BUCEX_MAX_JOBS:-24}" \
  --memory-gb "${BUCEX_MEMORY_GB:-150}" "$@"
