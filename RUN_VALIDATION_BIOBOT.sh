#!/usr/bin/env bash
# Run this release from its own checkout; resources are shared by this queue.
set -euo pipefail
cd -P "$(dirname "$0")"
tier="${1:-screen}"
selection="${2:-reference}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
case "$tier" in screen|paper) ;; *) echo 'Select screen or paper.' >&2; exit 2;; esac
case "$selection" in reference|fixed|all) ;; *) echo 'Select reference, fixed, or all.' >&2; exit 2;; esac
export BUCEX_RESULTS_ROOT="${BUCEX_RESULTS_ROOT:-$PWD/results/serra_1984_validation}"
export BUCEX_CPUS="${BUCEX_CPUS:-20}"
export BUCEX_MEMORY_GB="${BUCEX_MEMORY_GB:-120}"
export BUCEX_MAX_JOBS="${BUCEX_MAX_JOBS:-10}"
export BUCEX_BATCH="horizon_$selection" BUCEX_TIER="$tier"
exec bash RUN_OVERNIGHT_BIOBOT.sh --tier "$tier" --batch "$BUCEX_BATCH" --skip-priors "$@"
