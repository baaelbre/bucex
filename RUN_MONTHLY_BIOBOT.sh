#!/usr/bin/env bash
set -euo pipefail
cd -P "$(dirname "$0")"
batch=monthly_all
if [[ $# -gt 0 && "$1" != --* ]]; then batch="$1"; shift; fi
export BUCEX_BATCH="$batch"
export BUCEX_RESULTS_ROOT="${BUCEX_RESULTS_ROOT:-$PWD/results/serra_1983_monthly}"
exec bash RUN_OVERNIGHT_BIOBOT.sh --tier "${BUCEX_TIER:-screen}" --batch "$batch" \
    --cpus "${BUCEX_CPUS:-48}" --max-jobs "${BUCEX_MAX_JOBS:-24}" --memory-gb "${BUCEX_MEMORY_GB:-150}" "$@"
