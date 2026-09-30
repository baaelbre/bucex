#!/usr/bin/env bash
# One response per array element, with parallel chains inside the fit.
set -euo pipefail
cd -P "$(dirname "$0")"
batch=monthly_all
if [[ $# -gt 0 && "$1" != --* ]]; then batch="$1"; shift; fi
export BUCEX_BATCH="$batch"
export BUCEX_RESULTS_ROOT="${BUCEX_RESULTS_ROOT:-$PWD/results/serra_1982_monthly}"
export BUCEX_SCHEDULER="${BUCEX_SCHEDULER:-slurm}"
export VSC_CLUSTER="${VSC_CLUSTER:-gallade}"
export VSC_ARRAY_LIMIT="${VSC_ARRAY_LIMIT:-32}"
exec bash bash_scripts/submit.sh "${BUCEX_TIER:-screen}" "$batch" "$@"
