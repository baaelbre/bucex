#!/usr/bin/env bash
# Submit the independent reference before the screen-budget arrays.
set -euo pipefail
cd -P "$(dirname "$0")"
export VSC_CLUSTER="${VSC_CLUSTER:-gallade}"
export BUCEX_SCHEDULER=slurm
export VSC_ARRAY_LIMIT="${VSC_ARRAY_LIMIT:-24}"
export VSC_SEPARATE_ARRAY_LIMIT="${VSC_SEPARATE_ARRAY_LIMIT:-4}"
export VSC_MONTHLY_ARRAY_LIMIT="${VSC_MONTHLY_ARRAY_LIMIT:-6}"
export BUCEX_RESULTS_ROOT="${BUCEX_RESULTS_ROOT:-$PWD/results/serra_1983_final}"
exec bash bash_scripts/submit.sh paper final_paper "$@"
