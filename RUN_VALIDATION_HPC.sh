#!/usr/bin/env bash
# Ten origins, one independently scheduled fit per origin; optional fixed priors.
set -euo pipefail
cd -P "$(dirname "$0")"
tier="${1:-screen}"
selection="${2:-reference}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
case "$tier" in screen|paper) ;; *) echo 'Usage: bash RUN_VALIDATION_HPC.sh [screen|paper] [reference|fixed|all] [--dry-run|--retry-failed]' >&2; exit 2;; esac
case "$selection" in reference|fixed|all) ;; *) echo 'Select reference, fixed, or all.' >&2; exit 2;; esac
export VSC_CLUSTER="${VSC_CLUSTER:-gallade}"
export VSC_PROJECT="${VSC_PROJECT:-gvo00048}"
export VSC_ARRAY_LIMIT="${VSC_ARRAY_LIMIT:-10}"
export VSC_SEPARATE_ARRAY_LIMIT="${VSC_SEPARATE_ARRAY_LIMIT:-20}"
export BUCEX_VENV="${BUCEX_VENV:-$PWD/bucex_env_gallade_py311_1984}"
# SETUP_HPC_ENV.sh runs in a child shell. Resolve these here rather than relying
# on that child to export BUCEX_VENV back into the caller's shell.
export BUCEX_PYTHON="$BUCEX_VENV/bin/python"
export BUCEX_ENV_SETUP="$BUCEX_VENV/environment.sh"
export BUCEX_RESULTS_ROOT="${BUCEX_RESULTS_ROOT:-$PWD/results/serra_1984_validation}"
export BUCEX_SKIP_PRIOR_SIMULATIONS=1
exec "${BUCEX_SUBMIT_PYTHON:-/usr/bin/python3}" job_scripts/submit.py "$tier" "horizon_$selection" "$@"
