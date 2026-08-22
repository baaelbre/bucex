#!/usr/bin/env bash
set -euo pipefail

# Usage: run_08_simulation_laplace_mh.sh [N] [DRAWS] [WARMUP] [CHAINS] [SEED] [RESULTS_ROOT] [RUN_ID] [MH_STEPS]
N_TIME="${1:-240}"
DRAWS="${2:-400}"
WARMUP="${3:-400}"
CHAINS="${4:-2}"
SEED="${5:-11001}"
RESULTS_ROOT="${6:-results}"
RUN_ID="${7:-$(date +%Y%m%d_%H%M%S)}"
MH_STEPS="${8:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${BUCEX_PACKAGE_ROOT:-${PBS_O_WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
PYTHON_BIN="${BUCEX_PYTHON:-python}"
cd "${PROJECT_ROOT}"

export BUCEX_N_TIME="${N_TIME}"
export BUCEX_DRAWS="${DRAWS}"
export BUCEX_WARMUP="${WARMUP}"
export BUCEX_CHAINS="${CHAINS}"
export BUCEX_SEED="${SEED}"
export BUCEX_RESULTS_ROOT="${RESULTS_ROOT}"
export BUCEX_RUN_ID="${RUN_ID}"
export BUCEX_LAPLACE_MH_STEPS="${MH_STEPS}"
export BUCEX_PROGRESS="${BUCEX_PROGRESS:-0}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bucex_mpl_${PBS_JOBID:-$$}"
mkdir -p "${MPLCONFIGDIR}" "${RESULTS_ROOT}" logs

"${PYTHON_BIN}" -c 'import bucex; print("bucex", bucex.__version__)'
"${PYTHON_BIN}" -u examples/08_simulation_laplace_mh.py
