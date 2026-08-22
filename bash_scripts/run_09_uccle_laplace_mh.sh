#!/usr/bin/env bash
set -euo pipefail

# Usage: run_09_uccle_laplace_mh.sh [START] [END|latest] [SERIES] [DRAWS] [WARMUP] [CHAINS] [SEED] [DATA_DIR] [RESULTS_ROOT] [RUN_ID] [MH_STEPS]
START="${1:-1892-01-01}"
END="${2:-latest}"
SERIES="${3:-TXx,TXn,TNx,TNn}"
DRAWS="${4:-400}"
WARMUP="${5:-400}"
CHAINS="${6:-2}"
SEED="${7:-11010}"
DATA_DIR="${8:-data}"
RESULTS_ROOT="${9:-results}"
RUN_ID="${10:-$(date +%Y%m%d_%H%M%S)}"
MH_STEPS="${11:-1}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${BUCEX_PACKAGE_ROOT:-${PBS_O_WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
PYTHON_BIN="${BUCEX_PYTHON:-python}"
cd "${PROJECT_ROOT}"

export BUCEX_START="${START}"
export BUCEX_END="${END/latest/}"
export BUCEX_UCCLE_SERIES="${SERIES}"
export BUCEX_DRAWS="${DRAWS}"
export BUCEX_WARMUP="${WARMUP}"
export BUCEX_CHAINS="${CHAINS}"
export BUCEX_SEED="${SEED}"
export BUCEX_DATA_DIR="${DATA_DIR}"
export BUCEX_RESULTS_ROOT="${RESULTS_ROOT}"
export BUCEX_RUN_ID="${RUN_ID}"
export BUCEX_LAPLACE_MH_STEPS="${MH_STEPS}"
export BUCEX_PROGRESS="${BUCEX_PROGRESS:-0}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bucex_mpl_${PBS_JOBID:-$$}"
mkdir -p "${MPLCONFIGDIR}" "${RESULTS_ROOT}" logs

"${PYTHON_BIN}" -c 'import bucex; print("bucex", bucex.__version__)'
"${PYTHON_BIN}" -u examples/09_uccle_laplace_mh.py
