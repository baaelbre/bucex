#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash bash_scripts/run_00_uccle_record.sh \
#   [START] [END|latest] [DATA_DIR] [RESULTS_ROOT] [RUN_ID] [OVERWRITE]

START="${1:-1892-01-01}"
END="${2:-latest}"
DATA_DIR="${3:-data}"
RESULTS_ROOT="${4:-results}"
RUN_ID="${5:-$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${6:-0}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="${BUCEX_PACKAGE_ROOT:-${PBS_O_WORKDIR:-${DEFAULT_ROOT}}}"
cd "${PROJECT_ROOT}"

VENV_DIR="${BUCEX_VENV_DIR:-${HOME}/venvs/bucex_env}"
if [[ -n "${BUCEX_PYTHON:-}" ]]; then
  PYTHON_BIN="${BUCEX_PYTHON}"
elif [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
else
  echo "ERROR: BUCEX virtual environment not found: ${VENV_DIR}" >&2
  echo "Create it there or submit with BUCEX_VENV_DIR=/absolute/path/to/venv." >&2
  exit 2
fi
echo "python      = ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys, matplotlib, bucex; print("executable  =", sys.executable); print("matplotlib  =", matplotlib.__version__); print("bucex       =", bucex.__version__)'

export BUCEX_START="${START}"
export BUCEX_END="${END/latest/}"
export BUCEX_DATA_DIR="${DATA_DIR}"
export BUCEX_RESULTS_ROOT="${RESULTS_ROOT}"
export BUCEX_RUN_ID="${RUN_ID}"
export BUCEX_OVERWRITE="${OVERWRITE}"
export BUCEX_PROGRESS="${BUCEX_PROGRESS:-0}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/bucex_mpl_${USER:-user}_${PBS_JOBID:-$$}}"
mkdir -p "${MPLCONFIGDIR}" "${RESULTS_ROOT}"

echo "Running Uccle record figures"
echo "start       = ${START}"
echo "end         = ${END}"
echo "data dir    = ${DATA_DIR}"
echo "results     = ${RESULTS_ROOT}"
echo "run id      = ${RUN_ID}"
echo "workdir     = $(pwd)"
date
hostname

"${PYTHON_BIN}" -u examples/00_uccle_record.py

echo "Finished Uccle record figures"
date
