#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash bash_scripts/run_01_tail_simulations.sh \
#   [N_TIME] [PERIOD] [RESULTS_ROOT] [RUN_ID] [OVERWRITE]

N_TIME="${1:-800}"
PERIOD="${2:-4}"
RESULTS_ROOT="${3:-results}"
RUN_ID="${4:-$(date +%Y%m%d_%H%M%S)}"
OVERWRITE="${5:-0}"

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

export BUCEX_N_TIME="${N_TIME}"
export BUCEX_PERIOD="${PERIOD}"
export BUCEX_RESULTS_ROOT="${RESULTS_ROOT}"
export BUCEX_RUN_ID="${RUN_ID}"
export BUCEX_OVERWRITE="${OVERWRITE}"
export BUCEX_PROGRESS="${BUCEX_PROGRESS:-0}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-${TMPDIR:-/tmp}/bucex_mpl_${USER:-user}_${PBS_JOBID:-$$}}"
mkdir -p "${MPLCONFIGDIR}" "${RESULTS_ROOT}"

echo "Running tail simulations"
echo "n time      = ${N_TIME}"
echo "period      = ${PERIOD}"
echo "results     = ${RESULTS_ROOT}"
echo "run id      = ${RUN_ID}"
echo "workdir     = $(pwd)"
date
hostname

"${PYTHON_BIN}" -u examples/01_tail_simulations.py

echo "Finished tail simulations"
date
