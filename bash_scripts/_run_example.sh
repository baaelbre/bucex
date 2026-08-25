#!/usr/bin/env bash
set -euo pipefail

SCRIPT_PATH="$1"
DEFAULT_CONFIG="$2"
CONFIG_PATH="${3:-${DEFAULT_CONFIG}}"
MAX_WORKERS="${4:-${PBS_NP:-1}}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${BUCEX_PACKAGE_ROOT:-${PBS_O_WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
cd "${PROJECT_ROOT}"

VENV_DIR="${BUCEX_VENV_DIR:-${HOME}/venvs/bucex_env}"
if [[ -n "${BUCEX_PYTHON:-}" ]]; then
  PYTHON_BIN="${BUCEX_PYTHON}"
elif [[ -n "${VIRTUAL_ENV:-}" && -x "${VIRTUAL_ENV}/bin/python" ]]; then
  PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
elif [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
else
  echo "ERROR: no bucex Python environment was found." >&2
  echo "Activate it first, or set BUCEX_VENV_DIR or BUCEX_PYTHON." >&2
  exit 2
fi

mkdir -p logs
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

echo "script      = ${SCRIPT_PATH}"
echo "config      = ${CONFIG_PATH}"
echo "workers     = ${MAX_WORKERS}"
echo "python      = ${PYTHON_BIN}"
echo "workdir     = $(pwd)"
date
hostname
"${PYTHON_BIN}" -c 'import bucex, matplotlib; print("bucex       =", bucex.__version__); print("matplotlib  =", matplotlib.__version__)'

"${PYTHON_BIN}" -u job_scripts/run_parallel_chains.py \
  --script "${SCRIPT_PATH}" \
  --config "${CONFIG_PATH}" \
  --max-workers "${MAX_WORKERS}"

echo "Finished ${SCRIPT_PATH}"
date
