#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_ROOT}"
source config/laplace_sensitivity_grid.sh

if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available in this shell." >&2
  exit 2
fi

mkdir -p logs/laplace_sensitivity

QSUB_EXPORTS="BUCEX_VENV_DIR=${BUCEX_VENV_DIR:-${BUCEX_VENV:-${HOME}/venvs/bucex_env}}"
if [[ -n "${BUCEX_PYTHON_MODULE:-}" ]]; then
  QSUB_EXPORTS+=",BUCEX_PYTHON_MODULE=${BUCEX_PYTHON_MODULE}"
fi

FIT_JOB_ID="$(
  qsub \
    -v "${QSUB_EXPORTS}" \
    -t "1-${SENS_N_FIT_TASKS}%${SENS_MAX_CONCURRENT_FITS}" \
    job_scripts/submit_laplace_sensitivity_fits.pbs
)"
echo "Submitted fit array: ${FIT_JOB_ID} (${SENS_N_FIT_TASKS} tasks)"

# On the VSC PBS installation, afterok on the array parent waits for every
# array element. If your cluster requires afterokarray, replace afterok below.
COMBINE_JOB_ID="$(
  qsub \
    -v "${QSUB_EXPORTS}" \
    -W "depend=afterok:${FIT_JOB_ID}" \
    -t "1-${SENS_N_COMBINE_TASKS}%${SENS_MAX_CONCURRENT_COMBINES}" \
    job_scripts/submit_laplace_sensitivity_combine.pbs
)"
echo "Submitted combine array: ${COMBINE_JOB_ID} (${SENS_N_COMBINE_TASKS} tasks)"
echo "The combine array will start only after the complete fit array succeeds."
