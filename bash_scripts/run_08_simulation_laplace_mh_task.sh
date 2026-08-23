#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash bash_scripts/run_08_simulation_laplace_mh_task.sh \
#   SCENARIO CHAIN N_TIME PERIOD SIMULATION_SEED DRAWS WARMUP TOTAL_CHAINS \
#   MCMC_SEED RESULTS_ROOT RUN_ID OVERWRITE MH_STEPS SCENARIO_KEYS

SCENARIO="${1:?scenario key is required}"
CHAIN="${2:?chain index is required}"
N_TIME="${3:-1000}"
PERIOD="${4:-4}"
SIMULATION_SEED="${5:-13081997}"
DRAWS="${6:-1000}"
WARMUP="${7:-1000}"
TOTAL_CHAINS="${8:-4}"
MCMC_SEED="${9:-13081997}"
RESULTS_ROOT="${10:-results}"
RUN_ID="${11:?run id is required}"
OVERWRITE="${12:-0}"
MH_STEPS="${13:-1}"
SCENARIO_KEYS="${14:-stationary:linear:random_walk:llt:dynamic_season:llt_season}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${BUCEX_PACKAGE_ROOT:-${PBS_O_WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
cd "${PROJECT_ROOT}"
source config/laplace_mh_simulation_array.sh

laplace_mh_validate_positive_integer "chain" "${CHAIN}"
laplace_mh_validate_positive_integer "total chains" "${TOTAL_CHAINS}"
laplace_mh_validate_positive_integer "draws" "${DRAWS}"
laplace_mh_validate_positive_integer "warmup" "${WARMUP}"
laplace_mh_validate_positive_integer "MH steps" "${MH_STEPS}"
laplace_mh_parse_scenarios "${SCENARIO_KEYS}"
if (( CHAIN > TOTAL_CHAINS )); then
  echo "Chain ${CHAIN} exceeds TOTAL_CHAINS=${TOTAL_CHAINS}." >&2
  exit 2
fi
case ":${LMH_SCENARIO_KEYS}:" in
  *":${SCENARIO}:"*) ;;
  *)
    echo "Scenario ${SCENARIO} is not in SCENARIO_KEYS=${LMH_SCENARIO_KEYS}." >&2
    exit 2
    ;;
esac

if [[ -n "${BUCEX_PYTHON_MODULE:-}" ]]; then
  if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
  fi
  module load "${BUCEX_PYTHON_MODULE}"
fi

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

SIGNATURE="$(laplace_mh_run_signature \
  "${N_TIME}" "${PERIOD}" "${DRAWS}" "${WARMUP}" \
  "${TOTAL_CHAINS}" "${MH_STEPS}")"
SHARED_RUN_DIR="$(laplace_mh_run_directory \
  "${RESULTS_ROOT}" "${RUN_ID}" "${SIGNATURE}")"

export BUCEX_N_TIME="${N_TIME}"
export BUCEX_PERIOD="${PERIOD}"
export BUCEX_SIMULATION_SEED="${SIMULATION_SEED}"
export BUCEX_DRAWS="${DRAWS}"
export BUCEX_WARMUP="${WARMUP}"
export BUCEX_CHAINS=1
export BUCEX_SEED="${MCMC_SEED}"
export BUCEX_LAPLACE_MH_STEPS="${MH_STEPS}"
export BUCEX_SCENARIO_KEYS="${LMH_SCENARIO_KEYS}"
export BUCEX_SCENARIO_KEY="${SCENARIO}"
export BUCEX_CHAIN_INDEX="${CHAIN}"
export BUCEX_SHARED_RUN_DIR="${SHARED_RUN_DIR}"
export BUCEX_OVERWRITE="${OVERWRITE}"
export BUCEX_PROGRESS="${BUCEX_PROGRESS:-1}"
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bucex_mpl_${USER:-user}_${PBS_JOBID:-$$}_${SCENARIO}_c${CHAIN}"
mkdir -p "${MPLCONFIGDIR}" "${SHARED_RUN_DIR}"

echo "Running one Laplace-MH scenario-chain task"
echo "scenario       = ${SCENARIO}"
echo "chain          = ${CHAIN}/${TOTAL_CHAINS}"
echo "draws/warmup   = ${DRAWS}/${WARMUP}"
echo "n time/period  = ${N_TIME}/${PERIOD}"
echo "MH steps       = ${MH_STEPS}"
echo "shared run dir = ${SHARED_RUN_DIR}"
echo "python         = ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys, bucex, matplotlib; print("executable", sys.executable); print("bucex", bucex.__version__); print("matplotlib", matplotlib.__version__); assert bucex.__version__ == "1.1.4"'
date
hostname

"${PYTHON_BIN}" -u examples/_08_simulation_laplace_mh_task.py

echo "Finished one Laplace-MH scenario-chain task"
date
