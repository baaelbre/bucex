#!/usr/bin/env bash
set -euo pipefail

# Usage:
# bash bash_scripts/run_08_simulation_laplace_mh_finalize.sh \
#   N_TIME PERIOD SIMULATION_SEED DRAWS WARMUP CHAINS MCMC_SEED \
#   RESULTS_ROOT RUN_ID OVERWRITE MH_STEPS SCENARIO_KEYS

N_TIME="${1:-1000}"
PERIOD="${2:-4}"
SIMULATION_SEED="${3:-13081997}"
DRAWS="${4:-1000}"
WARMUP="${5:-1000}"
CHAINS="${6:-4}"
MCMC_SEED="${7:-13081997}"
RESULTS_ROOT="${8:-results}"
RUN_ID="${9:?run id is required}"
OVERWRITE="${10:-0}"
MH_STEPS="${11:-1}"
SCENARIO_KEYS="${12:-stationary:linear:random_walk:llt:dynamic_season:llt_season}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${BUCEX_PACKAGE_ROOT:-${PBS_O_WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}}"
cd "${PROJECT_ROOT}"
source config/laplace_mh_simulation_array.sh

laplace_mh_validate_positive_integer "chains" "${CHAINS}"
laplace_mh_validate_positive_integer "draws" "${DRAWS}"
laplace_mh_validate_positive_integer "warmup" "${WARMUP}"
laplace_mh_validate_positive_integer "MH steps" "${MH_STEPS}"
laplace_mh_parse_scenarios "${SCENARIO_KEYS}"

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
  exit 2
fi

SIGNATURE="$(laplace_mh_run_signature \
  "${N_TIME}" "${PERIOD}" "${DRAWS}" "${WARMUP}" "${CHAINS}" "${MH_STEPS}")"
SHARED_RUN_DIR="$(laplace_mh_run_directory \
  "${RESULTS_ROOT}" "${RUN_ID}" "${SIGNATURE}")"

CHAIN_DIRS=()
for ((chain = 1; chain <= CHAINS; chain++)); do
  printf -v chain_label "%02d" "${chain}"
  chain_dir="${SHARED_RUN_DIR}/tasks/chain${chain_label}"
  for scenario in "${LMH_SCENARIOS[@]}"; do
    fit_path="${chain_dir}/fits/${scenario}/combined.bucex"
    manifest_path="${chain_dir}/manifests/${scenario}.json"
    if [[ ! -f "${fit_path}" ]]; then
      echo "Missing array-task fit: ${fit_path}" >&2
      exit 3
    fi
    if [[ ! -f "${manifest_path}" ]]; then
      echo "Missing array-task manifest: ${manifest_path}" >&2
      exit 3
    fi
  done
  CHAIN_DIRS+=("$(cd "${chain_dir}" && pwd)")
done
COMBINE_RUNS="$(IFS=:; echo "${CHAIN_DIRS[*]}")"

export BUCEX_N_TIME="${N_TIME}"
export BUCEX_PERIOD="${PERIOD}"
export BUCEX_SIMULATION_SEED="${SIMULATION_SEED}"
export BUCEX_DRAWS="${DRAWS}"
export BUCEX_WARMUP="${WARMUP}"
export BUCEX_CHAINS="${CHAINS}"
export BUCEX_SEED="${MCMC_SEED}"
export BUCEX_RESULTS_ROOT="${RESULTS_ROOT}"
export BUCEX_RUN_ID="${RUN_ID}"
export BUCEX_OVERWRITE="${OVERWRITE}"
export BUCEX_LAPLACE_MH_STEPS="${MH_STEPS}"
export BUCEX_SCENARIO_KEYS="${LMH_SCENARIO_KEYS}"
export BUCEX_CHAIN_ONLY=0
export BUCEX_COMBINE_RUNS="${COMBINE_RUNS}"
export BUCEX_PROGRESS=0
export BUCEX_ARRAY_WORKFLOW=1
export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MPLCONFIGDIR="${TMPDIR:-/tmp}/bucex_mpl_${USER:-user}_${PBS_JOBID:-$$}_finalize"
mkdir -p "${MPLCONFIGDIR}" "${SHARED_RUN_DIR}"

echo "Combining Laplace-MH PBS array tasks"
echo "scenarios      = ${LMH_SCENARIO_KEYS}"
echo "chains         = ${CHAINS}"
echo "task fits      = $((LMH_N_SCENARIOS * CHAINS))"
echo "shared run dir = ${SHARED_RUN_DIR}"
echo "python         = ${PYTHON_BIN}"
"${PYTHON_BIN}" -c 'import sys, bucex, matplotlib; print("executable", sys.executable); print("bucex", bucex.__version__); print("matplotlib", matplotlib.__version__); assert bucex.__version__ == "1.1.4"'
date
hostname

"${PYTHON_BIN}" -u examples/08_simulation_laplace_mh.py

echo "Finished combining array tasks and generating final outputs"
echo "final output = ${SHARED_RUN_DIR}"
date
