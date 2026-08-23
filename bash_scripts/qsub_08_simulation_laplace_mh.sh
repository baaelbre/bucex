#!/usr/bin/env bash
set -euo pipefail

# Submit the complete example-08 fan-out/fan-in workflow. Configuration is by
# environment variable so the common final run is a single copy-paste command:
#
#   bash bash_scripts/qsub_08_simulation_laplace_mh.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="${BUCEX_PACKAGE_ROOT:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "${PROJECT_ROOT}"
source config/laplace_mh_simulation_array.sh

N_TIME="${N_TIME:-1000}"
PERIOD="${PERIOD:-4}"
SIMULATION_SEED="${SIMULATION_SEED:-13081997}"
DRAWS="${DRAWS:-1000}"
WARMUP="${WARMUP:-1000}"
CHAINS="${CHAINS:-4}"
MCMC_SEED="${MCMC_SEED:-13081997}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_sim_laplace_mh_final}"
OVERWRITE="${OVERWRITE:-0}"
MH_STEPS="${MH_STEPS:-1}"
SCENARIO_KEYS="${SCENARIO_KEYS:-${LMH_DEFAULT_SCENARIO_KEYS}}"
BUCEX_PROGRESS="${BUCEX_PROGRESS:-1}"
DEPENDENCY_KIND="${BUCEX_PBS_DEPENDENCY:-afterok}"

laplace_mh_validate_positive_integer "n time" "${N_TIME}"
laplace_mh_validate_positive_integer "period" "${PERIOD}"
laplace_mh_validate_positive_integer "draws" "${DRAWS}"
laplace_mh_validate_positive_integer "warmup" "${WARMUP}"
laplace_mh_validate_positive_integer "chains" "${CHAINS}"
laplace_mh_validate_positive_integer "MH steps" "${MH_STEPS}"
laplace_mh_parse_scenarios "${SCENARIO_KEYS}"
N_TASKS=$((LMH_N_SCENARIOS * CHAINS))
MAX_CONCURRENT="${MAX_CONCURRENT:-${N_TASKS}}"
laplace_mh_validate_positive_integer "maximum concurrent tasks" "${MAX_CONCURRENT}"
if (( MAX_CONCURRENT > N_TASKS )); then
  MAX_CONCURRENT="${N_TASKS}"
fi

if [[ "${RUN_ID}" == */* || "${RUN_ID}" == *","* ]]; then
  echo "RUN_ID must not contain a slash or comma: ${RUN_ID}" >&2
  exit 2
fi
for value in "${RESULTS_ROOT}" "${SCENARIO_KEYS}" "${BUCEX_VENV_DIR:-}" \
  "${BUCEX_PYTHON:-}" "${BUCEX_PYTHON_MODULE:-}"; do
  if [[ "${value}" == *","* ]]; then
    echo "qsub values must not contain commas: ${value}" >&2
    exit 2
  fi
done
if ! command -v qsub >/dev/null 2>&1; then
  echo "qsub is not available in this shell." >&2
  exit 127
fi

SIGNATURE="$(laplace_mh_run_signature \
  "${N_TIME}" "${PERIOD}" "${DRAWS}" "${WARMUP}" "${CHAINS}" "${MH_STEPS}")"
SHARED_RUN_DIR="$(laplace_mh_run_directory \
  "${RESULTS_ROOT}" "${RUN_ID}" "${SIGNATURE}")"

QSUB_VARS="N_TIME=${N_TIME},PERIOD=${PERIOD},SIMULATION_SEED=${SIMULATION_SEED},DRAWS=${DRAWS},WARMUP=${WARMUP},CHAINS=${CHAINS},MCMC_SEED=${MCMC_SEED},RESULTS_ROOT=${RESULTS_ROOT},RUN_ID=${RUN_ID},OVERWRITE=${OVERWRITE},MH_STEPS=${MH_STEPS},SCENARIO_KEYS=${LMH_SCENARIO_KEYS},BUCEX_PROGRESS=${BUCEX_PROGRESS}"
if [[ -n "${BUCEX_VENV_DIR:-}" ]]; then
  QSUB_VARS+=",BUCEX_VENV_DIR=${BUCEX_VENV_DIR}"
fi
if [[ -n "${BUCEX_PYTHON:-}" ]]; then
  QSUB_VARS+=",BUCEX_PYTHON=${BUCEX_PYTHON}"
fi
if [[ -n "${BUCEX_PYTHON_MODULE:-}" ]]; then
  QSUB_VARS+=",BUCEX_PYTHON_MODULE=${BUCEX_PYTHON_MODULE}"
fi

echo "Submitting ${N_TASKS} independent Laplace-MH tasks"
echo "scenarios       = ${LMH_SCENARIO_KEYS}"
echo "chains          = ${CHAINS}"
echo "draws/warmup    = ${DRAWS}/${WARMUP}"
echo "concurrency     = ${MAX_CONCURRENT}"
echo "shared run dir  = ${SHARED_RUN_DIR}"

FIT_ID="$(qsub \
  -v "${QSUB_VARS}" \
  -t "1-${N_TASKS}%${MAX_CONCURRENT}" \
  job_scripts/submit_08_simulation_laplace_mh_array.pbs)"
echo "fit array       = ${FIT_ID}"

FINAL_ID="$(qsub \
  -v "${QSUB_VARS}" \
  -W "depend=${DEPENDENCY_KIND}:${FIT_ID}" \
  job_scripts/submit_08_simulation_laplace_mh_finalize.pbs)"
echo "finalizer       = ${FINAL_ID}"
echo "results         = ${SHARED_RUN_DIR}"
echo
echo "Monitor with: qstat -u \"${USER:-user}\""
