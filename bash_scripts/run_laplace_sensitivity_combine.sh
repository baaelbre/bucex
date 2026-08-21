#!/usr/bin/env bash
set -euo pipefail

SIM_SEED="${1:?simulation seed is required}"
PROFILE="${2:?profile name is required}"
LEVEL_SLAB="${3:?level slab SD is required}"
TREND_SLAB="${4:?trend slab SD is required}"
SEASON_SLAB="${5:?season slab SD is required}"
INITIAL_SEASON_SD="${6:?initial-season prior SD is required}"
N_CHAINS="${7:-4}"
DRAWS="${8:-1000}"
WARMUP="${9:-1000}"
N_TIME="${10:-1000}"
PERIOD="${11:-4}"
RESULTS_ROOT="${12:-results/laplace_sensitivity}"

PROJECT_ROOT="${PBS_O_WORKDIR:-$(pwd)}"
cd "${PROJECT_ROOT}"

if [[ -n "${BUCEX_PYTHON_MODULE:-}" ]]; then
  if ! type module >/dev/null 2>&1; then
    source /etc/profile.d/modules.sh 2>/dev/null || true
  fi
  module load "${BUCEX_PYTHON_MODULE}"
fi

VENV_DIR="${BUCEX_VENV_DIR:-${BUCEX_VENV:-${HOME}/venvs/bucex_env}}"
if [[ -n "${BUCEX_PYTHON:-}" ]]; then
  PYTHON_BIN="${BUCEX_PYTHON}"
elif [[ -x "${VENV_DIR}/bin/python" ]]; then
  PYTHON_BIN="${VENV_DIR}/bin/python"
else
  echo "Virtual-environment Python not found: ${VENV_DIR}/bin/python" >&2
  exit 2
fi

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

FIT_SIGNATURE="n${N_TIME}p${PERIOD}_d${DRAWS}w${WARMUP}c1"
FIT_ROOT="${RESULTS_ROOT}/03_simulation_laplace"
CHAIN_DIRS=()
for ((CHAIN = 1; CHAIN <= N_CHAINS; CHAIN++)); do
  CHAIN_DIR="${FIT_ROOT}/s${SIM_SEED}_${PROFILE}_c${CHAIN}__${FIT_SIGNATURE}"
  if [[ ! -d "${CHAIN_DIR}" ]]; then
    echo "Missing chain directory: ${CHAIN_DIR}" >&2
    exit 3
  fi
  CHAIN_DIRS+=("${CHAIN_DIR}")
done

COMBINE_RUNS="$(IFS=:; echo "${CHAIN_DIRS[*]}")"

export BUCEX_RESULTS_ROOT="${RESULTS_ROOT}"
export BUCEX_RUN_ID="s${SIM_SEED}_${PROFILE}_combined"
export BUCEX_OVERWRITE=0
export BUCEX_N_TIME="${N_TIME}"
export BUCEX_PERIOD="${PERIOD}"
export BUCEX_SIMULATION_SEED="${SIM_SEED}"
export BUCEX_LEVEL_SLAB_SD="${LEVEL_SLAB}"
export BUCEX_TREND_SLAB_SD="${TREND_SLAB}"
export BUCEX_SEASON_SLAB_SD="${SEASON_SLAB}"
export BUCEX_INITIAL_SEASON_PRIOR_SD="${INITIAL_SEASON_SD}"
export BUCEX_DRAWS="${DRAWS}"
export BUCEX_WARMUP="${WARMUP}"
export BUCEX_CHAINS="${N_CHAINS}"
export BUCEX_CHAIN_ONLY=0
export BUCEX_PROGRESS=0
export BUCEX_COMBINE_RUNS="${COMBINE_RUNS}"

echo "Combining Laplace sensitivity chains"
echo "simulation seed  = ${SIM_SEED}"
echo "profile          = ${PROFILE}"
echo "chains           = ${N_CHAINS}"
echo "source runs      = ${COMBINE_RUNS}"
date
hostname

"${PYTHON_BIN}" -c "import sys, bucex, matplotlib; print('executable', sys.executable); print('bucex', bucex.__version__); print('matplotlib', matplotlib.__version__)"
"${PYTHON_BIN}" -u examples/03_simulation_laplace.py

echo "Finished combining chains and generating output"
date
