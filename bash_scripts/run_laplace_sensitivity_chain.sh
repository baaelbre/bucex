#!/usr/bin/env bash
set -euo pipefail

SIM_SEED="${1:?simulation seed is required}"
PROFILE="${2:?profile name is required}"
LEVEL_SLAB="${3:?level slab SD is required}"
TREND_SLAB="${4:?trend slab SD is required}"
SEASON_SLAB="${5:?season slab SD is required}"
INITIAL_SEASON_SD="${6:?initial-season prior SD is required}"
CHAIN="${7:?chain number is required}"
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
  echo "Set BUCEX_VENV_DIR or BUCEX_PYTHON when calling qsub." >&2
  exit 2
fi

export MPLBACKEND=Agg
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export BUCEX_RESULTS_ROOT="${RESULTS_ROOT}"
export BUCEX_RUN_ID="s${SIM_SEED}_${PROFILE}_c${CHAIN}"
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
export BUCEX_CHAINS=1
export BUCEX_CHAIN_ONLY=1
export BUCEX_PROGRESS=1
export BUCEX_SEED=$((SIM_SEED + 100000 + CHAIN))

echo "Running one Laplace sensitivity chain"
echo "simulation seed  = ${SIM_SEED}"
echo "profile          = ${PROFILE}"
echo "level slab       = ${LEVEL_SLAB}"
echo "trend slab       = ${TREND_SLAB}"
echo "season slab      = ${SEASON_SLAB}"
echo "initial season   = ${INITIAL_SEASON_SD}"
echo "chain            = ${CHAIN}"
echo "draws/warmup     = ${DRAWS}/${WARMUP}"
echo "results          = ${RESULTS_ROOT}"
echo "python           = ${PYTHON_BIN}"
"${PYTHON_BIN}" -c "import sys, bucex, matplotlib, numpy; print('executable', sys.executable); print('bucex', bucex.__version__); print('matplotlib', matplotlib.__version__)"
date
hostname

"${PYTHON_BIN}" -u examples/03_simulation_laplace.py

echo "Finished Laplace sensitivity chain"
date
