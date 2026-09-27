#!/bin/bash -l
set -euo pipefail
cd "${BUCEX_PROJECT_ROOT:-${SLURM_SUBMIT_DIR:?submit from release root}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
"${BUCEX_PYTHON:-python3}" -u -m research.seasonal.jobs \
    --tier "${BUCEX_TIER:?missing tier}" --batch "${BUCEX_BATCH:?missing batch}" \
    --resource "${BUCEX_RESOURCE:?missing resource}" --root "${BUCEX_RESULTS_ROOT:-results/serra_190_parallel}" \
    --index "${SLURM_ARRAY_TASK_ID:?missing array index}"
