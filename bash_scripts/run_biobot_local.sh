#!/usr/bin/env bash
# BIOBOT login/node fallback: run disjoint fit tasks with bounded concurrency.
set -euo pipefail
cd "$(dirname "$0")/.."
python_exe="${BUCEX_PYTHON:-python3}"
parallel="${BUCEX_MAX_JOBS:-4}"
tier="${1:-screen}"
mkdir -p job_scripts/logs
"$python_exe" -m research.seasonal.jobs --list | awk '{print $1}' | \
  xargs -P "$parallel" -I '{}' bash -c '
    index="$1"; python_exe="$2"; tier="$3"
    export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
    "$python_exe" -u -m research.seasonal.jobs --tier "$tier" --index "$index" \
       >"job_scripts/logs/${tier}_${index}.log" 2>&1
  ' _ '{}' "$python_exe" "$tier"
