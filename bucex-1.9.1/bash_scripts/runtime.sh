#!/usr/bin/env bash
# Source after changing to the project root. Preserve the active Python environment.
python_exe="${BUCEX_PYTHON:-python3}"
python_exe="$(command -v -- "$python_exe")" || { echo 'Cannot resolve BUCEX_PYTHON' >&2; exit 2; }
[[ "$python_exe" = /* && -x "$python_exe" ]] || { echo 'BUCEX_PYTHON must resolve to an executable absolute path' >&2; exit 2; }
root="${BUCEX_RESULTS_ROOT:-results/serra_191_parallel}"
mkdir -p "$root" job_scripts/logs
root="$(cd -P "$root" && pwd)"
export BUCEX_PYTHON="$python_exe" BUCEX_RESULTS_ROOT="$root" BUCEX_PROJECT_ROOT="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
