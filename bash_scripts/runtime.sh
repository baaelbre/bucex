#!/usr/bin/env bash
# Source after changing to the project root. Preserve the active Python environment.
python_request="${BUCEX_PYTHON:-python3}"
python_exe="$(command -v -- "$python_request")" || {
    printf 'Cannot resolve BUCEX_PYTHON=%s on host %s\n' "$python_request" "${HOSTNAME:-unknown}" >&2
    echo 'Select a compute-visible Python path before submission; use cd -P to resolve directory aliases.' >&2
    exit 2
}
# Resolve directory aliases such as /user/data -> /kyukon/data on the submit host.
# Do not dereference bin/python itself: its symlink belongs to the virtual env.
python_dir="$(cd -P -- "$(dirname -- "$python_exe")" && pwd -P)" || exit 2
python_exe="$python_dir/$(basename -- "$python_exe")"
[[ -x "$python_exe" ]] || { printf 'Python is not executable: %s\n' "$python_exe" >&2; exit 2; }
if ! "$python_exe" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else "BUCEX requires Python >= 3.10; found " + sys.version.split()[0])'; then
    printf 'Cannot run a supported Python at %s on host %s; check the virtual environment and its module.\n' "$python_exe" "${HOSTNAME:-unknown}" >&2
    exit 2
fi
root="${BUCEX_RESULTS_ROOT:-results/serra_193_parallel}"
mkdir -p "$root" job_scripts/logs
root="$(cd -P "$root" && pwd)"
export BUCEX_PYTHON="$python_exe" BUCEX_RESULTS_ROOT="$root" BUCEX_PROJECT_ROOT="$PWD"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
