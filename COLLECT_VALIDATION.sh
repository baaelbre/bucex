#!/usr/bin/env bash
set -euo pipefail
cd -P "$(dirname "$0")"
tier="${1:-screen}"
selection="${2:-reference}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
case "$selection" in reference|fixed|all) ;; *) echo 'Select reference, fixed, or all.' >&2; exit 2;; esac
export BUCEX_RESULTS_ROOT="${BUCEX_RESULTS_ROOT:-$PWD/results/serra_1984_validation}"
source bash_scripts/runtime.sh
exec "$python_exe" -u -m research.seasonal.finish --root "$root" --tier "$tier" --batch "horizon_$selection" "$@"
