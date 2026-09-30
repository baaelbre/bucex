#!/usr/bin/env bash
set -euo pipefail
cd -P "$(dirname "$0")"
export BUCEX_RESULTS_ROOT="${BUCEX_RESULTS_ROOT:-$PWD/results/serra_1982_monthly}"
source bash_scripts/runtime.sh
exec "$python_exe" -u -m research.seasonal.finish --root "$root" --tier "${BUCEX_TIER:-screen}" \
    --batch "${1:-monthly_all}"
