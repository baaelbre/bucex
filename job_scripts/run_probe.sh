#!/bin/bash
set -euo pipefail
trap 'status=$?; printf "BUCEX probe failed: status=%s line=%s host=%s\n" "$status" "$LINENO" "${HOSTNAME:-unknown}" >&2; exit "$status"' ERR
printf 'BUCEX 1.9.8 compute probe starting on %s\nPython requested: %s\n' "${HOSTNAME:-unknown}" "${BUCEX_PYTHON:-unset}"
cd -P "${BUCEX_PROJECT_ROOT:?missing project root}"
source bash_scripts/compute_environment.sh
source bash_scripts/runtime.sh
"$python_exe" -m compileall -q bucex research job_scripts
"$python_exe" -u -m research.seasonal.probe --root "$root/${BUCEX_TIER:?}/startup_probes"
if [[ "${BUCEX_SKIP_PRIOR_SIMULATIONS:-0}" != 1 ]]; then
    suite=core; draws=2000
    if [[ "${BUCEX_BATCH:-}" == sweetspot* ]]; then suite=sweetspot; draws=1000; fi
    if [[ "$BUCEX_TIER" == paper ]]; then
        draws=10000
        if [[ "$suite" == sweetspot ]]; then draws=5000; fi
    fi
    "$python_exe" -u -m research.seasonal.prior_simulations --root "$root" --tier "$BUCEX_TIER" --suite "$suite" --draws "$draws"
fi
