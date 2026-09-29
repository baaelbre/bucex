#!/bin/bash
set -euo pipefail
trap 'status=$?; printf "BUCEX probe failed: status=%s line=%s host=%s\n" "$status" "$LINENO" "${HOSTNAME:-unknown}" >&2; exit "$status"' ERR
printf 'BUCEX 1.9.6 compute probe starting on %s\nPython requested: %s\n' "${HOSTNAME:-unknown}" "${BUCEX_PYTHON:-unset}"
cd -P "${BUCEX_PROJECT_ROOT:?missing project root}"
if [[ -n "${BUCEX_ENV_SETUP:-}" ]]; then source "$BUCEX_ENV_SETUP"; fi
source bash_scripts/runtime.sh
"$python_exe" -m compileall -q bucex research job_scripts
"$python_exe" -u -m research.seasonal.probe --root "$root/${BUCEX_TIER:?}/startup_probes"
if [[ "${BUCEX_SKIP_PRIOR_SIMULATIONS:-0}" != 1 ]]; then
    "$python_exe" -u -m research.seasonal.prior_simulations --root "$root" --tier "$BUCEX_TIER"
fi
