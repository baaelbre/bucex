#!/bin/bash
set -euo pipefail
trap 'status=$?; printf "BUCEX probe failed: status=%s line=%s host=%s\n" "$status" "$LINENO" "${HOSTNAME:-unknown}" >&2; exit "$status"' ERR
printf 'BUCEX 1.9.4 compute probe starting on %s\nPython requested: %s\n' "${HOSTNAME:-unknown}" "${BUCEX_PYTHON:-unset}"
cd -P "${BUCEX_PROJECT_ROOT:?missing project root}"
if [[ -n "${BUCEX_ENV_SETUP:-}" ]]; then source "$BUCEX_ENV_SETUP"; fi
source bash_scripts/runtime.sh
"$python_exe" -m compileall -q bucex research job_scripts
exec "$python_exe" -u -m research.seasonal.probe --root "$root/${BUCEX_TIER:?}/startup_probes"
