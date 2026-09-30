#!/bin/bash -l
set -euo pipefail
trap 'status=$?; printf "Collection failed: status=%s line=%s host=%s\n" "$status" "$LINENO" "${HOSTNAME:-unknown}" >&2; exit "$status"' ERR
printf 'BUCEX 1.9.8.3 collection on %s\n' "${HOSTNAME:-unknown}"
cd -P "${BUCEX_PROJECT_ROOT:?}"
source bash_scripts/compute_environment.sh
source bash_scripts/runtime.sh
export MPLBACKEND=Agg
if [[ "${BUCEX_FINISH:-0}" == 1 ]]; then
    complete=()
    [[ "${BUCEX_REQUIRE_COMPLETE:-0}" != 1 ]] || complete+=(--require-complete)
    exec "$python_exe" -u -m research.seasonal.finish --root "$root" --tier "$BUCEX_TIER" --batch "$BUCEX_BATCH" "${complete[@]}"
fi
args=()
[[ "${BUCEX_COLLECT_FIGURES:-0}" != 1 ]] || args+=(--figures)
[[ "${BUCEX_REQUIRE_COMPLETE:-0}" != 1 ]] || args+=(--require-complete)
"$python_exe" -u -m research.seasonal.collect_jobs --root "$root" \
    --tier "${BUCEX_TIER:?}" --batch "${BUCEX_BATCH:?}" "${args[@]}"
archive="${BUCEX_OUTPUT_ARCHIVE:-$root/bucex-1.9.8.3-${BUCEX_TIER}-${BUCEX_BATCH}-${SLURM_JOB_ID:?}.zip}"
"$python_exe" -u -m research.seasonal.export_results --root "$root" --tier "$BUCEX_TIER" --output "$archive"
printf 'Compact review archive: %s\nCollected figures and tables: %s/%s/collected/%s\n' "$archive" "$root" "$BUCEX_TIER" "$BUCEX_BATCH"
