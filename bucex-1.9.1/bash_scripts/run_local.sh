#!/usr/bin/env bash
# Bounded fit concurrency for biobot. Invoke only one batch/tier at a time.
set -euo pipefail
cd -P "$(dirname "$0")/.."
source bash_scripts/runtime.sh
parallel="${BUCEX_MAX_JOBS:-4}"
tier="${1:-screen}";batch="${2:-validation}"
[[ $# -le 3 && ( $# -lt 3 || "$3" == --dry-run ) ]] || { echo 'Use screen|paper [batch] [--dry-run]' >&2; exit 2; }
[[ "$parallel" =~ ^[1-9][0-9]*$ ]] || { echo 'BUCEX_MAX_JOBS must be positive' >&2; exit 2; }
"$python_exe" -m research.seasonal.jobs --tier "$tier" --batch "$batch" --verify --root "$root"
if [[ "${3:-}" == --dry-run ]]; then
    "$python_exe" -m research.seasonal.jobs --tier "$tier" --batch "$batch" --list
    printf 'Maximum simultaneous fits: %s\n' "$parallel"
    exit 0
fi
"$python_exe" -m research.seasonal.jobs --tier "$tier" --batch "$batch" --list | awk '{print $1}' | \
  xargs -P "$parallel" -I '{}' bash -c '
    index="$1";python_exe="$2";tier="$3";batch="$4";root="$5"
    "$python_exe" -u -m research.seasonal.jobs --tier "$tier" --batch "$batch" --index "$index" --root "$root" \
      >"job_scripts/logs/local_${tier}_${batch}_${index}.log" 2>&1
  ' _ '{}' "$python_exe" "$tier" "$batch" "$root"
