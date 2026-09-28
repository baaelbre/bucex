#!/usr/bin/env bash
# Bounded local execution; one task is a separate response fit or one coupled shared fit.
set -euo pipefail
cd -P "$(dirname "$0")/.."
source bash_scripts/runtime.sh
tier="${1:-screen}"; batch="${2:-experiments}"
if [[ $# -gt 0 ]]; then shift; fi
if [[ $# -gt 0 ]]; then shift; fi
exec "$python_exe" -u -m research.seasonal.local --tier "$tier" --batch "$batch" --root "$root" "$@"
