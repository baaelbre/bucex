#!/usr/bin/env bash
# One queue owns the screen and paper resource budget on this BIOBOT host.
set -euo pipefail
cd -P "$(dirname "$0")"
source bash_scripts/runtime.sh
probe=true
for arg in "$@"; do
  case "$arg" in --status|--dry-run|--help|-h) probe=false ;; esac
done
if [[ "$probe" == true ]]; then
  "$python_exe" -m compileall -q bucex research
  "$python_exe" -u -m research.seasonal.probe --root "$root/startup_checks" --max-parallel 1
fi
exec "$python_exe" -u -m research.seasonal.overnight --root "$root" "$@"
