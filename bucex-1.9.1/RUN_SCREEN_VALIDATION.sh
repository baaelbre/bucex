#!/usr/bin/env bash
set -euo pipefail
cd -P "$(dirname "$0")"
exec bash bash_scripts/run_local.sh screen "${1:-validation}" "${@:2}"
