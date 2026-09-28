#!/usr/bin/env bash
set -euo pipefail
cd -P "$(dirname "$0")"
batch=experiments
if [[ $# -gt 0 && "$1" != --* ]]; then batch="$1"; shift; fi
exec bash bash_scripts/submit.sh screen "$batch" "$@"
