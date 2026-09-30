#!/usr/bin/env bash
set -euo pipefail
cd -P "$(dirname "$0")"
batch=sweetspot_validation
if [[ $# -gt 0 && "$1" != --* ]]; then batch="$1"; shift; fi
export BUCEX_BATCH="$batch"
exec bash RUN_OVERNIGHT_BIOBOT.sh --tier paper --batch "$batch" --cpus "${BUCEX_CPUS:-48}" --max-jobs "${BUCEX_MAX_JOBS:-24}" --memory-gb "${BUCEX_MEMORY_GB:-150}" "$@"
