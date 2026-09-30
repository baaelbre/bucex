#!/usr/bin/env bash
# BIOBOT: 51 recent-origin fits. Use --batch sweetspot to run all 162 here.
set -euo pipefail
cd -P "$(dirname "$0")"
export BUCEX_BATCH=sweetspot_validation
exec bash RUN_OVERNIGHT_BIOBOT.sh --tier screen --batch sweetspot_validation \
  --cpus "${BUCEX_CPUS:-48}" --max-jobs "${BUCEX_MAX_JOBS:-24}" \
  --memory-gb "${BUCEX_MEMORY_GB:-150}" "$@"
