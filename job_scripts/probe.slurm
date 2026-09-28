#!/bin/bash
set -euo pipefail
cd -P "${BUCEX_PROJECT_ROOT:?missing project root}"
exec bash -l job_scripts/run_probe.sh
