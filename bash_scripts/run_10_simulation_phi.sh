#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/_run_example.sh" examples/10_simulation_phi.py examples/config/phi/simulation_stationary.json "$@"
