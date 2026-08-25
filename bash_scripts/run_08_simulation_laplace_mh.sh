#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/_run_example.sh" examples/08_simulation_laplace_mh.py examples/config/simulation.json "$@"
