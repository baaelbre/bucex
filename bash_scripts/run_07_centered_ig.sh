#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${HERE}/_run_example.sh" examples/07_centered_ig.py examples/config/centered_ig.json "$@"
