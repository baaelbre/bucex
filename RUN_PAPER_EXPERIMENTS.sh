#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
exec bash bash_scripts/submit_biobot.sh paper
