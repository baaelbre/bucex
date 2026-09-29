#!/usr/bin/env bash
# Source on a compute node. Clear an inherited Python module before loading the
# environment's own module; never activate this architecture-specific venv on login.
if [[ -n "${BUCEX_ENV_SETUP:-}" ]]; then
    [[ -r "$BUCEX_ENV_SETUP" ]] || { printf 'Cannot read environment setup: %s\n' "$BUCEX_ENV_SETUP" >&2; exit 2; }
    if type module >/dev/null 2>&1; then module unload Python >/dev/null 2>&1 || true; fi
    unset PYTHONHOME VIRTUAL_ENV
    source "$BUCEX_ENV_SETUP"
fi
