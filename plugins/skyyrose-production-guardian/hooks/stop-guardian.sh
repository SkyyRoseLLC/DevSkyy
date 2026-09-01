#!/usr/bin/env bash
set -euo pipefail

root="${CLAUDE_PROJECT_DIR:-$PWD}"
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/guardian.py" learn --root "$root" >/dev/null
python3 "${CLAUDE_PLUGIN_ROOT}/scripts/guardian.py" check --root "$root" --mode fast
