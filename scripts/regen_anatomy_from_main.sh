#!/usr/bin/env bash
# Regenerate .wolf/anatomy.md scoped to git main and origin/main.
#
# Uses the repository-local generator so anatomy maintenance does not execute a
# third-party browser tool or require its vulnerable dependency tree.
#
# Idempotent. Safe to run any number of times. Output is deterministic given
# the same git state.
#
# Exit code 0 on success, non-zero if generation fails.

set -euo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel)"
cd "$REPO_ROOT"

python3 scripts/anatomy_filter_main.py

# If the timestamp is the only diff vs HEAD, revert the file so the working
# tree stays clean and post-commit hooks cannot create a regeneration loop.
ANATOMY_TIMESTAMP_RE='^> Auto-maintained locally\. Last scanned:'
if git diff --quiet -I "$ANATOMY_TIMESTAMP_RE" -- .wolf/anatomy.md 2>/dev/null; then
  git checkout -- .wolf/anatomy.md 2>/dev/null || true
fi
