#!/usr/bin/env bash
set -euo pipefail

root="${CLAUDE_PROJECT_DIR:-$PWD}"
if [[ -d "$root/.git" || -f "$root/.git" ]]; then
  printf '%s\n' '[ProductionGuardian] Active: corrections become regression locks; legacy production sources, stale artifacts, and unverified ship claims fail closed.'
fi
