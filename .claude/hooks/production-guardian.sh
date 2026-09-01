#!/usr/bin/env bash
set -euo pipefail

mode="${1:-fast}"
root="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"
guardian="$root/plugins/skyyrose-production-guardian/scripts/guardian.py"

if [[ ! -f "$guardian" ]]; then
  printf '%s\n' '[ProductionGuardian] BLOCKED: plugin runner is missing.' >&2
  exit 2
fi

case "$mode" in
  observe)
    python3 "$guardian" observe --root "$root" --event post-tool
    ;;
  preflight)
    python3 "$guardian" preflight --root "$root"
    ;;
  fast|ship|ci)
    python3 "$guardian" check --root "$root" --mode "$mode"
    ;;
  *)
    printf 'Unknown production guardian mode: %s\n' "$mode" >&2
    exit 2
    ;;
esac
