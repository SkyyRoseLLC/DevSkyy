#!/usr/bin/env bash
set -euo pipefail

plugin_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
portable_root="$(mktemp -d)"
trap 'rm -r "${portable_root}"' EXIT

tar -C "${plugin_root}" --exclude='.git' --exclude='__pycache__' -cf - . | tar -C "${portable_root}" -xf -

developer_path_pattern='/Users/the''ceo/DevSkyy|/Users/the''ceo/plugins'
if rg -n "${developer_path_pattern}" \
  "${portable_root}/scripts/generate-branded-skill-enhancements.py" \
  "${portable_root}/scripts/capture-branded-authorities.py" \
  "${portable_root}/scripts/validate-branded-evidence.py" \
  "${portable_root}/scripts/verify.sh" \
  "${portable_root}/skills/fashion-branded-skills" \
  "${portable_root}/skills/fashion-theme-team/brain/branded-skills" \
  "${portable_root}/vendor" \
  "${portable_root}/.codex-plugin/plugin.json" >/dev/null; then
  printf 'FAIL: portable package contains a developer-machine dependency\n' >&2
  exit 1
fi

FASHION_SKIP_PORTABLE_CHECK=1 bash "${portable_root}/scripts/verify.sh" >/dev/null
printf 'PASS: clean copied package verifies without the developer source tree\n'
