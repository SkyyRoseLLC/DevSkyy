#!/usr/bin/env bash
# Auto-format staged WordPress PHP files with the repository's WPCS rules.
#
# PHPCBF returns 1 when it successfully fixes violations, so this wrapper
# normalizes 0 and 1 to success while preserving real tool/config failures.

set -uo pipefail

REPO_ROOT="$(git rev-parse --show-toplevel 2> /dev/null || pwd)"
THEME_DIR="$REPO_ROOT/wordpress-theme/skyyrose-flagship"
PHPCBF="$THEME_DIR/vendor/bin/phpcbf"
STANDARD="$THEME_DIR/.phpcs.xml"

if [[ ! -x "$PHPCBF" ]]; then
  echo "PHP formatter missing: $PHPCBF" >&2
  echo "Run: cd '$THEME_DIR' && composer install" >&2
  exit 2
fi

if [[ ! -f "$STANDARD" ]]; then
  echo "PHP coding standard missing: $STANDARD" >&2
  exit 2
fi

declare -a files=()
for file in "$@"; do
  case "$file" in
    -*) continue ;;
  esac
  [[ -f "$file" ]] || continue
  files+=("$file")
done

if [[ ${#files[@]} -eq 0 ]]; then
  exit 0
fi

"$PHPCBF" --standard="$STANDARD" --extensions=php --report=summary "${files[@]}"
status=$?

# 0: already clean. 1: all fixable violations were repaired.
if [[ "$status" -le 1 ]]; then
  exit 0
fi

exit "$status"
