#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  ./scripts/make-pr.sh --title "<PR Title>" --problem "<text>" --implementation "<text>" --testing "<text>" \
    [--issue "<issue-or-bug-id>"] [--screenshots "<artifact links>"] [--notes "<migrations/env/security/deploy notes>"] \
    [--base main] [--dry-run]

Required:
  --title
  --problem
  --implementation
  --testing

The script enforces a clean working tree before creating a PR.
It requires gh and an authenticated session ("gh auth login").
If --dry-run is set, it prints the generated body instead of creating the PR.
USAGE
}

TITLE=""
PROBLEM=""
IMPLEMENTATION=""
TESTING=""
ISSUE_ID=""
SCREENSHOTS=""
NOTES=""
BASE_BRANCH="main"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --title)
      TITLE="$2"
      shift 2
      ;;
    --problem)
      PROBLEM="$2"
      shift 2
      ;;
    --implementation)
      IMPLEMENTATION="$2"
      shift 2
      ;;
    --testing)
      TESTING="$2"
      shift 2
      ;;
    --issue)
      ISSUE_ID="$2"
      shift 2
      ;;
    --screenshots)
      SCREENSHOTS="$2"
      shift 2
      ;;
    --notes)
      NOTES="$2"
      shift 2
      ;;
    --base)
      BASE_BRANCH="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown flag: $1"
      usage
      exit 1
      ;;
  esac
done

if [[ -z "${TITLE}" || -z "${PROBLEM}" || -z "${IMPLEMENTATION}" || -z "${TESTING}" ]]; then
  echo "Missing required argument(s)." >&2
  usage
  exit 1
fi

if ! command -v gh >/dev/null 2>&1; then
  echo "GitHub CLI is required: install and run gh auth login first." >&2
  exit 1
fi

CURRENT_BRANCH="$(git branch --show-current)"
if [[ -z "${CURRENT_BRANCH}" ]]; then
  echo "Cannot determine current branch (detached HEAD)." >&2
  exit 1
fi

TMP_BODY="$(mktemp)"
trap 'rm -f "${TMP_BODY}"' EXIT

cat <<EOF >"${TMP_BODY}"
### Problem statement
${PROBLEM}

### Implementation summary
${IMPLEMENTATION}

### Test evidence
${TESTING}

### Linked issue or bug ID
${ISSUE_ID:-None}

### UI screenshots or recordings
${SCREENSHOTS:-None}

### Migration, environment, security, and deployment notes
${NOTES:-None}
EOF

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "=== make-pr dry-run body ==="
  echo
  cat "${TMP_BODY}"
  exit 0
fi

if [[ -n "$(git status --short)" ]]; then
  echo "Refusing to create PR with uncommitted work in working tree." >&2
  echo "Run git status --short and commit only intended changes first." >&2
  exit 1
fi

if ! gh auth status >/dev/null 2>&1; then
  echo "GitHub CLI authentication required. Run: gh auth login" >&2
  exit 1
fi

if ! git remote get-url origin >/dev/null 2>&1; then
  echo "No git remote named origin. Add one before creating PR." >&2
  exit 1
fi

echo "Creating PR: ${CURRENT_BRANCH} -> ${BASE_BRANCH}"
gh pr create \
  --title "${TITLE}" \
  --body-file "${TMP_BODY}" \
  --base "${BASE_BRANCH}" \
  --head "${CURRENT_BRANCH}" \
  --repo "$(git remote get-url origin)" \
  --push
