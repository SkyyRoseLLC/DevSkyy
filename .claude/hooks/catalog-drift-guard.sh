#!/usr/bin/env bash
# catalog-drift-guard.sh — PostToolUse hook: warn on drift after editing catalog/registry files.
#
# Fires after Edit/Write/MultiEdit. Checks if the edited file is a catalog
# or registry file, and if so runs the validator in --quiet mode.
# Prints a warning (non-blocking) if checks fail — the pre-commit hook blocks.

set -euo pipefail

# ── Resolve paths ──────────────────────────────────────────────────────────
REPO_ROOT="$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null || echo "")"
if [[ -z "$REPO_ROOT" ]]; then
  exit 0
fi

# ── Gate: only fire on catalog/registry files ─────────────────────────────
# Claude Code passes the edited file path via CLAUDE_TOOL_INPUT_FILE_PATH env var.
# If not set (older harness), read from the hook's stdin JSON.
EDITED_FILE="${CLAUDE_TOOL_INPUT_FILE_PATH:-}"

if [[ -z "$EDITED_FILE" ]]; then
  # Claude Code passes tool input as JSON on stdin. Slurp ALL of it — JSON is
  # multi-line, so reading a single line truncates it and silently no-ops the
  # guard. `timeout 1 cat` reads everything but never hangs on an idle stdin.
  STDIN_JSON="$(timeout 1 cat 2>/dev/null || true)"
  if [[ -n "$STDIN_JSON" ]]; then
    EDITED_FILE="$(printf '%s' "$STDIN_JSON" | python3 -c "import sys,json; print(json.load(sys.stdin).get('file_path',''))" 2>/dev/null || true)"
  fi
fi

# Normalize to relative path for pattern matching
REL_FILE="${EDITED_FILE#"$REPO_ROOT/"}"

CATALOG_PATTERNS=(
  "wordpress-theme/skyyrose-flagship/data/skyyrose-catalog.csv"
  "wordpress-theme/skyyrose-flagship/data/visual-manifest.json"
  "wordpress-theme/skyyrose-flagship/data/logo-registry.json"
  "wordpress-theme/skyyrose-flagship/data/product-similarities.json"
  "skyyrose/elite_studio/sku_resolver.py"
  "skyyrose/elite_studio/logo_registry.py"
  "skyyrose/elite_studio/commerce.py"
)

# Per feedback_canonical_sources_only.md (locked 2026-05-27): dossier .md
# files are the second authoritative source and any touch must announce.
DOSSIER_GLOB_PREFIX="wordpress-theme/skyyrose-flagship/data/dossiers/"

MATCHED=0
for pattern in "${CATALOG_PATTERNS[@]}"; do
  if [[ "$REL_FILE" == "$pattern" ]]; then
    MATCHED=1
    break
  fi
done

if [[ "$MATCHED" == "0" && "$REL_FILE" == "$DOSSIER_GLOB_PREFIX"*.md ]]; then
  MATCHED=1
fi

# Per-collection identity.json is the canon SOT master (added 2026-06-14).
if [[ "$MATCHED" == "0" && "$REL_FILE" == wordpress-theme/skyyrose-flagship/data/collections/*/identity.json ]]; then
  MATCHED=1
fi

if [[ "$MATCHED" == "0" ]]; then
  exit 0
fi

# ── Detect Python ──────────────────────────────────────────────────────────
PYTHON=""
for candidate in \
    "${REPO_ROOT}/.venv/bin/python3" \
    "$(command -v python3 2>/dev/null || true)"; do
  if [[ -n "$candidate" && -x "$candidate" ]]; then
    PYTHON="$candidate"
    break
  fi
done

if [[ -z "$PYTHON" ]]; then
  exit 0
fi

# Product SOT is the single machine-facing contract for every product consumer.
# It must refresh after any catalog, logo-placement, or dossier change; the image
# and Woo contracts are hash-bound projections and regenerate in the same pass.
PRODUCT_SOT_TRIGGER=0
if [[ "$REL_FILE" == "wordpress-theme/skyyrose-flagship/data/skyyrose-catalog.csv" \
   || "$REL_FILE" == "wordpress-theme/skyyrose-flagship/data/logo-registry.json" \
   || "$REL_FILE" == "$DOSSIER_GLOB_PREFIX"*.md ]]; then
  PRODUCT_SOT_TRIGGER=1
fi
if [[ "$PRODUCT_SOT_TRIGGER" == "1" ]]; then
  if (cd "$REPO_ROOT" \
    && "$PYTHON" -m skyyrose.core.product_sot >/dev/null \
    && "$PYTHON" -m skyyrose.core.sot_images >/dev/null \
    && "$PYTHON" scripts/launch/woocommerce_product_contract.py >/dev/null); then
    echo ""
    echo "[catalog-drift-guard] product-sot.json + sot-images.json + Woo contract regenerated after ${REL_FILE} edit."
    echo ""
  else
    echo ""
    echo "[catalog-drift-guard] WARNING: product contract regeneration FAILED after editing ${REL_FILE}."
    echo "  Run: python3 -m skyyrose.core.product_sot"
    echo ""
  fi
fi

VALIDATOR="${REPO_ROOT}/scripts/validate_catalog_consistency.py"
if [[ ! -f "$VALIDATOR" ]]; then
  exit 0
fi

# ── Run validator (quiet, non-blocking) ────────────────────────────────────
CHECKS="dossier_slugs,logo_skus,csv_image_columns_resolve,product_sot_current,sot_images_current,woocommerce_sync_current"
if ! "$PYTHON" "$VALIDATOR" --quiet --checks "$CHECKS" 2>/dev/null; then
  echo ""
  echo "[catalog-drift-guard] WARNING: Catalog consistency check failed after editing ${REL_FILE}."
  echo "  Run: python3 scripts/validate_catalog_consistency.py --checks $CHECKS"
  echo "  Legacy collection/lookbook generators remain disabled until production is clean."
  echo "  The pre-commit hook will block commits until checks pass."
  echo ""
fi

# Always announce the touch so the active agent re-reads the source before
# any subsequent product claim. Memory may be stale; canonical files won.
echo ""
echo "[catalog-drift-guard] CANONICAL PRODUCT DATA TOUCHED: ${REL_FILE}"
echo "  Re-read data/product-sot.json (or call skyyrose.core.product_sot)"
echo "  before claiming any product fact or dispatching creative work."
echo ""

exit 0
