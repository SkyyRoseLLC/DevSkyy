#!/usr/bin/env bash
set -euo pipefail

THEME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THEME_NAME="$(basename "$THEME_DIR")"
DIST_DIR="$THEME_DIR/dist"
ARCHIVE="$DIST_DIR/${THEME_NAME}.zip"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/skyyrose2-package.XXXXXX")"
STAGE_ARCHIVE="$STAGE_DIR/${THEME_NAME}.zip"
trap 'rm -rf "$STAGE_DIR"' EXIT

cd "$THEME_DIR"
npm run build
npm run verify
mkdir -p "$DIST_DIR" "$STAGE_DIR/$THEME_NAME"

rsync -a ./ "$STAGE_DIR/$THEME_NAME/" \
	--exclude '.DS_Store' \
	--exclude 'dist/' \
	--exclude 'node_modules/' \
	--exclude '.gitignore' \
	--exclude 'assets/css/bundles/' \
	--exclude 'CLAUDE.local.md' \
	--exclude 'package-lock.json' \
	--exclude '*.map'

SYMLINK="$(find "$STAGE_DIR/$THEME_NAME" -type l -print -quit)"
if [[ -n "$SYMLINK" ]]; then
	echo "Refusing to package symbolic link: $SYMLINK" >&2
	exit 1
fi

# A marketplace package is a content artifact, not a workstation snapshot.
# Normalize mtimes to ZIP's earliest portable date, strip platform metadata,
# and feed entries in lexical order so identical source bytes yield one hash.
find "$STAGE_DIR/$THEME_NAME" -exec touch -t 198001010000 {} +
(
	cd "$STAGE_DIR"
	LC_ALL=C find "$THEME_NAME" -print | LC_ALL=C sort | zip -X -q "$STAGE_ARCHIVE" -@
)
mv -f "$STAGE_ARCHIVE" "$ARCHIVE"
echo "Created $ARCHIVE"
