#!/usr/bin/env bash
set -euo pipefail

THEME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THEME_NAME="$(basename "$THEME_DIR")"
DIST_DIR="$THEME_DIR/dist"
ARCHIVE="$DIST_DIR/${THEME_NAME}.zip"
STAGE_DIR="$(mktemp -d "${TMPDIR:-/tmp}/skyyrose2-package.XXXXXX")"
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
	--exclude 'package-lock.json' \
	--exclude '*.map'

find "$STAGE_DIR/$THEME_NAME" -name '.DS_Store' -delete
rm -f "$ARCHIVE"
( cd "$STAGE_DIR" && zip -qr "$ARCHIVE" "$THEME_NAME" )
echo "Created $ARCHIVE"
