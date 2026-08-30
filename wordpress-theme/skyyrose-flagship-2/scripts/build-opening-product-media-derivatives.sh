#!/usr/bin/env bash
# Regenerate the tracked, bounded WebP card derivatives from the exact
# founder-admitted product fronts. This is a local source build helper; the
# marketplace package contains only the resulting derivatives.
set -euo pipefail

THEME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCE_THEME_DIR="$(cd "$THEME_DIR/../skyyrose-flagship" && pwd)"
MANIFEST="$THEME_DIR/data/opening-product-media.json"

command -v magick >/dev/null || { echo 'ImageMagick (magick) is required.' >&2; exit 1; }

node - "$MANIFEST" <<'NODE' | while IFS=$'\t' read -r source derivative; do
const fs = require('fs');
const manifest = JSON.parse(fs.readFileSync(process.argv[2], 'utf8'));
for (const record of Object.values(manifest.products || {})) {
  for (const view of record.views || []) {
    if (view.role === 'on_model_front' && view.admission === 'founder_launch_front') {
      if (!view.source || !view.derivative) throw new Error('Every admitted front needs source and derivative paths.');
      console.log(`${view.source}\t${view.derivative}`);
    }
  }
}
NODE
	[[ -f "$SOURCE_THEME_DIR/$source" ]] || { echo "Missing admitted source: $source" >&2; exit 1; }
	mkdir -p "$(dirname "$THEME_DIR/$derivative")"
	magick "$SOURCE_THEME_DIR/$source" -resize '600x800>' -strip -quality 76 -define webp:method=6 "$THEME_DIR/$derivative"
	bytes="$(stat -f '%z' "$THEME_DIR/$derivative")"
	if [[ "$bytes" -gt 80000 ]]; then
		magick "$SOURCE_THEME_DIR/$source" -resize '600x800>' -strip -quality 68 -define webp:method=6 "$THEME_DIR/$derivative"
		bytes="$(stat -f '%z' "$THEME_DIR/$derivative")"
	fi
	[[ "$bytes" -le 80000 ]] || { echo "Derivative exceeds 80KB: $derivative ($bytes)" >&2; exit 1; }
	done
