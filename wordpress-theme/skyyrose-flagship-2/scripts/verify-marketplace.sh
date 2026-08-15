#!/usr/bin/env bash
set -euo pipefail

THEME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$THEME_DIR"

find . -name '*.php' -not -path './vendor/*' -print0 | xargs -0 -n1 php -l >/dev/null
for document in theme.json data/product-presentation-registry.json; do
	jq empty "$document"
done

python3 scripts/build-product-presentation-registry.py --check
python3 scripts/build-pot.py --check
php scripts/test-marketplace-registry.php
node scripts/build-assets.mjs --check

STYLE_VERSION="$(sed -n 's/^Version:[[:space:]]*//p' style.css | head -n1)"
PACKAGE_VERSION="$(jq -r '.version' package.json)"
STABLE_TAG="$(sed -n 's/^Stable tag:[[:space:]]*//p' readme.txt | head -n1)"
if [[ -z "$STYLE_VERSION" || "$STYLE_VERSION" != "$PACKAGE_VERSION" || "$STYLE_VERSION" != "$STABLE_TAG" ]]; then
	echo "Theme version drift: style=$STYLE_VERSION package=$PACKAGE_VERSION readme=$STABLE_TAG" >&2
	exit 1
fi

for required in \
	404.php archive.php front-page.php home.php index.php page.php search.php single.php \
	template-collection.php woocommerce/archive-product.php woocommerce/cart/cart.php \
	woocommerce/checkout/form-checkout.php woocommerce/checkout/thankyou.php \
	woocommerce/content-product.php woocommerce/single-product.php \
	theme.json screenshot.png readme.txt README.md CHANGELOG.md LICENSE.txt rtl.css editor-style.css \
	npm-shrinkwrap.json \
	languages/skyyrose-flagship-2.pot; do
	test -s "$required" || { echo "Missing required marketplace artifact: $required" >&2; exit 1; }
done

if rg -n --glob '!node_modules/**' --glob '!dist/**' --glob '!scripts/verify-marketplace.sh' --glob '!README.md' --glob '!readme.txt' --glob '!CHANGELOG.md' \
	'\b(TODO|FIXME|Lorem ipsum|dummy data)\b' .; then
	echo 'Placeholder marker found in delivered theme.' >&2
	exit 1
fi

if rg -n --glob '!scripts/verify-marketplace.sh' 'Cormorant Garamond|Playfair Display|Bebas Neue|Yellowtail' \
	theme.json editor-style.css rtl.css inc scripts; then
	echo 'Retired font found in marketplace architecture.' >&2
	exit 1
fi

echo 'SkyyRose Flagship 2 marketplace verification passed.'
