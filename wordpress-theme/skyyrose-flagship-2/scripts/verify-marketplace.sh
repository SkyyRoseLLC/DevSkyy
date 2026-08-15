#!/usr/bin/env bash
set -euo pipefail

THEME_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$THEME_DIR"

find . -name '*.php' -not -path './vendor/*' -print0 | xargs -0 -n1 php -l >/dev/null
for document in theme.json data/product-presentation-registry.json; do
	jq empty "$document"
done

python3 scripts/build-product-presentation-registry.py --check
node scripts/build-assets.mjs --check

for required in \
	404.php archive.php front-page.php home.php index.php page.php search.php single.php \
	template-collection.php woocommerce/archive-product.php woocommerce/cart/cart.php \
	woocommerce/checkout/form-checkout.php woocommerce/checkout/thankyou.php \
	woocommerce/content-product.php woocommerce/single-product.php \
	theme.json screenshot.png readme.txt README.md CHANGELOG.md LICENSE.txt rtl.css editor-style.css \
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
