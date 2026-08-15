#!/usr/bin/env bash
# V2 candidate self-healing gate: source failures fail before staging does.
set -euo pipefail

THEME_DIR="${1:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
REPO_ROOT="$(cd "$THEME_DIR/../.." && pwd)"

require_file() {
	if [[ ! -f "$THEME_DIR/$1" ]]; then
		echo "FAIL missing required V2 route: $1" >&2
		exit 1
	fi
}

for route in \
	front-page.php page.php home.php single.php archive.php search.php 404.php page-wishlist.php \
	template-collection.php functions.php header.php footer.php \
	template-parts/home/kids-capsule-reveal.php \
	woocommerce/archive-product.php woocommerce/content-product.php woocommerce/single-product.php \
	woocommerce/cart/cart.php woocommerce/checkout/form-checkout.php woocommerce/checkout/thankyou.php; do
	require_file "$route"
done

for approved_visual in \
	assets/sot/images/hero/black-rose-lake-merritt-monument-v2.png \
	assets/sot/images/hero/black-rose-typography-star-salon-v3.png \
	assets/sot/images/hero/love-hurts-golden-gate-monument-v2.png \
	assets/sot/images/hero/signature-golden-gate-monuments-v2.webp \
	assets/sot/images/hero/black-rose-bay-bridge-monuments-v4.webp \
	assets/sot/images/hero/love-hurts-rose-aisle-monuments-v3.webp \
	assets/sot/images/hero/kids-capsule-heir-throne-v3.webp \
	assets/sot/images/home/on-model/signature-sg-005.webp \
	assets/sot/images/home/on-model/black-rose-br-004.webp \
	assets/sot/images/home/on-model/love-hurts-lh-004.webp \
	assets/sot/images/home/on-model/kids-capsule-kids-001.webp \
	assets/sot/images/home/on-model/kids-capsule-mascot-red-guard.webp \
	assets/sot/images/home/on-model/kids-capsule-mascot-purple-guard.webp \
	assets/sot/images/products/kids-002-purple-set.jpeg \
	assets/sot/images/products/kids-001-product-proof-400.webp \
	assets/sot/images/products/kids-002-product-proof-400.webp \
	assets/sot/brand/skyyrose-logo-animated-384w.webp \
	assets/sot/brand/skyyrose-logo-still-384w.webp; do
	require_file "$approved_visual"
done

if ! rg -q "function skyyrose2_asset_suffix" "$THEME_DIR/functions.php"; then
	echo "FAIL asset fallback missing: V2 must never enqueue absent minified bundles" >&2
	exit 1
fi

if ! rg -q "skyyrose2_seo_head" "$THEME_DIR/functions.php" || ! rg -q "skyyrose2_schema_head" "$THEME_DIR/functions.php"; then
	echo "FAIL SEO metadata/schema adapter missing" >&2
	exit 1
fi

if ! rg -q "woocommerce_add_to_cart_fragments" "$THEME_DIR/functions.php"; then
	echo "FAIL cart fragment adapter missing" >&2
	exit 1
fi

if ! rg -q "data-brand-animation" "$THEME_DIR/functions.php" || ! rg -q "data-brand-animation" "$THEME_DIR/assets/js/theme.js"; then
	echo "FAIL optimized SkyyRose global header animation adapter missing" >&2
	exit 1
fi

if rg -q "preventDefault" "$THEME_DIR/assets/js/theme.js"; then
	echo "FAIL scroll/input cancellation detected in V2 motion runtime" >&2
	exit 1
fi

if ! rg -q "data-home-model-loop" "$THEME_DIR/front-page.php" || \
	! rg -q "data-home-model-toggle" "$THEME_DIR/front-page.php" || \
	! rg -q "data-loop-copy" "$THEME_DIR/front-page.php"; then
	echo "FAIL V2 homepage model loop markup or accessible controls missing" >&2
	exit 1
fi

hero_model_block="$(sed -n '/\$hero_models = array(/,/^);/p' "$THEME_DIR/front-page.php")"
hero_model_count="$(printf '%s\n' "$hero_model_block" | rg -c "'slug'[[:space:]]*=>" || true)"
if [ "$hero_model_count" -ne 3 ] || printf '%s\n' "$hero_model_block" | rg -q "kids-capsule"; then
	echo "FAIL V2 homepage opening hero must contain only Signature, Black Rose, and Love Hurts; Kids Capsule is reserved for its dedicated reveal" >&2
	exit 1
fi

for opening_collection in signature black-rose love-hurts; do
	if ! printf '%s\n' "$hero_model_block" | rg -q "'slug'[[:space:]]*=>[[:space:]]*'$opening_collection'"; then
		echo "FAIL V2 homepage opening hero missing $opening_collection" >&2
		exit 1
	fi
done

if ! rg -q "heroModelLoop" "$THEME_DIR/assets/js/theme.js" || \
	! rg -q "sr-home-model-loop" "$THEME_DIR/assets/css/theme.css" || \
	! rg -q "prefers-reduced-motion: reduce" "$THEME_DIR/assets/css/theme.css"; then
	echo "FAIL V2 homepage model loop runtime or motion fallback missing" >&2
	exit 1
fi

if ! rg -q "data-feature-id=\"kids-royal-procession-v1\"" "$THEME_DIR/template-parts/home/kids-capsule-reveal.php" || \
	! rg -q "data-house-royal-procession" "$THEME_DIR/template-parts/home/kids-capsule-reveal.php" || \
	! rg -q "data-procession-chapter=\"invitation\"" "$THEME_DIR/template-parts/home/kids-capsule-reveal.php" || \
	! rg -q "data-procession-chapter=\"guardians\"" "$THEME_DIR/template-parts/home/kids-capsule-reveal.php" || \
	! rg -q "data-procession-chapter=\"heir\"" "$THEME_DIR/template-parts/home/kids-capsule-reveal.php"; then
	echo "FAIL Kids Capsule Royal Procession semantic chapter contract missing" >&2
	exit 1
fi

if ! rg -q "function skyyrose2_get_products_by_skus" "$THEME_DIR/functions.php" || \
	! rg -Fq "array( 'kids-001', 'kids-002' )" "$THEME_DIR/front-page.php" || \
	! rg -q "required_collection" "$THEME_DIR/functions.php"; then
	echo "FAIL Kids Capsule exact-SKU collection resolver missing" >&2
	exit 1
fi

if ! rg -q "function skyyrose2_presentation_registry" "$THEME_DIR/functions.php" || \
	! rg -q "woocommerce_cart_is_empty" "$THEME_DIR/woocommerce/cart/cart.php" || \
	! rg -Fq "function_exists( 'wc_get_page_permalink' )" "$THEME_DIR/404.php" || \
	rg -Fq "if ( empty( \$products ) && 'pre-order' === \$collection )" "$THEME_DIR/functions.php"; then
	echo "FAIL V2 truth and WooCommerce compatibility contract missing" >&2
	exit 1
fi

if ! rg -q '"state": "FROZEN"' "$REPO_ROOT/.fashion-theme/candidate-manifest.json" || \
	rg -q '"state": "BLOCKED"|"state": "UNFROZEN"|"status": "UNKNOWN"' "$REPO_ROOT/.fashion-theme/catalog-binding.json" "$REPO_ROOT/.fashion-theme/shot-manifest.json"; then
	echo "FAIL candidate-bound evidence is blocked, unknown, or unfrozen" >&2
	exit 1
fi

reveal_line="$(rg -n "template-parts/home/kids-capsule-reveal" "$THEME_DIR/front-page.php" | cut -d: -f1)"
collections_line="$(rg -n 'section id="collections"' "$THEME_DIR/front-page.php" | cut -d: -f1)"
if [ -z "$reveal_line" ] || [ -z "$collections_line" ] || [ "$reveal_line" -le "$collections_line" ]; then
	echo "FAIL Kids Capsule reveal must exist exactly once after the three-world opening" >&2
	exit 1
fi

if [ "$(rg -c "template-parts/home/kids-capsule-reveal" "$THEME_DIR/front-page.php")" -ne 1 ]; then
	echo "FAIL Kids Capsule reveal placement is duplicated" >&2
	exit 1
fi

if ! rg -q "Kids Capsule Royal Procession" "$THEME_DIR/assets/js/kids-capsule-reveal.js" || \
	rg -q "preventDefault|wheel|touchmove" "$THEME_DIR/assets/js/kids-capsule-reveal.js" || \
	! rg -q "prefers-reduced-motion: reduce" "$THEME_DIR/assets/css/theme.css" || \
	! rg -q "sr-kids-procession" "$THEME_DIR/assets/css/theme.css"; then
	echo "FAIL Kids Capsule progressive enhancement or fallback contract missing" >&2
	exit 1
fi

if rg -q '\$[0-9]|[0-9]+[.]?[0-9]* USD|hard.?coded (price|stock)' "$THEME_DIR/template-parts/home/kids-capsule-reveal.php"; then
	echo "FAIL hard-coded commerce truth detected in Kids Capsule reveal" >&2
	exit 1
fi

# Anti-generic floor: collection storytelling must be implemented as distinct
# compositions and card systems, never merely a shared template with recolored copy.
for collection in signature black-rose love-hurts kids-capsule; do
	if ! rg -Fq "[data-collection=\"$collection\"] .sr2-collection-identity" "$THEME_DIR/assets/css/theme.css"; then
		echo "FAIL generic collection hero: missing $collection composition" >&2
		exit 1
	fi
	if ! rg -Fq ".sr2-c-product-portal[data-presentation=\"$collection\"]" "$THEME_DIR/assets/css/theme.css"; then
		echo "FAIL generic product card: missing $collection card direction" >&2
		exit 1
	fi
done

find "$THEME_DIR" -name '*.php' -print0 | xargs -0 -n1 php -l >/dev/null
node --check "$THEME_DIR/assets/js/theme.js"
node --check "$THEME_DIR/assets/js/kids-capsule-reveal.js"
jq empty "$REPO_ROOT/.fashion-theme/v2-remodel-ledger.json" "$REPO_ROOT/.fashion-theme/catalog-binding.json" "$REPO_ROOT/.fashion-theme/shot-manifest.json"
git -C "$REPO_ROOT" diff --check -- "$THEME_DIR" "$REPO_ROOT/.fashion-theme" "$REPO_ROOT/.wolf"

echo "PASS V2 candidate source gate"
