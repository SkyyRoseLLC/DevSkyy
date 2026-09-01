<?php
/**
 * Static regression checks for the V2 visual-commerce contract.
 *
 * This complements the live browser/staging suite. It fails when a product
 * card can render arbitrary WooCommerce media or when motion loses its
 * reduced-motion fallback.
 */

declare( strict_types=1 );

$theme = dirname( __DIR__ );

$fail = static function ( string $message ): void {
	fwrite( STDERR, "FAIL visual contract: {$message}\n" );
	exit( 1 );
};

$read = static function ( string $relative ) use ( $theme, $fail ): string {
	$path = $theme . '/' . ltrim( $relative, '/' );
	if ( ! is_readable( $path ) ) {
		$fail( "missing source file {$relative}" );
	}
	return (string) file_get_contents( $path );
};

$card = $read( 'template-parts/commerce/product-card.php' );
if ( substr_count( $card, 'skyyrose2_product_verified_card_media' ) < 1 || strpos( $card, 'data-verified-media="true"' ) === false ) {
	$fail( 'product cards are not bound to verified media' );
}
if ( strpos( $card, 'wc_placeholder_img' ) !== false || strpos( $card, '$card_product->get_image_id()' ) !== false ) {
	$fail( 'product cards retain an arbitrary or placeholder image fallback' );
}
foreach ( array( 'data-media-count', '--sr2-media-count', 'data-primary-media-role', '$verified_media[0]' ) as $token ) {
	if ( strpos( $card, $token ) === false ) {
		$fail( "product card missing aligned media token {$token}" );
	}
}

$content_product = $read( 'woocommerce/content-product.php' );
if ( strpos( $content_product, 'skyyrose2_product_verified_card_media' ) === false ) {
	$fail( 'native WooCommerce loop does not fail closed for unverified media' );
}

foreach ( array( 'front-page.php', 'template-parts/immersive/world.php', 'template-parts/home/kids-capsule-reveal.php' ) as $surface ) {
	$source = $read( $surface );
	if ( strpos( $source, 'skyyrose2_product_verified_card_media' ) === false ) {
		$fail( "product proof surface missing verified media reconciliation: {$surface}" );
	}
}

$wishlist = $read( 'page-wishlist.php' );
foreach ( array( 'data-wishlist-page', 'skyyrose2_get_products', 'data-wishlist-empty' ) as $token ) {
	if ( strpos( $wishlist, $token ) === false ) {
		$fail( "fresh-install wishlist contract missing {$token}" );
	}
}
$theme_js = $read( 'assets/js/theme.js' );
foreach ( array( 'skyyrose2-wishlist-v1', 'data-wishlist-toggle', 'localStorage' ) as $token ) {
	if ( strpos( $theme_js, $token ) === false ) {
		$fail( "theme-owned wishlist runtime missing {$token}" );
	}
}

$hero = $read( 'template-parts/commerce/product-hero.php' );
foreach ( array( 'woocommerce_product_get_image_id', 'woocommerce_product_get_gallery_image_ids', 'data-verified-media' ) as $token ) {
	if ( strpos( $hero, $token ) === false ) {
		$fail( "PDP media reconciliation missing {$token}" );
	}
}

$registry = json_decode( $read( 'data/product-presentation-registry.json' ), true );
if ( ! is_array( $registry ) || empty( $registry['scene_products'] ) ) {
	$fail( 'scene-to-SKU assignments are missing from the generated registry' );
}

$theme_css = $read( 'assets/css/theme.css' );
foreach ( array( 'signature', 'black-rose', 'love-hurts', 'kids-capsule', 'jersey-series' ) as $presentation ) {
	if ( strpos( $theme_css, '[data-presentation="' . $presentation . '"]' ) === false ) {
		$fail( "collection-specific product frame missing {$presentation}" );
	}
}
if ( strpos( $theme_css, '@media (prefers-reduced-motion: reduce)' ) === false || strpos( $theme_css, '.sr2-collection-hero__effects { display: none; }' ) === false ) {
	$fail( 'animated heroes do not have a reduced-motion fallback' );
}
foreach ( array( 'sr2-signature-monument-drift', 'sr2-black-rose-monument-drift', 'sr2-love-hurts-monument-drift', 'sr2-kids-monument-drift' ) as $animation ) {
	if ( strpos( $theme_css, $animation ) === false ) {
		$fail( "collection-specific hero animation missing {$animation}" );
	}
}

fwrite( STDOUT, "PASS visual contract: verified cards, reconciled PDP media, assigned scenes, and collection frames\n" );
