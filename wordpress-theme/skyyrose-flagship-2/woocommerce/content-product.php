<?php
/**
 * House of Roses WooCommerce loop card.
 *
 * Product, price, availability, media, and purchase behavior remain owned by
 * WooCommerce. The theme contributes only the collection portal composition.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

global $product;

if ( ! is_a( $product, 'WC_Product' ) || ! $product->is_visible() ) {
	return;
}

if ( ! function_exists( 'skyyrose2_product_verified_card_media' ) || empty( skyyrose2_product_verified_card_media( $product ) ) ) {
	return;
}

$loop_index = max( 0, (int) wc_get_loop_prop( 'loop' ) - 1 );
?>
<li <?php wc_product_class( 'sr2-c-product-card-wrap sr2-c-product-card-wrap--portal', $product ); ?>>
	<?php
	get_template_part(
		'template-parts/commerce/product-card',
		null,
		array(
			'product' => $product,
			'index'   => $loop_index,
		)
	);
	?>
</li>
