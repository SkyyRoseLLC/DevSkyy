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

// wc_product_class() advances the native loop counter below; read it first.
$loop_index = max( 0, (int) wc_get_loop_prop( 'loop' ) );
?>
<li <?php wc_product_class( 'sr2-c-product-card-wrap sr2-c-product-card-wrap--portal', $product ); ?>>
	<?php
	get_template_part(
		'template-parts/commerce/product-card',
		null,
		array(
			'product' => $product,
			'index'   => $loop_index,
			'sizes'   => ( is_shop() || is_product_taxonomy() ) && ! wc_get_loop_prop( 'name' ) ? skyyrose2_shop_card_sizes() : '',
			'media_priority' => ( is_shop() || is_product_taxonomy() ) && ! wc_get_loop_prop( 'name' ) && $loop_index < 2 ? ( 0 === $loop_index ? 'high' : 'eager' ) : 'lazy',
		)
	);
	?>
</li>
