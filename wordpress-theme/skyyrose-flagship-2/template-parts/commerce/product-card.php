<?php
/**
 * Collection-owned product portal card.
 *
 * Expected arguments:
 * - product: WC_Product.
 * - index: zero-based loop index used for image priority and chapter numbering.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

global $product;

$card_product = isset( $args['product'] ) && is_a( $args['product'], 'WC_Product' ) ? $args['product'] : $product;
$card_index   = isset( $args['index'] ) ? max( 0, (int) $args['index'] ) : 0;

if ( ! $card_product || ! $card_product->is_visible() ) {
	return;
}

// WooCommerce's loop add-to-cart renderer reads the global product. V2 cards
// are also rendered from editorial portals outside the native loop, so scope
// that global to this card and restore it after rendering to prevent a stale
// or neighboring product from receiving the action.
$previous_global_product = $product;
$product                 = $card_product;

$product_id       = $card_product->get_id();
$product_url      = method_exists( $card_product, 'get_permalink' ) ? $card_product->get_permalink() : get_permalink( $product_id );
$product_name     = $card_product->get_name();
$image_id         = $card_product->get_image_id();
$collections      = skyyrose2_collections();
$presentation     = 'house';
$presentation_name = __( 'SkyyRose', 'skyyrose-flagship-2' );
$collection_data  = null;
$presentation_record = skyyrose2_product_presentation( $card_product );
$collection_slug  = isset( $presentation_record['collection'] ) ? sanitize_title( $presentation_record['collection'] ) : '';
$is_jersey        = 'jersey-series' === ( $presentation_record['presentation'] ?? '' );

if ( $is_jersey ) {
	$presentation      = 'jersey-series';
	$presentation_name = __( 'Jersey Series', 'skyyrose-flagship-2' );
} else {
	if ( $collection_slug && isset( $collections[ $collection_slug ] ) ) {
		$presentation      = $collection_slug;
		$collection_data   = $collections[ $collection_slug ];
		$presentation_name = $collection_data['name'];
	}
}

$portal_order = array(
	'signature'     => 1,
	'black-rose'    => 2,
	'love-hurts'    => 3,
	'kids-capsule'  => 4,
	'jersey-series' => 5,
);
$chapter_number = isset( $portal_order[ $presentation ] ) ? $portal_order[ $presentation ] : $card_index + 1;
$artifact_uri   = $collection_data && ! empty( $collection_data['artifact'] ) ? skyyrose2_sot_asset_uri( $collection_data['artifact'] ) : '';
$scene_frames   = $collection_data ? skyyrose2_product_card_reel_frames( $collection_data ) : array();
$scene_uri      = ! empty( $scene_frames[0]['uri'] ) ? $scene_frames[0]['uri'] : '';
$price_html     = $card_product->get_price_html();
$stock_html     = function_exists( 'wc_get_stock_html' ) ? wc_get_stock_html( $card_product ) : '';
$stock_state    = $card_product->is_in_stock() ? 'available' : 'unavailable';
$stock_label    = $card_product->is_in_stock() ? __( 'Available', 'skyyrose-flagship-2' ) : __( 'Unavailable', 'skyyrose-flagship-2' );
$loading        = $card_index < 4 ? 'eager' : 'lazy';
$fetchpriority  = 0 === $card_index ? 'high' : 'auto';
$product_type   = method_exists( $card_product, 'get_type' ) ? $card_product->get_type() : 'unknown';
$purchasable    = method_exists( $card_product, 'is_purchasable' ) ? ( $card_product->is_purchasable() ? 'true' : 'false' ) : 'unknown';
$image_attrs    = array(
	'class'         => 'sr2-c-product-portal__product-image',
	'loading'       => $loading,
	'fetchpriority' => $fetchpriority,
	'decoding'      => 'async',
);
?>
<article
	class="sr2-c-product-portal"
	data-card-direction="house-portal"
	data-presentation="<?php echo esc_attr( $presentation ); ?>"
	data-collection="<?php echo esc_attr( $collection_slug ?: $presentation ); ?>"
	data-product-type="<?php echo esc_attr( $product_type ); ?>"
	data-purchasable="<?php echo esc_attr( $purchasable ); ?>"
	data-availability="<?php echo esc_attr( $stock_state ); ?>"
	data-media-state="<?php echo esc_attr( $image_id ? 'ready' : 'missing' ); ?>"
>
	<header class="sr2-c-product-portal__chapter" aria-hidden="true">
		<span><?php echo esc_html( sprintf( '%02d', $chapter_number ) ); ?></span>
		<span><?php echo esc_html( $presentation_name ); ?></span>
	</header>

	<a
		class="sr2-c-product-portal__media"
		href="<?php echo esc_url( $product_url ); ?>"
		aria-label="<?php echo esc_attr( sprintf( __( 'View %s', 'skyyrose-flagship-2' ), $product_name ) ); ?>"
	>
		<?php if ( $scene_uri ) : ?>
			<img class="sr2-c-product-portal__scene" src="<?php echo esc_url( $scene_uri ); ?>" alt="" width="1280" height="720" loading="lazy" decoding="async" aria-hidden="true">
		<?php endif; ?>
		<span class="sr2-c-product-portal__shade" aria-hidden="true"></span>
		<span class="sr2-c-product-portal__frame" aria-hidden="true"><i></i><b></b></span>
		<?php if ( $artifact_uri ) : ?>
			<img
				class="sr2-c-product-portal__artifact"
				src="<?php echo esc_url( $artifact_uri ); ?>"
				alt=""
				width="320"
				height="320"
				loading="lazy"
				decoding="async"
				aria-hidden="true"
			>
		<?php endif; ?>
		<?php if ( $image_id ) : ?>
			<?php echo wp_kses_post( wp_get_attachment_image( $image_id, 'woocommerce_thumbnail', false, $image_attrs ) ); ?>
		<?php elseif ( function_exists( 'wc_placeholder_img' ) ) : ?>
			<?php echo wp_kses_post( wc_placeholder_img( 'woocommerce_thumbnail', $image_attrs ) ); ?>
		<?php endif; ?>
	</a>

	<div class="sr2-c-product-portal__body">
		<p class="sr2-c-product-portal__collection"><?php echo esc_html( $presentation_name ); ?></p>
		<h2 class="woocommerce-loop-product__title sr2-c-product-portal__title">
			<a href="<?php echo esc_url( $product_url ); ?>"><?php echo esc_html( $product_name ); ?></a>
		</h2>

		<div class="sr2-c-product-portal__commerce">
			<?php if ( $price_html ) : ?>
				<p class="price sr2-c-product-portal__price"><span class="screen-reader-text"><?php esc_html_e( 'Price:', 'skyyrose-flagship-2' ); ?></span><?php echo wp_kses_post( $price_html ); ?></p>
			<?php endif; ?>
			<div class="sr2-c-product-portal__availability" data-state="<?php echo esc_attr( $stock_state ); ?>">
				<span class="sr2-c-product-portal__availability-mark" aria-hidden="true"></span>
				<?php if ( $stock_html ) : ?>
					<?php echo wp_kses_post( $stock_html ); ?>
				<?php else : ?>
					<span><?php echo esc_html( $stock_label ); ?></span>
				<?php endif; ?>
			</div>
		</div>

		<div class="sr2-c-product-portal__actions">
			<?php if ( function_exists( 'woocommerce_template_loop_add_to_cart' ) ) : ?>
				<?php woocommerce_template_loop_add_to_cart(); ?>
			<?php endif; ?>
			<a class="sr2-c-product-portal__details" href="<?php echo esc_url( $product_url ); ?>">
				<?php esc_html_e( 'View piece', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span>
			</a>
		</div>
	</div>
</article>
<?php
$product = $previous_global_product;
