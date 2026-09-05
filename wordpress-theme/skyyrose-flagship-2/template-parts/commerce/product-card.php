<?php
/**
 * Collection editorial product card.
 *
 * Expected arguments:
 * - product: WC_Product.
 * - index: zero-based loop index used for image priority.
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

$approved_front = skyyrose2_approved_card_front( $card_product );
$is_top = 'kids-capsule' !== $collection_slug && in_array( $presentation_record['garment_type'] ?? '', array( 'shirt', 'crewneck', 'hoodie', 'jersey', 'jacket', 'bomber jacket' ), true );
$card_crop = $approved_front && $is_top ? 'top' : 'full';
$frame_collection = $collections[ $collection_slug ] ?? array();
$frame_asset = $frame_collection['portal_statue'] ?? array();
$frame_uri = ! empty( $frame_asset['small'] ) ? skyyrose2_sot_asset_uri( $frame_asset['small'] ) : '';
$media_fallback = ! $image_id ? skyyrose2_product_media_fallback( $card_product ) : array();
$price_html     = $card_product->get_price_html();
$stock_html     = function_exists( 'wc_get_stock_html' ) ? wc_get_stock_html( $card_product ) : '';
$stock_state    = $card_product->is_in_stock() ? 'available' : 'unavailable';
$stock_label    = $card_product->is_in_stock() ? __( 'Available', 'skyyrose-flagship-2' ) : __( 'Unavailable', 'skyyrose-flagship-2' );
$quick_view_image = $image_id && function_exists( 'wp_get_attachment_image_url' ) ? wp_get_attachment_image_url( $image_id, 'woocommerce_single' ) : ( $media_fallback['src'] ?? '' );
if ( $approved_front ) {
	$quick_view_image = $approved_front['src'];
}
$quick_view_excerpt = wp_trim_words( wp_strip_all_tags( $card_product->get_short_description() ), 26, '…' );
$loading        = $card_index < 4 ? 'eager' : 'lazy';
$fetchpriority  = 0 === $card_index ? 'high' : 'auto';
$product_type   = method_exists( $card_product, 'get_type' ) ? $card_product->get_type() : 'unknown';
$purchasable    = method_exists( $card_product, 'is_purchasable' ) ? ( $card_product->is_purchasable() ? 'true' : 'false' ) : 'unknown';
$image_attrs    = array(
	'class'         => 'sr2-c-editorial-card__product-image',
	'loading'       => $loading,
	'fetchpriority' => $fetchpriority,
	'decoding'      => 'async',
);
?>
<article
	class="sr2-c-editorial-card"
	data-card-direction="collection-editorial"
	data-card-crop="<?php echo esc_attr( $card_crop ); ?>"
	data-card-frame="<?php echo esc_attr( $frame_uri ? 'v2-statue' : 'plain' ); ?>"
	data-presentation="<?php echo esc_attr( $presentation ); ?>"
	data-collection="<?php echo esc_attr( $collection_slug ?: $presentation ); ?>"
	data-product-type="<?php echo esc_attr( $product_type ); ?>"
	data-purchasable="<?php echo esc_attr( $purchasable ); ?>"
	data-availability="<?php echo esc_attr( $stock_state ); ?>"
	data-media-state="<?php echo esc_attr( ( $approved_front || $image_id || $media_fallback ) ? 'ready' : 'missing' ); ?>"
>

	<a
		class="sr2-c-editorial-card__media"
		href="<?php echo esc_url( $product_url ); ?>"
		aria-label="<?php echo esc_attr( sprintf( __( 'View %s', 'skyyrose-flagship-2' ), $product_name ) ); ?>"
	>
		<?php if ( $frame_uri ) : ?>
			<img class="sr2-c-editorial-card__frame" src="<?php echo esc_url( $frame_uri ); ?>" alt="" aria-hidden="true" width="<?php echo esc_attr( (string) ( $frame_asset['width'] ?? 970 ) ); ?>" height="<?php echo esc_attr( (string) ( $frame_asset['height'] ?? 1620 ) ); ?>" loading="<?php echo esc_attr( $loading ); ?>" decoding="async">
			<span class="sr2-c-editorial-card__frame-label" aria-hidden="true"><span class="sr2-c-editorial-card__inscription"><?php echo esc_html( $presentation_name ); ?></span></span>
		<?php endif; ?>
		<span class="sr2-c-editorial-card__photo-window">
		<?php if ( $approved_front ) : ?>
			<img class="sr2-c-editorial-card__product-image" src="<?php echo esc_url( $approved_front['src'] ); ?>" alt="<?php echo esc_attr( $approved_front['alt'] ); ?>" width="<?php echo esc_attr( (string) $approved_front['width'] ); ?>" height="<?php echo esc_attr( (string) $approved_front['height'] ); ?>" loading="<?php echo esc_attr( $loading ); ?>" fetchpriority="<?php echo esc_attr( $fetchpriority ); ?>" decoding="async">
		<?php elseif ( $image_id ) : ?>
			<?php echo wp_kses_post( wp_get_attachment_image( $image_id, 'woocommerce_thumbnail', false, $image_attrs ) ); ?>
		<?php elseif ( $media_fallback ) : ?>
			<img class="sr2-c-editorial-card__product-image" src="<?php echo esc_url( $media_fallback['src'] ); ?>" alt="<?php echo esc_attr( $media_fallback['alt'] ); ?>" width="<?php echo esc_attr( (string) $media_fallback['width'] ); ?>" height="<?php echo esc_attr( (string) $media_fallback['height'] ); ?>" loading="<?php echo esc_attr( $loading ); ?>" fetchpriority="<?php echo esc_attr( $fetchpriority ); ?>" decoding="async">
		<?php elseif ( function_exists( 'wc_placeholder_img' ) ) : ?>
			<?php echo wp_kses_post( wc_placeholder_img( 'woocommerce_thumbnail', $image_attrs ) ); ?>
		<?php endif; ?>
		</span>

	</a>

	<div class="sr2-c-editorial-card__body">
		<p class="sr2-c-editorial-card__collection"><?php echo esc_html( $presentation_name ); ?></p>
		<h2 class="woocommerce-loop-product__title sr2-c-editorial-card__title">
			<a href="<?php echo esc_url( $product_url ); ?>"><?php echo esc_html( $product_name ); ?></a>
		</h2>

		<div class="sr2-c-editorial-card__commerce">
			<?php if ( $price_html ) : ?>
				<p class="price sr2-c-editorial-card__price"><span class="screen-reader-text"><?php esc_html_e( 'Price:', 'skyyrose-flagship-2' ); ?></span><?php echo wp_kses_post( $price_html ); ?></p>
			<?php endif; ?>
			<div class="sr2-c-editorial-card__availability" data-state="<?php echo esc_attr( $stock_state ); ?>">
				<span class="sr2-c-editorial-card__availability-mark" aria-hidden="true"></span>
				<?php if ( $stock_html ) : ?>
					<?php echo wp_kses_post( $stock_html ); ?>
				<?php else : ?>
					<span><?php echo esc_html( $stock_label ); ?></span>
				<?php endif; ?>
			</div>
		</div>

		<div class="sr2-c-editorial-card__actions">
			<?php if ( function_exists( 'woocommerce_template_loop_add_to_cart' ) ) : ?>
				<?php woocommerce_template_loop_add_to_cart(); ?>
			<?php endif; ?>
			<button
				type="button"
				class="sr2-c-editorial-card__quick-view"
				data-quick-view
				data-quick-view-name="<?php echo esc_attr( $product_name ); ?>"
				data-quick-view-collection="<?php echo esc_attr( $presentation_name ); ?>"
				data-quick-view-price="<?php echo esc_attr( wp_strip_all_tags( $price_html ) ); ?>"
				data-quick-view-availability="<?php echo esc_attr( $stock_label ); ?>"
				data-quick-view-excerpt="<?php echo esc_attr( $quick_view_excerpt ); ?>"
				data-quick-view-image="<?php echo esc_url( $quick_view_image ); ?>"
				data-quick-view-url="<?php echo esc_url( $product_url ); ?>"
			>
				<?php esc_html_e( 'Quick view', 'skyyrose-flagship-2' ); ?>
			</button>
			<a class="sr2-c-editorial-card__details" href="<?php echo esc_url( $product_url ); ?>">
				<?php esc_html_e( 'View piece', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span>
			</a>
		</div>
	</div>
</article>
<?php
$product = $previous_global_product;
