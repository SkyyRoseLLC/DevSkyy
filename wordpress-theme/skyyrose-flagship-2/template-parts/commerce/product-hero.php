<?php
/**
 * House of Roses product purchase spread.
 *
 * This keeps the canonical WooCommerce hook sequence intact so extensions,
 * product types, variation forms, stock state, and Product structured data
 * continue to use server-authoritative behavior.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

$hero_product           = isset( $args['product'] ) && is_a( $args['product'], 'WC_Product' ) ? $args['product'] : null;
$hero_presentation      = isset( $args['presentation'] ) ? sanitize_html_class( $args['presentation'] ) : 'house';
$hero_presentation_name = isset( $args['presentation_name'] ) ? (string) $args['presentation_name'] : __( 'SkyyRose', 'skyyrose-flagship-2' );
$hero_collection_data   = isset( $args['collection_data'] ) && is_array( $args['collection_data'] ) ? $args['collection_data'] : null;

if ( ! $hero_product ) {
	return;
}

do_action( 'woocommerce_before_single_product' );

if ( post_password_required() ) {
	echo get_the_password_form(); // phpcs:ignore WordPress.Security.EscapeOutput.OutputNotEscaped -- Core-generated password form.
	return;
}

$verified_media      = function_exists( 'skyyrose2_product_verified_card_media' ) ? skyyrose2_product_verified_card_media( $hero_product ) : array();
$verified_media_ids  = array_values( array_filter( array_map( static function ( $media ) {
	return isset( $media['id'] ) ? absint( $media['id'] ) : 0;
}, $verified_media ) ) );
$gallery_count       = count( $verified_media_ids );
$stock_html          = wc_get_stock_html( $hero_product );
$stock_state         = $hero_product->is_in_stock() ? 'available' : 'unavailable';
$portal_kicker       = $hero_collection_data && ! empty( $hero_collection_data['kicker'] ) ? $hero_collection_data['kicker'] : '';
$portal_story        = $hero_collection_data && ! empty( $hero_collection_data['card_story'] ) ? $hero_collection_data['card_story'] : '';
$portal_artifact_uri = $hero_collection_data && ! empty( $hero_collection_data['artifact'] ) ? skyyrose2_sot_asset_uri( $hero_collection_data['artifact'] ) : '';
$product_type        = $hero_product->get_type();
$purchasable         = $hero_product->is_purchasable() ? 'true' : 'false';
?>
<div
	id="product-<?php the_ID(); ?>"
	<?php wc_product_class( 'sr2-pdp-product sr2-pdp-product--portal', $hero_product ); ?>
	data-presentation="<?php echo esc_attr( $hero_presentation ); ?>"
	data-product-type="<?php echo esc_attr( $product_type ); ?>"
	data-purchasable="<?php echo esc_attr( $purchasable ); ?>"
	data-gallery-count="<?php echo esc_attr( (string) $gallery_count ); ?>"
	data-availability="<?php echo esc_attr( $stock_state ); ?>"
	data-verified-media="<?php echo ! empty( $verified_media_ids ) ? 'true' : 'false'; ?>"
>
	<div class="sr2-pdp-product__media" role="region" aria-label="<?php esc_attr_e( 'Published product views', 'skyyrose-flagship-2' ); ?>">
		<p class="sr2-pdp-product__media-label">
			<span><?php esc_html_e( 'Product archive', 'skyyrose-flagship-2' ); ?></span>
			<span><?php echo esc_html( sprintf( _n( '%d published view', '%d published views', $gallery_count, 'skyyrose-flagship-2' ), $gallery_count ) ); ?></span>
		</p>
		<?php
		/**
		 * Keep WooCommerce's gallery hook sequence while constraining it to
		 * manifest-reconciled product attachments for this product only.
		 */
		$previous_global_product = $GLOBALS['product'] ?? null;
		$GLOBALS['product']      = $hero_product;
		if ( ! empty( $verified_media_ids ) ) {
			$verified_image_filter = static function ( $image_id, $product ) use ( $hero_product, $verified_media_ids ) {
				return $product === $hero_product ? $verified_media_ids[0] : $image_id;
			};
			$verified_gallery_filter = static function ( $image_ids, $product ) use ( $hero_product, $verified_media_ids ) {
				return $product === $hero_product ? array_slice( $verified_media_ids, 1 ) : $image_ids;
			};
			add_filter( 'woocommerce_product_get_image_id', $verified_image_filter, 999, 2 );
			add_filter( 'woocommerce_product_get_gallery_image_ids', $verified_gallery_filter, 999, 2 );
			do_action( 'woocommerce_before_single_product_summary' );
			remove_filter( 'woocommerce_product_get_image_id', $verified_image_filter, 999 );
			remove_filter( 'woocommerce_product_get_gallery_image_ids', $verified_gallery_filter, 999 );
		} else {
			do_action( 'woocommerce_show_product_sale_flash' );
			?>
			<div class="sr2-pdp-product__media-unavailable" role="status">
				<strong><?php esc_html_e( 'Product views are being prepared.', 'skyyrose-flagship-2' ); ?></strong>
				<span><?php esc_html_e( 'This piece will return when its approved on-model proof is published.', 'skyyrose-flagship-2' ); ?></span>
			</div>
			<?php
		}
		$GLOBALS['product'] = $previous_global_product;
		?>
	</div>

	<div class="summary entry-summary sr2-pdp-product__summary">
		<header class="sr2-pdp-product__identity">
			<div>
				<p class="sr2-pdp-product__collection"><?php echo esc_html( $hero_presentation_name ); ?></p>
				<?php if ( $portal_kicker ) : ?><p class="sr2-pdp-product__kicker"><?php echo esc_html( $portal_kicker ); ?></p><?php endif; ?>
			</div>
			<?php if ( $portal_artifact_uri ) : ?>
				<img
					class="sr2-pdp-product__artifact"
					src="<?php echo esc_url( $portal_artifact_uri ); ?>"
					alt=""
					width="160"
					height="160"
					loading="lazy"
					decoding="async"
					aria-hidden="true"
				>
			<?php endif; ?>
		</header>

		<?php if ( $stock_html ) : ?>
			<div class="sr2-pdp-product__availability" data-state="<?php echo esc_attr( $stock_state ); ?>" aria-live="polite">
				<span class="sr2-pdp-product__availability-mark" aria-hidden="true"></span>
				<?php echo wp_kses_post( $stock_html ); ?>
			</div>
		<?php endif; ?>

		<?php if ( $portal_story ) : ?>
			<aside class="sr2-pdp-product__house-note">
				<span><?php esc_html_e( 'House note', 'skyyrose-flagship-2' ); ?></span>
				<p><?php echo esc_html( $portal_story ); ?></p>
			</aside>
		<?php endif; ?>

		<button class="sr2-pdp-product__fit-guide" type="button" data-size-guide-open aria-haspopup="dialog" aria-controls="sr2-size-guide-dialog">
			<?php esc_html_e( 'Fit + size guide', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span>
		</button>

		<?php
		/**
		 * Hook: woocommerce_single_product_summary.
		 *
		 * Preserves title, rating, price, excerpt, every product-type add-to-cart
		 * form, product meta, sharing, and WooCommerce Product structured data.
		 */
		do_action( 'woocommerce_single_product_summary' );
		?>
	</div>

	<div class="sr2-pdp-product__after-summary">
		<?php
		/**
		 * Hook: woocommerce_after_single_product_summary.
		 *
		 * Preserves tabs, upsells, and related products.
		 */
		do_action( 'woocommerce_after_single_product_summary' );
		?>
	</div>
</div>
<?php do_action( 'woocommerce_after_single_product' ); ?>
