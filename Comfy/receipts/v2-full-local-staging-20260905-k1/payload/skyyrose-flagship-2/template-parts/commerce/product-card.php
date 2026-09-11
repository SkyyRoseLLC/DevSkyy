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
$rose_uri       = skyyrose2_sot_asset_uri( 'images/logos/rose-gold-rose.webp' );
$portal_statue  = $collection_data && ! empty( $collection_data['portal_statue'] ) && is_array( $collection_data['portal_statue'] ) ? $collection_data['portal_statue'] : array();
$portal_statue_uri = ! empty( $portal_statue['src'] ) ? skyyrose2_sot_asset_uri( $portal_statue['src'] ) : '';
$portal_statue_small_uri = ! empty( $portal_statue['small'] ) ? skyyrose2_sot_asset_uri( $portal_statue['small'] ) : '';
$portal_statue_width = ! empty( $portal_statue['width'] ) ? (int) $portal_statue['width'] : 970;
$portal_statue_height = ! empty( $portal_statue['height'] ) ? (int) $portal_statue['height'] : 1620;
$has_portal_statue = (bool) ( $portal_statue_uri && $portal_statue_small_uri );
$scene_frames   = $collection_data ? skyyrose2_product_card_reel_frames( $collection_data ) : array();
$scene_uri      = ! empty( $scene_frames[0]['uri'] ) ? $scene_frames[0]['uri'] : '';
$verified_media  = skyyrose2_product_verified_card_media( $card_product );
$primary_media   = ! empty( $verified_media[0] ) ? $verified_media[0] : array();
$primary_image_id = ! empty( $primary_media['id'] ) ? (int) $primary_media['id'] : 0;
$has_verified_media = (bool) $primary_image_id;
$media_verification_label = __( 'Product imagery is being verified.', 'skyyrose-flagship-2' );
$media_link_label = sprintf( __( 'View %s', 'skyyrose-flagship-2' ), $product_name );
if ( ! $has_verified_media ) {
	$media_link_label .= '. ' . $media_verification_label;
}
$reel_media      = array_values(
	array_filter(
		$verified_media,
		static function ( $media ) {
			return in_array(
				(string) ( $media['role'] ?? '' ),
				array( 'on_model_front', 'on_model_back' ),
				true
			);
		}
	)
);
// The portal opens only into an on-model proof pair. Mannequin, packshot, and
// 3D evidence stays on the PDP; it never replaces the cinematic model card.
$reel_media      = array_slice( $reel_media, 0, 2 );
$reel_count      = count( $reel_media );
$price_html     = $card_product->get_price_html();
$stock_html     = function_exists( 'wc_get_stock_html' ) ? wc_get_stock_html( $card_product ) : '';
$stock_state    = $card_product->is_in_stock() ? 'available' : 'unavailable';
$stock_label    = $card_product->is_in_stock() ? __( 'Available', 'skyyrose-flagship-2' ) : __( 'Unavailable', 'skyyrose-flagship-2' );
$quick_view_image = $primary_image_id && function_exists( 'wp_get_attachment_image_url' ) ? wp_get_attachment_image_url( $primary_image_id, 'woocommerce_single' ) : '';
$quick_view_excerpt = wp_trim_words( wp_strip_all_tags( $card_product->get_short_description() ), 26, '…' );
$loading        = 0 === $card_index ? 'eager' : 'lazy';
$fetchpriority  = 0 === $card_index ? 'high' : 'auto';
$product_type   = method_exists( $card_product, 'get_type' ) ? $card_product->get_type() : 'unknown';
$purchasable    = method_exists( $card_product, 'is_purchasable' ) ? ( $card_product->is_purchasable() ? 'true' : 'false' ) : 'unknown';
$image_attrs    = array(
	'class'         => 'sr2-c-product-portal__product-image',
	'loading'       => $loading,
		'fetchpriority' => $fetchpriority,
		'decoding'      => 'async',
		'sizes'         => '(max-width: 640px) calc(100vw - 3rem), (max-width: 1024px) 46vw, 25vw',
);
?>
<article
	class="sr2-c-product-portal sr2-c-product-portal--ornate"
	data-card-direction="ornate-frame"
	data-portal-frame="<?php echo esc_attr( $has_portal_statue ? 'statue' : 'constructed' ); ?>"
	data-product-reel
	data-presentation="<?php echo esc_attr( $presentation ); ?>"
	data-collection="<?php echo esc_attr( $collection_slug ?: $presentation ); ?>"
	data-product-type="<?php echo esc_attr( $product_type ); ?>"
	data-purchasable="<?php echo esc_attr( $purchasable ); ?>"
	data-availability="<?php echo esc_attr( $stock_state ); ?>"
	data-media-state="<?php echo esc_attr( $has_verified_media ? 'verified' : 'unverified' ); ?>"
>
	<header class="sr2-c-product-portal__chapter" aria-hidden="true">
		<span><?php echo esc_html( sprintf( '%02d', $chapter_number ) ); ?></span>
		<span><?php echo esc_html( $presentation_name ); ?></span>
	</header>

	<a
		class="sr2-c-product-portal__media"
		href="<?php echo esc_url( $product_url ); ?>"
		aria-label="<?php echo esc_attr( $media_link_label ); ?>"
	>
		<?php if ( $has_verified_media && $scene_uri ) : ?>
			<img class="sr2-c-product-portal__scene" src="<?php echo esc_url( $scene_uri ); ?>" alt="" width="1280" height="720" loading="lazy" decoding="async" aria-hidden="true">
		<?php endif; ?>
		<span class="sr2-c-product-portal__shade" aria-hidden="true"></span>
		<?php if ( $has_portal_statue ) : ?>
			<img
				class="sr2-c-product-portal__statue"
				src="<?php echo esc_url( $portal_statue_small_uri ); ?>"
				srcset="<?php echo esc_attr( $portal_statue_small_uri . ' 640w, ' . $portal_statue_uri . ' ' . $portal_statue_width . 'w' ); ?>"
				sizes="(max-width: 640px) calc(100vw - 3rem), (max-width: 1024px) 46vw, 25vw"
				alt=""
				width="<?php echo esc_attr( (string) $portal_statue_width ); ?>"
				height="<?php echo esc_attr( (string) $portal_statue_height ); ?>"
				loading="<?php echo esc_attr( $loading ); ?>"
				decoding="async"
				aria-hidden="true"
			>
		<?php endif; ?>
		<span class="sr2-c-product-portal__architecture" aria-hidden="true">
			<span class="sr2-c-product-portal__pillar sr2-c-product-portal__pillar--left"><i></i><b></b></span>
			<span class="sr2-c-product-portal__pillar sr2-c-product-portal__pillar--right"><i></i><b></b></span>
			<span class="sr2-c-product-portal__stone-arch">
				<span class="sr2-c-product-portal__rose-socket sr2-c-product-portal__rose-socket--left"><img src="<?php echo esc_url( $rose_uri ); ?>" alt="" width="72" height="72" loading="lazy" decoding="async"></span>
				<span class="sr2-c-product-portal__rose-socket sr2-c-product-portal__rose-socket--keystone"><img src="<?php echo esc_url( $rose_uri ); ?>" alt="" width="96" height="96" loading="lazy" decoding="async"></span>
				<span class="sr2-c-product-portal__rose-socket sr2-c-product-portal__rose-socket--right"><img src="<?php echo esc_url( $rose_uri ); ?>" alt="" width="72" height="72" loading="lazy" decoding="async"></span>
			</span>
		</span>
		<span class="sr2-c-product-portal__frame" aria-hidden="true">
			<?php if ( $has_verified_media && $artifact_uri ) : ?>
				<img class="sr2-c-product-portal__frame-crest" src="<?php echo esc_url( $artifact_uri ); ?>" alt="" width="96" height="96" loading="lazy" decoding="async">
			<?php endif; ?>
			<i></i><b></b>
		</span>
		<?php if ( $has_verified_media && $artifact_uri ) : ?>
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
		<?php if ( $has_verified_media ) : ?>
			<?php echo wp_kses_post( wp_get_attachment_image( $primary_image_id, 'woocommerce_thumbnail', false, $image_attrs ) ); ?>
		<?php else : ?>
			<span class="sr2-c-product-portal__media-missing" role="status">
				<?php echo esc_html( $media_verification_label ); ?>
			</span>
		<?php endif; ?>
		<?php if ( $reel_count > 1 ) : ?>
			<div class="sr2-c-product-portal__reel" data-reel-count="<?php echo esc_attr( (string) $reel_count ); ?>" aria-label="<?php echo esc_attr( sprintf( __( 'More views of %s', 'skyyrose-flagship-2' ), $product_name ) ); ?>">
				<div class="sr2-c-product-portal__reel-track">
					<?php foreach ( $reel_media as $view_index => $view_media ) : ?>
						<?php
						$view_image_id = (int) $view_media['id'];
						$view_role     = (string) $view_media['role'];
						$view_label    = array(
							'on_model_front' => __( 'On model / front', 'skyyrose-flagship-2' ),
							'on_model_back'  => __( 'On model / back', 'skyyrose-flagship-2' ),
							'mannequin_front' => __( 'Mannequin / front', 'skyyrose-flagship-2' ),
							'mannequin_back' => __( 'Mannequin / back', 'skyyrose-flagship-2' ),
							'render_3d'       => __( '3D render', 'skyyrose-flagship-2' ),
							'packshot'        => __( 'Product detail', 'skyyrose-flagship-2' ),
						);
						?>
						<figure class="sr2-c-product-portal__reel-frame" data-view-role="<?php echo esc_attr( $view_role ); ?>">
							<?php echo wp_kses_post( wp_get_attachment_image( $view_image_id, 'woocommerce_thumbnail', false, array( 'class' => 'sr2-c-product-portal__reel-image', 'loading' => 'lazy', 'decoding' => 'async', 'sizes' => '(max-width: 640px) 30vw, 12vw', 'alt' => sprintf( __( '%s view %d', 'skyyrose-flagship-2' ), $product_name, $view_index + 1 ) ) ) ); ?>
							<figcaption><?php echo esc_html( $view_label[ $view_role ] ?? sprintf( __( 'View %02d', 'skyyrose-flagship-2' ), $view_index + 1 ) ); ?></figcaption>
						</figure>
					<?php endforeach; ?>
				</div>
				<span class="sr2-c-product-portal__reel-prompt" aria-hidden="true"><?php esc_html_e( 'Hover to explore views', 'skyyrose-flagship-2' ); ?></span>
			</div>
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
			<button
				type="button"
				class="sr2-c-product-portal__quick-view"
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
			<a class="sr2-c-product-portal__details" href="<?php echo esc_url( $product_url ); ?>">
				<?php esc_html_e( 'View piece', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span>
			</a>
		</div>
	</div>
</article>
<?php
$product = $previous_global_product;
