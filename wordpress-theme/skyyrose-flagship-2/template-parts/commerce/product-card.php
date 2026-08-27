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

$verified_media = function_exists( 'skyyrose2_product_verified_card_media' ) ? skyyrose2_product_verified_card_media( $card_product ) : array();
if ( empty( $verified_media ) || empty( $verified_media[0]['id'] ) ) {
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
$image_id         = absint( $verified_media[0]['id'] );
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

$artifact_uri   = $collection_data && ! empty( $collection_data['artifact'] ) ? skyyrose2_sot_asset_uri( $collection_data['artifact'] ) : '';
$rose_uri       = skyyrose2_sot_asset_uri( 'images/logos/rose-gold-rose.webp' );
$brand_mark_uri = skyyrose2_sot_asset_uri( 'brand/skyyrose-logo-still-384w.webp' );
$portal_statue  = $collection_data && ! empty( $collection_data['portal_statue'] ) && is_array( $collection_data['portal_statue'] ) ? $collection_data['portal_statue'] : array();
$portal_statue_uri = ! empty( $portal_statue['src'] ) ? skyyrose2_sot_asset_uri( $portal_statue['src'] ) : '';
$portal_statue_small_uri = ! empty( $portal_statue['small'] ) ? skyyrose2_sot_asset_uri( $portal_statue['small'] ) : '';
$portal_statue_width = ! empty( $portal_statue['width'] ) ? (int) $portal_statue['width'] : 970;
$portal_statue_height = ! empty( $portal_statue['height'] ) ? (int) $portal_statue['height'] : 1620;
$has_portal_statue = (bool) ( $portal_statue_uri && $portal_statue_small_uri );
$scene_frames   = $collection_data ? skyyrose2_product_card_reel_frames( $collection_data ) : array();
$scene_uri      = ! empty( $scene_frames[0]['uri'] ) ? $scene_frames[0]['uri'] : '';
$view_image_ids = array_values( array_filter( array_map( static function ( $media ) {
	return isset( $media['id'] ) ? absint( $media['id'] ) : 0;
}, $verified_media ) ) );
$price_html     = $card_product->get_price_html();
$stock_html     = function_exists( 'wc_get_stock_html' ) ? wc_get_stock_html( $card_product ) : '';
$stock_state    = $card_product->is_in_stock() ? 'available' : 'unavailable';
$stock_label    = $card_product->is_in_stock() ? __( 'Available', 'skyyrose-flagship-2' ) : __( 'Unavailable', 'skyyrose-flagship-2' );
$loading        = $card_index < 4 ? 'eager' : 'lazy';
$fetchpriority  = 0 === $card_index ? 'high' : 'auto';
$product_type   = method_exists( $card_product, 'get_type' ) ? $card_product->get_type() : 'unknown';
$purchasable    = method_exists( $card_product, 'is_purchasable' ) ? ( $card_product->is_purchasable() ? 'true' : 'false' ) : 'unknown';
$can_quick_add  = 'simple' === $product_type && 'true' === $purchasable && 'available' === $stock_state;
$purchase_mode  = $can_quick_add ? 'quick-add' : ( 'variable' === $product_type ? 'choose-options' : 'view-piece' );
$purchase_label = $can_quick_add ? __( 'Add to bag', 'skyyrose-flagship-2' ) : ( 'choose-options' === $purchase_mode ? __( 'Choose size', 'skyyrose-flagship-2' ) : __( 'View piece', 'skyyrose-flagship-2' ) );
$purchase_note  = $can_quick_add ? __( 'Ready to add to bag.', 'skyyrose-flagship-2' ) : ( 'choose-options' === $purchase_mode ? __( 'Choose options on the product page.', 'skyyrose-flagship-2' ) : __( 'View current product details and availability.', 'skyyrose-flagship-2' ) );
$product_story  = wp_trim_words( wp_strip_all_tags( (string) $card_product->get_short_description() ), 28, '…' );
$variant_options = array();

// Card variants are a compact, read-only disclosure of currently purchasable,
// in-stock WooCommerce variation data. Selection remains on the PDP, where
// WooCommerce owns the exact combination, price, stock, and validation state.
if ( 'variable' === $product_type && method_exists( $card_product, 'get_available_variations' ) && function_exists( 'wc_attribute_label' ) ) {
	foreach ( (array) $card_product->get_available_variations( 'array' ) as $variation ) {
		if ( ! is_array( $variation ) || empty( $variation['is_purchasable'] ) || empty( $variation['is_in_stock'] ) || empty( $variation['attributes'] ) || ! is_array( $variation['attributes'] ) ) {
			continue;
		}

		foreach ( $variation['attributes'] as $attribute_name => $attribute_value ) {
			$attribute_key = str_replace( 'attribute_', '', (string) $attribute_name );
			$attribute_value = wc_clean( (string) $attribute_value );
			if ( '' === $attribute_key || '' === $attribute_value ) {
				continue;
			}

			if ( taxonomy_exists( $attribute_key ) ) {
				$term = get_term_by( 'slug', $attribute_value, $attribute_key );
				if ( $term && ! is_wp_error( $term ) ) {
					$attribute_value = $term->name;
				}
			}

			$attribute_label = wc_attribute_label( $attribute_key );
			if ( ! isset( $variant_options[ $attribute_label ] ) ) {
				$variant_options[ $attribute_label ] = array();
			}
			$variant_options[ $attribute_label ][ sanitize_title( $attribute_value ) ] = $attribute_value;
		}
	}
}

$variant_labels = array();
foreach ( $variant_options as $attribute_label => $values ) {
	$values = array_values( $values );
	if ( empty( $values ) ) {
		continue;
	}
	$visible_values = array_slice( $values, 0, 4 );
	$hidden_count   = count( $values ) - count( $visible_values );
	if ( $hidden_count > 0 ) {
		$visible_values[] = sprintf(
			/* translators: %d: number of additional currently available values. */
			__( '+%d more', 'skyyrose-flagship-2' ),
			$hidden_count
		);
	}
	$variant_labels[] = sprintf(
		/* translators: 1: WooCommerce attribute label, 2: currently available attribute values. */
		__( '%1$s: %2$s', 'skyyrose-flagship-2' ),
		$attribute_label,
		implode( ' / ', $visible_values )
	);
}

$has_hover_insight = (bool) ( $product_story || $variant_labels );
$render_card_insight = static function ( $story, $labels ) {
	if ( $story ) :
		?>
		<p class="sr2-c-product-portal__hover-story"><?php echo esc_html( $story ); ?></p>
		<?php
	endif;
	if ( $labels ) :
		?>
		<div class="sr2-c-product-portal__hover-variants">
			<span><?php esc_html_e( 'Available variants', 'skyyrose-flagship-2' ); ?></span>
			<ul>
				<?php foreach ( $labels as $label ) : ?>
					<li><?php echo esc_html( $label ); ?></li>
				<?php endforeach; ?>
			</ul>
		</div>
		<?php
	endif;
};
$image_attrs    = array(
	'class'         => 'sr2-c-product-portal__product-image',
	'loading'       => $loading,
	'fetchpriority' => $fetchpriority,
	'decoding'      => 'async',
);
?>
<article
	class="sr2-c-product-portal sr2-c-product-portal--ornate"
	data-card-direction="ornate-frame"
	data-portal-frame="<?php echo esc_attr( $has_portal_statue ? 'statue' : 'constructed' ); ?>"
	data-product-reel
	data-presentation="<?php echo esc_attr( $presentation ); ?>"
	data-collection="<?php echo esc_attr( $collection_slug ?: $presentation ); ?>"
	data-product-id="<?php echo esc_attr( (string) $product_id ); ?>"
	data-product-type="<?php echo esc_attr( $product_type ); ?>"
	data-purchase-mode="<?php echo esc_attr( $purchase_mode ); ?>"
	data-purchasable="<?php echo esc_attr( $purchasable ); ?>"
	data-availability="<?php echo esc_attr( $stock_state ); ?>"
	data-media-state="ready"
	data-verified-media="true"
	data-primary-media-role="<?php echo esc_attr( $verified_media[0]['role'] ?? '' ); ?>"
	data-media-count="<?php echo esc_attr( (string) count( $view_image_ids ) ); ?>"
	data-hover-insight="<?php echo esc_attr( $has_hover_insight ? 'true' : 'false' ); ?>"
	style="--sr2-media-count:<?php echo esc_attr( (string) max( 1, count( $view_image_ids ) ) ); ?>;"
>
	<header class="sr2-c-product-portal__chapter" aria-hidden="true">
		<img class="sr2-c-product-portal__chapter-mark" src="<?php echo esc_url( $brand_mark_uri ); ?>" alt="" width="384" height="216" loading="lazy" decoding="async">
		<span class="sr2-c-product-portal__chapter-collection"><?php echo esc_html( $presentation_name ); ?></span>
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
			<?php if ( $artifact_uri ) : ?>
				<img class="sr2-c-product-portal__frame-crest" src="<?php echo esc_url( $artifact_uri ); ?>" alt="" width="96" height="96" loading="lazy" decoding="async">
			<?php endif; ?>
			<i></i><b></b>
		</span>
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
		<?php echo wp_kses_post( wp_get_attachment_image( $image_id, 'woocommerce_thumbnail', false, array_merge( $image_attrs, array( 'alt' => $product_name ) ) ) ); ?>
		<?php if ( count( $view_image_ids ) > 1 ) : ?>
			<div class="sr2-c-product-portal__reel" data-reel-count="<?php echo esc_attr( (string) count( $view_image_ids ) ); ?>" aria-label="<?php echo esc_attr( sprintf( __( 'More views of %s', 'skyyrose-flagship-2' ), $product_name ) ); ?>">
				<div class="sr2-c-product-portal__reel-track">
					<?php foreach ( $view_image_ids as $view_index => $view_image_id ) : ?>
						<figure class="sr2-c-product-portal__reel-frame">
							<?php echo wp_kses_post( wp_get_attachment_image( $view_image_id, 'woocommerce_thumbnail', false, array( 'class' => 'sr2-c-product-portal__reel-image', 'loading' => 'lazy', 'decoding' => 'async', 'alt' => sprintf( __( '%s view %d', 'skyyrose-flagship-2' ), $product_name, $view_index + 1 ) ) ) ); ?>
							<figcaption><?php echo esc_html( 0 === $view_index ? __( 'On model / front', 'skyyrose-flagship-2' ) : ( isset( $verified_media[ $view_index ]['role'] ) ? ucwords( str_replace( '_', ' / ', (string) $verified_media[ $view_index ]['role'] ) ) : sprintf( __( 'View %02d', 'skyyrose-flagship-2' ), $view_index + 1 ) ) ); ?></figcaption>
						</figure>
					<?php endforeach; ?>
				</div>
				<span class="sr2-c-product-portal__reel-prompt" aria-hidden="true"><?php esc_html_e( 'Hover to explore views', 'skyyrose-flagship-2' ); ?></span>
			</div>
		<?php endif; ?>
		<?php if ( $has_hover_insight ) : ?>
			<div class="sr2-c-product-portal__hover-insight" data-portal-hover-insight>
				<?php $render_card_insight( $product_story, $variant_labels ); ?>
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
		<?php if ( $has_hover_insight ) : ?>
			<details class="sr2-c-product-portal__touch-insight" data-portal-touch-insight>
				<summary><?php esc_html_e( 'Piece details', 'skyyrose-flagship-2' ); ?></summary>
				<div class="sr2-c-product-portal__touch-insight-content">
					<?php $render_card_insight( $product_story, $variant_labels ); ?>
				</div>
			</details>
		<?php endif; ?>

		<div class="sr2-c-product-portal__actions">
			<?php if ( $can_quick_add ) : ?>
				<a
					href="<?php echo esc_url( $card_product->add_to_cart_url() ); ?>"
					class="button product_type_simple add_to_cart_button ajax_add_to_cart sr2-c-product-portal__quick-add"
					data-portal-quick-add
					data-product_id="<?php echo esc_attr( (string) $product_id ); ?>"
					data-product_sku="<?php echo esc_attr( $card_product->get_sku() ); ?>"
					data-quantity="1"
					aria-label="<?php echo esc_attr( sprintf( __( 'Add %s to bag', 'skyyrose-flagship-2' ), $product_name ) ); ?>"
				>
					<?php echo esc_html( $purchase_label ); ?>
				</a>
			<?php else : ?>
				<a class="sr2-c-product-portal__purchase-link" href="<?php echo esc_url( $product_url ); ?>">
					<?php echo esc_html( $purchase_label ); ?> <span aria-hidden="true">↗</span>
				</a>
			<?php endif; ?>
			<a class="sr2-c-product-portal__details" href="<?php echo esc_url( $product_url ); ?>">
				<?php esc_html_e( 'View piece', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span>
			</a>
			<span class="sr2-c-product-portal__action-status" data-portal-action-status role="status" aria-live="polite" aria-atomic="true">
				<?php echo esc_html( $purchase_note ); ?>
			</span>
		</div>
	</div>
</article>
<?php
$product = $previous_global_product;
