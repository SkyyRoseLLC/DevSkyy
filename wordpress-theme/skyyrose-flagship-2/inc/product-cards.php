<?php
/**
 * Shared product-card and Jersey Series rendering.
 *
 * Extracted from the flagship production architecture so V2 keeps commerce
 * concerns modular while retaining its SOT-first, skyyrose2-prefixed API.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

/**
 * Render the shared, collection-aware WooCommerce loop card.
 *
 * @param WC_Product $product Current WooCommerce product.
 * @param int        $index Product loop position.
 */
function skyyrose2_render_product_loop_card( $product, $index = 0 ) {
	if ( ! $product || ! is_a( $product, 'WC_Product' ) || ! $product->is_visible() ) {
		return;
	}

	get_template_part(
		'template-parts/commerce/product-card',
		null,
		array(
			'product' => $product,
			'index'   => max( 0, (int) $index ),
		)
	);
}

/**
 * Render the numbered Jersey Series as its own house chapter.
 *
 * The visual reveal is editorial; all names, media, prices and availability
 * continue to resolve from the current WooCommerce products.
 */
function skyyrose2_render_black_rose_jersey_series( $show_product_grid = true ) {
	if ( ! function_exists( 'wc_get_product' ) || ! function_exists( 'wc_get_product_id_by_sku' ) ) {
		return;
	}

	$registry = skyyrose2_presentation_registry();
	$series   = isset( $registry['products'] ) && is_array( $registry['products'] ) ? $registry['products'] : array();
	$pieces = array();

	foreach ( $series as $sku => $chapter_data ) {
		if ( 'jersey-series' !== ( $chapter_data['presentation'] ?? '' ) ) {
			continue;
		}
		$product_id = wc_get_product_id_by_sku( $sku );
		$product    = $product_id ? wc_get_product( $product_id ) : false;
		if ( $product ) {
			$pieces[] = array(
				'sku'     => $sku,
				'chapter' => __( $chapter_data['jersey_chapter'] ?? 'Jersey Series', 'skyyrose-flagship-2' ),
				'start'   => (float) ( $chapter_data['film_start'] ?? 0 ),
				'product' => $product,
			);
		}
	}

	if ( empty( $pieces ) ) {
		return;
	}
	?>
	<section class="sr2-jersey-reveal" aria-labelledby="sr2-jersey-series-title" data-presentation="jersey-series">
		<div class="sr2-jersey-reveal__head">
			<p class="sr2-eyebrow"><?php esc_html_e( 'Jersey Series / The Town Line', 'skyyrose-flagship-2' ); ?></p>
			<h2 id="sr2-jersey-series-title"><?php esc_html_e( 'Every number carries the tour.', 'skyyrose-flagship-2' ); ?></h2>
			<p><?php esc_html_e( 'Oakland is the origin. San Francisco, The Bay, and San Jose become chapters on The Town Line: SkyyRose’s fictional house journey. Every price, size, and availability decision stays on the live product page.', 'skyyrose-flagship-2' ); ?></p>
		</div>
		<div class="sr2-house-film" data-house-film data-house-film-autoplay="once" data-media-status="founder-review-candidate">
			<div class="sr2-house-film__media">
				<video width="1672" height="941" muted playsinline preload="none" poster="<?php echo esc_url( SKYYROSE2_URI . '/assets/sot/images/hero/jersey-series-town-line-train-v1.webp' ); ?>" data-house-film-video aria-label="<?php esc_attr_e( 'The Town Line Jersey Series previsualization', 'skyyrose-flagship-2' ); ?>">
					<source data-src="<?php echo esc_url( SKYYROSE2_URI . '/assets/video/skyyrose-tour-around-the-bay.webm' ); ?>" type="video/webm">
					<source data-src="<?php echo esc_url( SKYYROSE2_URI . '/assets/video/skyyrose-tour-around-the-bay.mp4' ); ?>" type="video/mp4">
				</video>
				<div class="sr2-house-film__controls">
					<button type="button" data-house-film-toggle><?php esc_html_e( 'Play film', 'skyyrose-flagship-2' ); ?></button>
					<button type="button" data-house-film-sound hidden><?php esc_html_e( 'Turn sound on', 'skyyrose-flagship-2' ); ?></button>
					<span class="screen-reader-text" aria-live="polite" data-house-film-status></span>
				</div>
			</div>
			<nav class="sr2-house-film__chapters" aria-label="<?php esc_attr_e( 'Jersey Series film chapters', 'skyyrose-flagship-2' ); ?>">
				<?php foreach ( $pieces as $piece ) : ?>
					<a href="<?php echo esc_url( get_permalink( $piece['product']->get_id() ) ); ?>" data-house-film-chapter data-start="<?php echo esc_attr( (string) $piece['start'] ); ?>"><span><?php echo esc_html( strtoupper( $piece['sku'] ) ); ?></span><strong><?php echo esc_html( $piece['chapter'] ); ?></strong></a>
				<?php endforeach; ?>
			</nav>
			<p class="sr2-house-film__transcript"><?php esc_html_e( 'Visual transcript: a fictional SkyyRose Town Line train introduces the Jersey Series. Available on-model review views move through Foundation, Oakland, San Francisco, The Bay, and San Jose before the tour closes on the house line. This previsualization is not product-media approval.', 'skyyrose-flagship-2' ); ?></p>
		</div>
		<?php if ( $show_product_grid ) : ?>
		<div class="sr2-jersey-reveal__grid">
			<?php foreach ( $pieces as $piece ) : ?>
				<?php
				$product     = $piece['product'];
				$product_url = get_permalink( $product->get_id() );
				$image_id    = $product->get_image_id();
				?>
				<article class="sr2-jersey-reveal__piece">
					<p><?php echo esc_html( $piece['chapter'] ); ?></p>
					<a href="<?php echo esc_url( $product_url ); ?>" aria-label="<?php echo esc_attr( sprintf( __( 'View %s', 'skyyrose-flagship-2' ), $product->get_name() ) ); ?>">
						<?php
						if ( $image_id ) {
							echo wp_kses_post( wp_get_attachment_image( $image_id, 'woocommerce_single', false, array( 'loading' => 'lazy', 'decoding' => 'async' ) ) );
						} else {
							echo '<span class="sr2-product__image-empty" aria-hidden="true"></span>';
						}
						?>
					</a>
					<h3><a href="<?php echo esc_url( $product_url ); ?>"><?php echo esc_html( $product->get_name() ); ?></a></h3>
					<a class="sr2-text-link" href="<?php echo esc_url( $product_url ); ?>"><?php esc_html_e( 'View the piece', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span></a>
				</article>
			<?php endforeach; ?>
		</div>
		<?php endif; ?>
	</section>
	<?php
}
