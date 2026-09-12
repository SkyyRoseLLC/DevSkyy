<?php
/**
 * Full-frame, responsive composition with independently shoppable products.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;
$scene = $args['scene'];
$record = $scene['hero_composition'];
$collection = $args['collection'];
$index = absint( $args['index'] ?? 0 );
$scene_id = sanitize_html_class( strtolower( $scene['scene_id'] ) );
$products = skyyrose2_resolve_commerce_scene_products( $scene, $collection );
$srcset = array();
$final_manifest_path = SKYYROSE2_DIR . '/data/approved-scroll-world-scenes.json';
$final_manifest = is_readable( $final_manifest_path ) ? json_decode( file_get_contents( $final_manifest_path ), true ) : array(); // phpcs:ignore WordPress.WP.AlternativeFunctions.file_get_contents_file_get_contents
$final_scene = $final_manifest['scenes'][ $scene['scene_id'] ] ?? array();
foreach ( $final_scene['responsive_posters'] ?? array() as $derivative ) {
	$path = (string) ( $derivative['path'] ?? '' );
	if ( 0 === strpos( $path, 'assets/scroll-world/derived/approved-scenes/' ) && false === strpos( $path, '..' ) && ! empty( $derivative['width'] ) ) {
		$srcset[] = SKYYROSE2_URI . '/' . $path . ' ' . absint( $derivative['width'] ) . 'w';
	}
}
foreach ( (array) ( $record['variants'] ?? array() ) as $variant ) {
	$asset = (string) ( $variant['asset'] ?? '' );
	if ( 0 === strpos( $asset, 'generated-candidates/hero-commerce-c1/' ) && false === strpos( $asset, '..' ) && ! empty( $variant['width'] ) ) {
		$srcset[] = skyyrose2_scroll_world_asset_uri( $asset ) . ' ' . absint( $variant['width'] ) . 'w';
	}
}
?>
<article class="sr2-world sr2-world--commerce sr2-world--hero-composed" style="--scene-ratio: <?php echo esc_attr( (string) ( absint( $scene['width'] ) / max( 1, absint( $scene['height'] ) ) ) ); ?>;" data-scene-id="<?php echo esc_attr( $scene_id ); ?>" data-product-state="<?php echo esc_attr( $products['state'] ); ?>" data-composite-state="<?php echo empty( $scene['scene_motion'] ) ? 'candidate' : 'founder-approved-local-motion'; ?>" data-product-count="<?php echo esc_attr( count( $scene['product_bindings'] ) ); ?>">
	<figure class="sr2-hero-commerce__frame" style="--scene-aspect: <?php echo absint( $scene['width'] ); ?> / <?php echo absint( $scene['height'] ); ?>;">
		<img data-scene-poster data-src="<?php echo esc_url( skyyrose2_collection_scene_uri( $scene ) ); ?>"<?php if ( $srcset ) : ?> data-srcset="<?php echo esc_attr( implode( ', ', $srcset ) ); ?>" sizes="(max-width: 767px) 100vw, 92vw"<?php endif; ?> alt="<?php echo esc_attr( $scene['direction'] ); ?>" width="<?php echo esc_attr( $scene['width'] ); ?>" height="<?php echo esc_attr( $scene['height'] ); ?>" decoding="async">
		<noscript><style>[data-scene-poster]{display:none!important}</style><img src="<?php echo esc_url( skyyrose2_collection_scene_uri( $scene ) ); ?>"<?php if ( $srcset ) : ?> srcset="<?php echo esc_attr( implode( ', ', $srcset ) ); ?>" sizes="(max-width: 767px) 100vw, 92vw"<?php endif; ?> alt="<?php echo esc_attr( $scene['direction'] ); ?>" width="<?php echo esc_attr( $scene['width'] ); ?>" height="<?php echo esc_attr( $scene['height'] ); ?>" loading="lazy" decoding="async"></noscript>

		<?php if ( ! empty( $scene['scene_motion'] ) ) : ?>
			<video
				id="scene-motion-<?php echo esc_attr( $scene_id ); ?>"
				data-collection-scene-motion
				data-scene-label="<?php echo esc_attr( $scene['label'] ); ?>"
				data-desktop-src="<?php echo esc_url( skyyrose2_scroll_world_asset_uri( $scene['scene_motion']['desktop'] ) ); ?>"
				data-mobile-src="<?php echo esc_url( skyyrose2_scroll_world_asset_uri( $scene['scene_motion']['mobile'] ) ); ?>"
				muted loop playsinline preload="none"
				width="<?php echo esc_attr( $scene['width'] ); ?>"
				height="<?php echo esc_attr( $scene['height'] ); ?>"
				aria-hidden="true" tabindex="-1"
			></video>
		<?php endif; ?>
	</figure>
	<?php if ( ! empty( $scene['scene_motion'] ) ) : ?>
		<div class="sr2-scene-motion-controls">
			<button type="button" class="sr2-scene-motion__toggle" data-scene-motion-toggle
				aria-controls="scene-motion-<?php echo esc_attr( $scene_id ); ?>"
				aria-label="<?php echo esc_attr( 'Play motion: ' . $scene['label'] ); ?>" hidden>Play motion</button>
		</div>
	<?php endif; ?>
	<div class="sr2-hero-commerce__details">
		<header class="sr2-hero-commerce__story">
			<p class="sr2-eyebrow"><?php echo esc_html( sprintf( __( 'Chapter %02d', 'skyyrose-flagship-2' ), $index + 1 ) ); ?></p>
			<h3<?php if ( 2 === $index ) : ?> data-sr2-type-motion="statement"<?php endif; ?>><?php echo esc_html( $scene['label'] ); ?></h3>
			<p><?php echo esc_html( $scene['copy'] ); ?></p>
			<?php if ( ! empty( $record['review_message'] ) ) : ?><p class="sr2-hero-commerce__review"><?php echo esc_html( $record['review_message'] ); ?></p><?php endif; ?>
		</header>
		<nav class="sr2-hero-commerce__products" aria-labelledby="<?php echo esc_attr( 'scene-products-title-' . $scene_id ); ?>">
			<h4 id="<?php echo esc_attr( 'scene-products-title-' . $scene_id ); ?>" tabindex="-1"><?php esc_html_e( 'Explore the pieces', 'skyyrose-flagship-2' ); ?></h4>
			<ul>
				<?php foreach ( $products['slots'] as $slot ) : ?>
					<li>
						<?php if ( $slot['product'] ) : ?>
							<a href="<?php echo esc_url( $slot['product']->get_permalink() ); ?>">
								<span><?php echo esc_html( $slot['product']->get_name() ); ?></span>
								<small><?php echo wp_kses_post( $slot['product']->get_price_html() ); ?></small>
								<em><?php echo esc_html( skyyrose2_scene_product_action_label( $slot['product'], ! empty( $scene['preorder_product_links_required'] ) ) ); ?> &rarr;</em>
							</a>
						<?php else : ?>
							<span><?php echo esc_html( sprintf( __( '%s is not currently available.', 'skyyrose-flagship-2' ), strtoupper( $slot['sku'] ) ) ); ?></span>
						<?php endif; ?>
					</li>
				<?php endforeach; ?>
			</ul>
		</nav>
	</div>
</article>
