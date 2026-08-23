<?php
/**
 * Template Name: SkyyRose Collection 2
 *
 * Shared story-commerce template for every collection world.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

$slug        = sanitize_title( get_post_field( 'post_name', get_queried_object_id() ) );
$collections = skyyrose2_collections();
$collection  = $collections[ $slug ] ?? $collections['signature'];
$shop_url    = '#shop';
$hero_desktop = skyyrose2_sot_asset_uri( $collection['hero'] );
$hero_tablet  = ! empty( $collection['hero_tablet'] ) ? skyyrose2_sot_asset_uri( $collection['hero_tablet'] ) : '';
$hero_mobile  = ! empty( $collection['hero_mobile'] ) ? skyyrose2_sot_asset_uri( $collection['hero_mobile'] ) : '';
$hero_motion = skyyrose2_collection_hero_motion( $slug, $collection['hero'] );

get_header();
?>
<main id="primary" class="sr2-collection" data-collection="<?php echo esc_attr( $slug ); ?>">
	<?php if ( function_exists( 'wc_print_notices' ) ) : ?>
		<div class="sr2-commerce-notices" aria-live="polite"><?php wc_print_notices(); ?></div>
	<?php endif; ?>
	<section class="sr2-collection-hero" aria-label="<?php echo esc_attr( sprintf( __( '%s collection monument scene', 'skyyrose-flagship-2' ), $collection['name'] ) ); ?>" data-hero-depth>
		<div class="sr2-collection-hero__media">
			<picture>
				<?php if ( $hero_mobile ) : ?><source media="(max-width: 47.99em)" srcset="<?php echo esc_url( $hero_mobile ); ?>"><?php endif; ?>
				<?php if ( $hero_tablet ) : ?><source media="(max-width: 74.99em)" srcset="<?php echo esc_url( $hero_tablet ); ?>"><?php endif; ?>
				<img src="<?php echo esc_url( $hero_desktop ); ?>" alt="" width="1440" height="810" fetchpriority="high" decoding="sync" data-art-direction="<?php echo esc_attr( $hero_mobile ? 'responsive' : 'desktop-fallback' ); ?>">
			</picture>
			<?php if ( $hero_motion ) : ?>
				<video class="sr2-collection-hero__motion" muted loop playsinline preload="none" poster="<?php echo esc_url( $hero_desktop ); ?>" data-collection-hero-video aria-hidden="true">
					<source data-src="<?php echo esc_url( $hero_motion['webm'] ); ?>" type="video/webm">
					<source data-src="<?php echo esc_url( $hero_motion['mp4'] ); ?>" type="video/mp4">
				</video>
			<?php endif; ?>
		</div>
		<div class="sr2-collection-hero__effects" data-scene-motion="<?php echo esc_attr( $slug ); ?>" aria-hidden="true">
			<span class="sr2-scene-effect sr2-scene-effect--hook"></span>
			<span class="sr2-scene-effect sr2-scene-effect--clouds"></span>
			<span class="sr2-scene-effect sr2-scene-effect--bridge-lights"></span>
			<span class="sr2-scene-effect sr2-scene-effect--signature-glow"></span>
			<span class="sr2-scene-effect sr2-scene-effect--throne-glow"></span>
			<span class="sr2-scene-effect sr2-scene-effect--petals">
				<?php for ( $petal = 0; $petal < 12; $petal++ ) : ?>
					<i></i>
				<?php endfor; ?>
			</span>
		</div>
		<div class="sr2-collection-hero__veil" aria-hidden="true"></div>
		<button class="sr2-collection-hero__motion-toggle" type="button" data-scene-motion-toggle aria-pressed="false" hidden><?php esc_html_e( 'Pause motion', 'skyyrose-flagship-2' ); ?></button>
	</section>

	<header class="sr2-collection-identity">
		<div class="sr2-collection-identity__copy">
			<p class="sr2-eyebrow"><?php echo esc_html( $collection['kicker'] ); ?></p>
			<h1 id="sr2-collection-title" class="sr2-collection-identity__title"><?php echo esc_html( $collection['name'] ); ?></h1>
			<p class="sr2-collection-identity__line"><?php echo esc_html( $collection['line'] ); ?></p>
			<div class="sr2-actions"><a class="sr2-button sr2-button--fill" href="<?php echo esc_url( $shop_url ); ?>"><?php echo esc_html( $collection['hero_cta'] ); ?></a><a class="sr2-button" href="#world"><?php echo esc_html( $collection['world_cta'] ); ?></a><a class="sr2-button" href="<?php echo esc_url( skyyrose2_immersive_url( $slug ) ); ?>"><?php esc_html_e( 'Enter the full scene', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span></a></div>
		</div>
		<p class="sr2-collection-identity__chapter" aria-hidden="true">WORLD / <?php echo esc_html( strtoupper( str_replace( '-', ' ', $slug ) ) ); ?></p>
	</header>

	<section class="sr2-collection-intro">
		<p><?php echo esc_html( $collection['kicker'] ); ?></p>
		<h2><?php echo esc_html( $collection['headline'] ); ?></h2>
		<span><?php echo esc_html( $collection['manifesto'] ); ?></span>
	</section>

	<section id="world" class="sr2-worlds" aria-labelledby="sr2-world-story-title" data-horizontal-world data-scroll-world-pinned>
		<header class="sr2-section-head sr2-section-head--split"><div><p><?php esc_html_e( 'Scroll World / Immersive Shopping', 'skyyrose-flagship-2' ); ?></p><h2 id="sr2-world-story-title"><?php echo esc_html( $collection['world_heading'] ); ?></h2><p><?php echo esc_html( $collection['world_intro'] ); ?></p></div><div class="sr2-rail-controls"><span data-rail-count>01 / <?php echo esc_html( sprintf( '%02d', count( $collection['world'] ) ) ); ?></span><button type="button" data-rail-prev aria-label="<?php esc_attr_e( 'Previous world chapter', 'skyyrose-flagship-2' ); ?>">←</button><button type="button" data-rail-next aria-label="<?php esc_attr_e( 'Next world chapter', 'skyyrose-flagship-2' ); ?>">→</button></div></header>
		<div class="sr2-worlds__stage" data-scroll-world-stage>
			<div class="sr2-worlds__rail" tabindex="0" aria-label="<?php echo esc_attr( sprintf( __( '%s scroll-world chapters', 'skyyrose-flagship-2' ), $collection['name'] ) ); ?>" data-horizontal-rail>
				<?php foreach ( $collection['world'] as $index => $scene ) : ?>
					<?php $scene_uri = skyyrose2_collection_scene_uri( $scene ); ?>
					<?php $scene_product = skyyrose2_collection_scene_product( $slug, $index ); ?>
					<article class="sr2-world">
						<img src="<?php echo esc_url( $scene_uri ); ?>" alt="" width="1920" height="1080" loading="lazy" decoding="async">
						<div class="sr2-world__shade" aria-hidden="true"></div>
						<?php if ( $scene_product && $scene_product->get_image_id() ) : ?>
							<a class="sr2-world__product" href="<?php echo esc_url( $scene_product->get_permalink() ); ?>" aria-label="<?php echo esc_attr( sprintf( __( 'Shop %s from the %s collection', 'skyyrose-flagship-2' ), $scene_product->get_name(), $collection['name'] ) ); ?>">
								<?php echo wp_kses_post( wp_get_attachment_image( $scene_product->get_image_id(), 'woocommerce_thumbnail', false, array( 'class' => 'sr2-world__product-image', 'loading' => 'lazy', 'decoding' => 'async' ) ) ); ?>
								<span class="sr2-world__product-copy"><small><?php esc_html_e( 'Piece in this world', 'skyyrose-flagship-2' ); ?></small><strong><?php echo esc_html( $scene_product->get_name() ); ?></strong><em><?php esc_html_e( 'View product proof', 'skyyrose-flagship-2' ); ?> ↗</em></span>
							</a>
						<?php endif; ?>
						<div class="sr2-world__copy"><small><?php echo esc_html( sprintf( __( 'Chapter %02d', 'skyyrose-flagship-2' ), $index + 1 ) ); ?></small><strong><?php echo esc_html( $scene['label'] ); ?></strong><em><?php echo esc_html( $scene['copy'] ); ?></em><a href="#shop"><?php echo esc_html( sprintf( __( 'View %s pieces', 'skyyrose-flagship-2' ), $collection['name'] ) ); ?> <span aria-hidden="true">↗</span></a></div>
					</article>
				<?php endforeach; ?>
			</div>
			<div class="sr2-rail-progress" aria-hidden="true"><span data-rail-progress></span></div>
		</div>
	</section>

	<section id="shop" class="sr2-section sr2-section--products sr2-collection-shop">
		<header class="sr2-section-head sr2-section-head--split"><div><p><?php echo esc_html( $collection['shop_kicker'] ); ?></p><h2><?php echo esc_html( $collection['shop_heading'] ); ?></h2><p><?php echo esc_html( $collection['shop_intro'] ); ?></p></div><span class="sr2-section-index">03 / SHOP</span></header>
		<?php skyyrose2_product_cards( 12, $slug ); ?>
		<?php if ( 'black-rose' === $slug ) : ?>
			<div id="jersey-series" class="sr2-collection-subchapter" aria-labelledby="sr2-jersey-series-title">
				<p class="sr2-eyebrow"><?php esc_html_e( 'Black Rose release / The Town Line', 'skyyrose-flagship-2' ); ?></p>
				<h2 id="sr2-jersey-series-title"><?php esc_html_e( 'Jersey Series', 'skyyrose-flagship-2' ); ?></h2>
				<p><?php esc_html_e( 'A numbered Black Rose release with its own SkyyRose Tour reveal. The chapter stays visually distinct while every jersey remains part of the Black Rose collection.', 'skyyrose-flagship-2' ); ?></p>
				<?php skyyrose2_render_black_rose_jersey_series( true ); ?>
			</div>
		<?php endif; ?>
	</section>

	<section class="sr2-manifesto" aria-labelledby="sr2-manifesto-title">
		<div class="sr2-manifesto__sticky">
			<p class="sr2-eyebrow"><?php esc_html_e( 'Collection Code', 'skyyrose-flagship-2' ); ?></p>
			<h2 id="sr2-manifesto-title"><?php echo esc_html( $collection['headline'] ); ?></h2>
			<img class="sr2-manifesto__mark" src="<?php echo esc_url( skyyrose2_sot_asset_uri( $collection['atmosphere'] ) ); ?>" alt="" width="600" height="600" loading="lazy" decoding="async">
		</div>
		<div class="sr2-manifesto__scroll">
			<article><span>01</span><h3><?php esc_html_e( 'The meaning', 'skyyrose-flagship-2' ); ?></h3><p><?php echo esc_html( $collection['manifesto'] ); ?></p></article>
			<figure class="sr2-image-reveal"><picture><?php if ( ! empty( $collection['lookbook_mobile'] ) ) : ?><source media="(max-width: 47.99em)" srcset="<?php echo esc_url( skyyrose2_sot_asset_uri( $collection['lookbook_mobile'] ) ); ?>"><?php endif; ?><img src="<?php echo esc_url( skyyrose2_sot_asset_uri( $collection['lookbook'] ) ); ?>" alt="<?php echo esc_attr( $collection['name'] . ' lookbook' ); ?>" width="960" height="1200" loading="lazy" decoding="async"></picture><figcaption><?php echo esc_html( $collection['name'] ); ?> / LOOK 01</figcaption></figure>
			<article><span>02</span><h3><?php echo esc_html( $collection['invitation_title'] ); ?></h3><p><?php echo esc_html( $collection['invitation'] ); ?></p><a class="sr2-button sr2-button--fill" href="#shop"><?php echo esc_html( $collection['invitation_cta'] ); ?></a></article>
		</div>
	</section>

	<nav class="sr2-crossnav" aria-label="<?php esc_attr_e( 'Explore other collections', 'skyyrose-flagship-2' ); ?>">
		<p class="sr2-eyebrow"><?php esc_html_e( 'Continue Through the House', 'skyyrose-flagship-2' ); ?></p>
		<?php foreach ( $collections as $nav_slug => $nav_collection ) : ?>
			<?php if ( $nav_slug !== $slug ) : ?>
				<a data-collection="<?php echo esc_attr( $nav_slug ); ?>" href="<?php echo esc_url( skyyrose2_collection_url( $nav_slug ) ); ?>"><span><?php echo esc_html( $nav_collection['name'] ); ?></span><em><?php echo esc_html( $nav_collection['kicker'] ); ?></em><b aria-hidden="true">↗</b></a>
			<?php endif; ?>
		<?php endforeach; ?>
	</nav>
</main>
<?php
get_footer();
