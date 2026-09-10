<?php
/** Front Page — The Living Archive. Eight static acts, shared native commerce. @package SkyyRoseFlagship2 */
defined( 'ABSPATH' ) || exit;
$archive_collections = skyyrose2_collections();
$archive_hero_motion = skyyrose2_collection_hero_motion( 'black-rose', $archive_collections['black-rose']['hero'] );
$archive_shop = skyyrose2_shop_url();
$archive_about = skyyrose2_marketplace_page_url( 'about' );
$archive_artifact = skyyrose2_get_products_by_skus( array( 'sg-005' ), 'signature' );
$archive_heirs = skyyrose2_get_products_by_skus( array( 'kids-001', 'kids-002' ), 'kids-capsule' );
$archive_continue = array(
	__( 'Collections', 'skyyrose-flagship-2' ) => skyyrose2_marketplace_page_url( 'collections' ),
	__( 'Shop', 'skyyrose-flagship-2' ) => $archive_shop,
	__( 'Journal', 'skyyrose-flagship-2' ) => skyyrose2_marketplace_page_url( 'journal' ),
	__( 'Pre-Order', 'skyyrose-flagship-2' ) => skyyrose2_marketplace_page_url( 'pre-order' ),
	__( 'Client Services', 'skyyrose-flagship-2' ) => skyyrose2_marketplace_page_url( 'contact' ),
);
get_header();
?>
<main id="primary" class="sr2-archive" tabindex="-1">
<div class="sr2-archive__inner">
	<?php if ( function_exists( 'wc_print_notices' ) ) : ?><div class="sr2-commerce-notices" aria-live="polite"><?php wc_print_notices(); ?></div><?php endif; ?>
	<section id="sr2-archive-arrival" class="sr2-archive-scene" data-recovery-hero aria-labelledby="sr2-archive-title" data-archive-act="1">
		<div class="sr2-archive-scene__image" aria-hidden="true">
			<picture>
				<source media="(max-width: 47.99em)" srcset="<?php echo esc_url( skyyrose2_sot_asset_uri( 'images/hero/responsive/black-rose-bay-bridge-monuments-v4-640w.webp' ) ); ?>">
				<source media="(max-width: 74.99em)" srcset="<?php echo esc_url( skyyrose2_sot_asset_uri( 'images/hero/responsive/black-rose-bay-bridge-monuments-v4-1024w.webp' ) ); ?>">
				<img src="<?php echo esc_url( skyyrose2_sot_asset_uri( 'images/hero/responsive/black-rose-bay-bridge-monuments-v4-1440w.webp' ) ); ?>" width="1440" height="810" alt="" fetchpriority="high" loading="eager" decoding="async">
			</picture>
			<?php if ( $archive_hero_motion ) : ?><video data-recovery-hero-video muted loop playsinline preload="none" aria-hidden="true"><source data-src="<?php echo esc_url( $archive_hero_motion['webm'] ); ?>" type="video/webm"><source data-src="<?php echo esc_url( $archive_hero_motion['mp4'] ); ?>" type="video/mp4"></video><?php endif; ?>
		</div>
		<div class="sr2-archive-scene__veil" aria-hidden="true"></div>
		<div class="sr2-archive-scene__copy">
			<p class="sr2-world-index"><?php esc_html_e( 'Oakland, California / The Living Archive', 'skyyrose-flagship-2' ); ?></p>
			<h1 id="sr2-archive-title"><?php esc_html_e( 'Skyy Rose', 'skyyrose-flagship-2' ); ?></h1>
			<p class="sr2-archive-scene__intro"><?php esc_html_e( 'A house built by a father, named after a daughter, and rooted in The Town.', 'skyyrose-flagship-2' ); ?></p>
			<div class="sr2-archive-scene__actions"><a class="sr2-control sr2-control--primary" href="#sr2-archive-worlds"><?php esc_html_e( 'Choose your world', 'skyyrose-flagship-2' ); ?></a><a class="sr2-control sr2-control--secondary" href="<?php echo esc_url( $archive_shop ); ?>"><?php esc_html_e( 'Shop the house', 'skyyrose-flagship-2' ); ?></a></div>
		</div>
		<div id="skyy-hero-stage" class="sr2-archive-scene__concierge" role="region" aria-label="<?php esc_attr_e( 'Skyy, your house concierge', 'skyyrose-flagship-2' ); ?>">
			<noscript><img src="<?php echo esc_url( skyyrose2_sot_asset_uri( 'images/mascot/skyy-canonical-v2-512w.webp' ) ); ?>" width="160" height="240" alt="<?php esc_attr_e( 'Skyy, the house concierge', 'skyyrose-flagship-2' ); ?>" loading="lazy"><a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'contact' ) ); ?>"><?php esc_html_e( 'Ask Skyy / Client Services', 'skyyrose-flagship-2' ); ?></a></noscript>
		</div>
		<button class="sr2-recovery-motion-toggle" type="button" data-recovery-motion-toggle hidden><?php esc_html_e( 'Pause motion', 'skyyrose-flagship-2' ); ?></button>
		<nav class="sr2-archive-scene__worlds" aria-label="<?php esc_attr_e( 'Opening collection worlds', 'skyyrose-flagship-2' ); ?>">
			<?php $arrival_index = 0; foreach ( array( 'signature', 'black-rose', 'love-hurts' ) as $arrival_slug ) : ?>
				<a href="<?php echo esc_url( skyyrose2_collection_url( $arrival_slug ) ); ?>"><span aria-hidden="true"><?php echo esc_html( sprintf( '%02d', ++$arrival_index ) ); ?></span><b><?php echo esc_html( $archive_collections[ $arrival_slug ]['name'] ); ?></b></a>
			<?php endforeach; ?>
		</nav>
	</section>
	<section id="sr2-archive-worlds" class="sr2-archive-worlds" aria-labelledby="sr2-archive-worlds-title" tabindex="-1" data-archive-act="2">
		<header class="sr2-archive-section-head"><p class="sr2-world-index"><?php esc_html_e( 'II / The four worlds', 'skyyrose-flagship-2' ); ?></p><h2 id="sr2-archive-worlds-title"><?php esc_html_e( 'Four stories. One house.', 'skyyrose-flagship-2' ); ?></h2></header>
		<?php get_template_part( 'template-parts/home/living-archive-worlds', null, array( 'collections' => $archive_collections ) ); ?>
	</section>
	<section id="sr2-archive-oakland" class="sr2-archive-oakland" aria-labelledby="sr2-archive-oakland-title" data-archive-act="3">
		<p class="sr2-world-index"><?php esc_html_e( 'III / Oakland', 'skyyrose-flagship-2' ); ?></p>
		<h2 id="sr2-archive-oakland-title"><?php esc_html_e( 'Our Oakland roots', 'skyyrose-flagship-2' ); ?></h2>
		<div class="sr2-archive-oakland__provenance"><p><?php esc_html_e( 'SkyyRose began as Corey Foster’s promise to build a future Skyy Rose could recognize herself inside.', 'skyyrose-flagship-2' ); ?></p><p><?php esc_html_e( 'The house keeps Oakland in the frame: concrete, care, memory, and the refusal to shrink.', 'skyyrose-flagship-2' ); ?></p></div>
	</section>
	<section id="sr2-archive-artifact" class="sr2-archive-artifact" aria-labelledby="sr2-archive-artifact-title" data-archive-act="4" data-collection="signature">
		<div class="sr2-archive-artifact__copy"><p class="sr2-world-index"><?php esc_html_e( 'IV / Product as artifact', 'skyyrose-flagship-2' ); ?></p><h2 id="sr2-archive-artifact-title"><?php echo esc_html( $archive_collections['signature']['shop_heading'] ); ?></h2><p><?php echo esc_html( $archive_collections['signature']['card_story'] ); ?></p><a class="sr2-world-text-link" href="<?php echo esc_url( skyyrose2_collection_url( 'signature' ) ); ?>"><?php esc_html_e( 'Explore Signature', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span></a></div>
		<?php if ( $archive_artifact ) : ?>
			<?php get_template_part( 'template-parts/collections/product-edit', null, array( 'products' => array_values( $archive_artifact ) ) ); ?>
		<?php else : ?><p class="sr2-archive-empty"><?php esc_html_e( 'This piece is not currently listed. Explore the Signature collection for its current edit.', 'skyyrose-flagship-2' ); ?></p><?php endif; ?>
	</section>
	<section id="sr2-archive-heir" class="sr2-archive-heir" aria-labelledby="sr2-archive-heir-title" data-archive-act="5" data-collection="kids-capsule">
		<header class="sr2-archive-heir__heading"><p class="sr2-world-index"><?php esc_html_e( 'V / Kids Capsule', 'skyyrose-flagship-2' ); ?></p><h2 id="sr2-archive-heir-title"><?php esc_html_e( 'The Heir', 'skyyrose-flagship-2' ); ?></h2><p><?php echo esc_html( $archive_collections['kids-capsule']['manifesto'] ); ?></p></header>
		<?php if ( $archive_heirs ) : ?>
			<?php get_template_part( 'template-parts/collections/product-edit', null, array( 'products' => array_values( $archive_heirs ) ) ); ?>
		<?php else : ?><p class="sr2-archive-empty"><?php esc_html_e( 'No Kids Capsule pieces are currently listed.', 'skyyrose-flagship-2' ); ?></p><?php endif; ?>
		<a class="sr2-world-text-link" href="<?php echo esc_url( skyyrose2_collection_url( 'kids-capsule' ) ); ?>"><?php esc_html_e( 'Enter the heir’s world', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span></a>
	</section>
	<section id="sr2-archive-town-line" class="sr2-archive-town-line" aria-labelledby="sr2-archive-town-line-title" data-archive-act="6">
		<h2 id="sr2-archive-town-line-title" class="screen-reader-text"><?php esc_html_e( 'VI / Town Line prelude', 'skyyrose-flagship-2' ); ?></h2>
		<?php
		ob_start();
		get_template_part( 'template-parts/commerce/town-line' );
		$archive_town_line = trim( ob_get_clean() );
		if ( $archive_town_line ) {
			echo $archive_town_line; // phpcs:ignore WordPress.Security.EscapeOutput.OutputNotEscaped -- Shared template escapes native product data.
		} else {
			?>
			<p class="sr2-world-index"><?php esc_html_e( 'VI / The Town Line', 'skyyrose-flagship-2' ); ?></p><p><?php esc_html_e( 'No Jersey Series pieces are currently listed.', 'skyyrose-flagship-2' ); ?></p><a class="sr2-world-text-link" href="<?php echo esc_url( $archive_shop ); ?>"><?php esc_html_e( 'Browse the Shop', 'skyyrose-flagship-2' ); ?></a>
			<?php
		}
		?>
	</section>
	<section id="sr2-archive-legacy" class="sr2-archive-legacy" aria-labelledby="sr2-archive-legacy-title" data-archive-act="7">
		<figure><img src="<?php echo esc_url( skyyrose2_sot_asset_uri( 'images/about/skyy-rose-founder-hero.webp' ) ); ?>" width="724" height="1086" alt="<?php esc_attr_e( 'Skyy Rose, the daughter whose name inspired the SkyyRose house.', 'skyyrose-flagship-2' ); ?>" loading="lazy" decoding="async"><figcaption><?php esc_html_e( 'Skyy Rose', 'skyyrose-flagship-2' ); ?></figcaption></figure>
		<div class="sr2-archive-legacy__copy"><p class="sr2-world-index"><?php esc_html_e( 'VII / Legacy', 'skyyrose-flagship-2' ); ?></p><h2 id="sr2-archive-legacy-title"><?php esc_html_e( 'A legacy in bloom.', 'skyyrose-flagship-2' ); ?></h2><p><?php esc_html_e( 'SkyyRose is more than a brand—it is a promise. To my daughter. To our city. To every soul that believes in turning pain into purpose and dreams into reality.', 'skyyrose-flagship-2' ); ?></p><a class="sr2-world-text-link" href="<?php echo esc_url( $archive_about ); ?>"><?php esc_html_e( 'Read the house story', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span></a></div>
	</section>
	<section id="sr2-archive-continue" class="sr2-archive-continue" aria-labelledby="sr2-archive-continue-title" data-archive-act="8">
		<header><p class="sr2-world-index"><?php esc_html_e( 'VIII / Continue', 'skyyrose-flagship-2' ); ?></p><h2 id="sr2-archive-continue-title"><?php esc_html_e( 'Find your next chapter.', 'skyyrose-flagship-2' ); ?></h2></header>
		<nav aria-label="<?php esc_attr_e( 'Continue through the house', 'skyyrose-flagship-2' ); ?>"><ol>
			<?php $archive_destination_index = 0; ?>
			<?php foreach ( $archive_continue as $archive_label => $archive_url ) : ?>
				<li><a href="<?php echo esc_url( $archive_url ); ?>"><span class="sr2-world-index" aria-hidden="true"><?php echo esc_html( sprintf( '%02d', ++$archive_destination_index ) ); ?></span><span><?php echo esc_html( $archive_label ); ?></span><span aria-hidden="true">↗</span></a></li>
			<?php endforeach; ?>
		</ol></nav>
	</section>
</div>
</main>
<?php get_footer(); ?>
