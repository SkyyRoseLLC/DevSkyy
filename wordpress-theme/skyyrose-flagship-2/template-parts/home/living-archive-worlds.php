<?php
/** Four existing worlds, indexed at unequal desktop scales. @package SkyyRoseFlagship2 */
defined( 'ABSPATH' ) || exit;
$archive_worlds = $args['collections'] ?? array();
$archive_world_index = 0;
?>
<div class="sr2-archive-worlds__grid">
	<?php foreach ( $archive_worlds as $archive_world_slug => $archive_world ) : ?>
		<?php
		++$archive_world_index;
		$archive_world_large = in_array( $archive_world_index, array( 1, 4 ), true );
		$archive_world_sizes = $archive_world_large
			? '(max-width: 47.99em) calc((100vw - 3rem) / 2), (max-width: 63.99em) calc((100vw - 4rem) / 2), (max-width: 95.75em) calc(58.333333vw - 2rem), 862px'
			: '(max-width: 47.99em) calc((100vw - 3rem) / 2), (max-width: 63.99em) calc((100vw - 4rem) / 2), (max-width: 95.75em) calc(41.666667vw - 2rem), 607px';
		$archive_world_srcset = skyyrose2_sot_asset_uri( $archive_world['hero_mobile'] ) . ' 640w, ' . skyyrose2_sot_asset_uri( $archive_world['hero_tablet'] ) . ' 1024w, ' . skyyrose2_sot_asset_uri( $archive_world['hero'] ) . ' 1440w';
		?>
		<article class="sr2-archive-world" data-collection="<?php echo esc_attr( $archive_world_slug ); ?>"><a href="<?php echo esc_url( skyyrose2_collection_url( $archive_world_slug ) ); ?>">
			<div class="sr2-archive-world__image"><img src="<?php echo esc_url( skyyrose2_sot_asset_uri( $archive_world['hero_mobile'] ) ); ?>" srcset="<?php echo esc_attr( $archive_world_srcset ); ?>" sizes="<?php echo esc_attr( $archive_world_sizes ); ?>" width="1440" height="810" alt="" loading="lazy" decoding="async"></div>
			<div class="sr2-archive-world__identity"><span class="sr2-world-index" aria-hidden="true"><?php echo esc_html( sprintf( '%02d', $archive_world_index ) ); ?></span><h3><?php echo esc_html( $archive_world['name'] ); ?></h3><span class="sr2-archive-world__arrow" aria-hidden="true">↗</span><p><?php echo esc_html( $archive_world['kicker'] ); ?></p></div>
		</a></article>
	<?php endforeach; ?>
</div>
