<?php
/** Branded recovery route. @package SkyyRoseFlagship2 */
defined( 'ABSPATH' ) || exit;
get_header();
?>
<?php $shop_url = function_exists( 'wc_get_page_permalink' ) ? wc_get_page_permalink( 'shop' ) : home_url( '/shop/' ); ?>
<main id="primary" class="sr2-not-found" data-sr2-route="service"><div class="sr2-not-found__media"><img src="<?php echo esc_url( skyyrose2_sot_asset_uri( 'images/mascot/skyy-canonical-v2.png' ) ); ?>" alt="" width="1024" height="1024"></div><div><p class="sr2-eyebrow">404 / SkyyRose</p><h1><?php esc_html_e( 'This chapter moved.', 'skyyrose-flagship-2' ); ?></h1><p><?php esc_html_e( 'The house is still here. Find a collection, a piece, or a story.', 'skyyrose-flagship-2' ); ?></p><div class="sr2-not-found__actions"><a class="sr2-c-action" href="<?php echo esc_url( home_url( '/collections/' ) ); ?>"><?php esc_html_e( 'Explore worlds', 'skyyrose-flagship-2' ); ?></a><a class="sr2-text-link" href="<?php echo esc_url( $shop_url ); ?>"><?php esc_html_e( 'Shop pieces', 'skyyrose-flagship-2' ); ?> ↗</a></div></div></main>
<?php get_footer(); ?>
