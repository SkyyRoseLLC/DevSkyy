<?php
/** Search results with real WordPress and WooCommerce results. @package SkyyRoseFlagship2 */
defined( 'ABSPATH' ) || exit;
get_header();
$query = get_search_query();
$product_results = null;
if ( class_exists( 'WooCommerce' ) && $query ) {
	$product_results = new WP_Query(
		array(
			'post_type'      => 'product',
			'post_status'    => 'publish',
			's'              => $query,
			'posts_per_page' => 12,
		)
	);
}
?>
<main id="primary" class="sr2-search" data-sr2-route="shop"><header class="sr2-journal__hero"><p class="sr2-eyebrow"><?php esc_html_e( 'Search the House', 'skyyrose-flagship-2' ); ?></p><h1><?php printf( esc_html__( 'Results for “%s”', 'skyyrose-flagship-2' ), esc_html( $query ) ); ?></h1><form class="sr2-search__form" role="search" method="get" action="<?php echo esc_url( home_url( '/' ) ); ?>"><label class="screen-reader-text" for="sr2-search-field"><?php esc_html_e( 'Search', 'skyyrose-flagship-2' ); ?></label><input id="sr2-search-field" type="search" name="s" value="<?php echo esc_attr( $query ); ?>" required><button class="sr2-c-action" type="submit"><?php esc_html_e( 'Search', 'skyyrose-flagship-2' ); ?></button></form></header><?php if ( $product_results && $product_results->have_posts() ) : ?><section class="sr2-search__products" aria-labelledby="sr2-search-products-title"><header class="sr2-section-head"><p class="sr2-eyebrow"><?php esc_html_e( 'Pieces', 'skyyrose-flagship-2' ); ?></p><h2 id="sr2-search-products-title"><?php esc_html_e( 'Product results', 'skyyrose-flagship-2' ); ?></h2></header><div class="sr2-c-product-grid"><?php $product_index = 0; while ( $product_results->have_posts() ) : $product_results->the_post(); $product = wc_get_product( get_the_ID() ); skyyrose2_render_product_loop_card( $product, $product_index ); ++$product_index; endwhile; wp_reset_postdata(); ?></div></section><?php endif; ?><?php if ( have_posts() ) : ?><section class="sr2-search__list" aria-label="<?php esc_attr_e( 'Stories and pages', 'skyyrose-flagship-2' ); ?>"><header class="sr2-section-head"><p class="sr2-eyebrow"><?php esc_html_e( 'Stories + pages', 'skyyrose-flagship-2' ); ?></p></header><?php while ( have_posts() ) : the_post(); ?><article class="sr2-search__result"><p class="sr2-eyebrow"><?php echo esc_html( get_post_type_object( get_post_type() )->labels->singular_name ); ?></p><h2><a href="<?php the_permalink(); ?>"><?php the_title(); ?></a></h2><p><?php echo esc_html( wp_trim_words( get_the_excerpt(), 32 ) ); ?></p></article><?php endwhile; ?></section><nav class="sr2-pagination"><?php the_posts_pagination(); ?></nav><?php elseif ( ! $product_results || ! $product_results->have_posts() ) : ?><section class="sr2-search__empty"><h2><?php esc_html_e( 'Nothing surfaced.', 'skyyrose-flagship-2' ); ?></h2><p><?php esc_html_e( 'Try a collection, product, or story title.', 'skyyrose-flagship-2' ); ?></p><a class="sr2-c-action" href="<?php echo esc_url( function_exists( 'wc_get_page_permalink' ) ? wc_get_page_permalink( 'shop' ) : home_url( '/shop/' ) ); ?>"><?php esc_html_e( 'Shop all pieces', 'skyyrose-flagship-2' ); ?></a></section><?php endif; ?></main>
<?php get_footer(); ?>
