<?php
/** Theme-owned progressive wishlist; WooCommerce remains product truth. @package SkyyRoseFlagship2 */
defined( 'ABSPATH' ) || exit;

$wishlist_products = function_exists( 'skyyrose2_get_products' ) ? skyyrose2_get_products( 48 ) : array();
get_header();
while ( have_posts() ) : the_post();
?>
<main id="primary" class="sr2-generic-page sr2-c-service sr2-wishlist-page" data-sr2-route="service" data-wishlist-page>
	<header class="sr2-generic-head"><p class="sr2-eyebrow"><?php esc_html_e( 'Saved pieces', 'skyyrose-flagship-2' ); ?></p><h1><?php the_title(); ?></h1><p><?php esc_html_e( 'Keep the pieces that stay with you. Saved items live in this browser until you are ready to enter their product page.', 'skyyrose-flagship-2' ); ?></p></header>
	<div class="sr2-page-copy sr2-page-copy--generic"><?php the_content(); ?></div>
	<p class="sr2-wishlist-page__empty" data-wishlist-empty hidden><?php esc_html_e( 'Nothing saved yet. Start with a verified piece from the house.', 'skyyrose-flagship-2' ); ?></p>
	<div class="sr2-products sr2-products--prototype sr2-wishlist-page__grid" data-wishlist-grid>
		<?php foreach ( $wishlist_products as $wishlist_product ) : ?>
			<div data-wishlist-item data-product-id="<?php echo esc_attr( (string) $wishlist_product->get_id() ); ?>">
				<?php skyyrose2_render_product_loop_card( $wishlist_product, 0 ); ?>
			</div>
		<?php endforeach; ?>
	</div>
</main>
<?php endwhile; get_footer(); ?>
