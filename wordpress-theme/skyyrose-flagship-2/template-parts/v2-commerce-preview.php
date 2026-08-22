<?php
/**
 * Local-only commerce state preview.
 *
 * The installed WooCommerce templates remain authoritative in production.
 * This fixture lets the pitch review inspect the branded bag, checkout,
 * account, and tracking shells without inventing a live order or payment.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

global $route;
$routes = array(
	'cart'       => array( 'kicker' => 'Your bag', 'title' => 'Keep your pieces close.', 'copy' => 'Your selected pieces and live WooCommerce totals appear here after a real add-to-cart action.', 'cta' => 'Shop collections', 'href' => home_url( '/collections/' ) ),
	'checkout'   => array( 'kicker' => 'Secure checkout', 'title' => 'Your pieces are almost home.', 'copy' => 'Billing, shipping, order review, tax, payment, and recovery states are supplied by WooCommerce at runtime.', 'cta' => 'Return to the shop', 'href' => wc_get_page_permalink( 'shop' ) ),
	'account'    => array( 'kicker' => 'Client account', 'title' => 'Your archive, in one place.', 'copy' => 'Orders, downloads, addresses, and account ownership remain inside the WooCommerce account lifecycle.', 'cta' => 'Shop collections', 'href' => home_url( '/collections/' ) ),
	'order-tracking' => array( 'kicker' => 'Order tracking', 'title' => 'Follow the piece home.', 'copy' => 'Enter the order number and billing email to use the live WooCommerce tracking lookup.', 'cta' => 'Contact the house', 'href' => home_url( '/contact/' ) ),
);
$state = $routes[ $route ] ?? $routes['cart'];

get_header();
?>
<main id="primary" class="sr2-commerce-shell sr2-commerce-preview" data-sr2-route="<?php echo esc_attr( $route ); ?>">
	<header class="sr2-commerce-preview__hero">
		<p class="sr2-eyebrow"><?php echo esc_html( $state['kicker'] ); ?></p>
		<h1><?php echo esc_html( $state['title'] ); ?></h1>
		<p><?php echo esc_html( $state['copy'] ); ?></p>
	</header>
	<section class="sr2-commerce-preview__panel" aria-label="<?php echo esc_attr( $state['kicker'] ); ?> preview state">
		<?php if ( 'cart' === $route ) : ?>
			<p class="sr2-eyebrow">Empty bag state</p><h2>Nothing here yet.</h2><p>When WooCommerce confirms a product and variation, the bag updates with the live item, quantity, price, and remove controls.</p>
		<?php elseif ( 'checkout' === $route ) : ?>
			<div><p class="sr2-eyebrow">Customer details</p><label>First name<input type="text" placeholder="Preview only" disabled></label><label>Email<input type="email" placeholder="Live WooCommerce field" disabled></label></div><div><p class="sr2-eyebrow">Order review</p><h2>Secure payment handoff</h2><p>Payment methods, tax, shipping, validation, failure recovery, and order creation are intentionally not fabricated in this local fixture.</p></div>
		<?php elseif ( 'order-tracking' === $route ) : ?>
			<label>Order number<input type="text" placeholder="#0000" disabled></label><label>Billing email<input type="email" placeholder="you@example.com" disabled></label><button class="sr2-button sr2-button--fill" type="button" disabled>Track order</button>
		<?php else : ?>
			<p class="sr2-eyebrow">WooCommerce state authority</p><h2>Sign in to continue.</h2><p>Account ownership, saved addresses, order history, and password recovery are rendered by the live WooCommerce account template after install.</p>
		<?php endif; ?>
	</section>
	<a class="sr2-button sr2-button--fill" href="<?php echo esc_url( $state['href'] ); ?>"><?php echo esc_html( $state['cta'] ); ?> ↗</a>
</main>
<?php get_footer(); ?>
