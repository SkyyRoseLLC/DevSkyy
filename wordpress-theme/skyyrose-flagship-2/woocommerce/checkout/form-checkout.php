<?php
/** V2 checkout shell. WooCommerce retains all transaction authority. @package SkyyRoseFlagship2 */
defined( 'ABSPATH' ) || exit;
if ( ! WC()->cart ) { return; }
$checkout = WC()->checkout();
do_action( 'woocommerce_before_checkout_form', $checkout );
if ( ! $checkout->is_registration_enabled() && $checkout->is_registration_required() && ! is_user_logged_in() ) { echo esc_html( apply_filters( 'woocommerce_checkout_must_be_logged_in_message', __( 'You must be logged in to checkout.', 'skyyrose-flagship-2' ) ) ); return; }
?>
<section class="sr2-checkout sr2-commerce-shell" data-sr2-route="checkout"><header class="sr2-page-head"><p class="sr2-c-kicker"><?php esc_html_e( 'Secure checkout', 'skyyrose-flagship-2' ); ?></p><h1><?php esc_html_e( 'Your pieces are almost home.', 'skyyrose-flagship-2' ); ?></h1></header><form name="checkout" method="post" class="checkout woocommerce-checkout sr2-c-checkout" action="<?php echo esc_url( wc_get_checkout_url() ); ?>" enctype="multipart/form-data"><div class="sr2-c-checkout__grid"><div><h2><?php esc_html_e( 'Your details', 'skyyrose-flagship-2' ); ?></h2><?php do_action( 'woocommerce_checkout_before_customer_details' ); ?><div id="customer_details"><div class="col-1"><?php do_action( 'woocommerce_checkout_billing' ); ?></div><div class="col-2"><?php do_action( 'woocommerce_checkout_shipping' ); ?></div></div><?php do_action( 'woocommerce_checkout_after_customer_details' ); ?></div><aside><h2 id="order_review_heading"><?php esc_html_e( 'Order summary', 'skyyrose-flagship-2' ); ?></h2><?php do_action( 'woocommerce_checkout_before_order_review' ); ?><div id="order_review" class="woocommerce-checkout-review-order"><?php do_action( 'woocommerce_checkout_order_review' ); ?></div><?php do_action( 'woocommerce_checkout_after_order_review' ); ?></aside></div></form></section>
<?php do_action( 'woocommerce_after_checkout_form', $checkout ); ?>
