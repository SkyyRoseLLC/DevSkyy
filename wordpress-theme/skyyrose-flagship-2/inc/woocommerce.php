<?php
/**
 * WooCommerce integration and lifecycle hooks.
 *
 * Extracted from the flagship production architecture so V2 keeps commerce
 * concerns modular while retaining its SOT-first, skyyrose2-prefixed API.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

/** Keep custom Woo wrappers structurally clean. */
function skyyrose2_woocommerce_wrappers() {
	remove_action( 'woocommerce_before_main_content', 'woocommerce_output_content_wrapper', 10 );
	remove_action( 'woocommerce_after_main_content', 'woocommerce_output_content_wrapper_end', 10 );
}
add_action( 'wp', 'skyyrose2_woocommerce_wrappers' );
