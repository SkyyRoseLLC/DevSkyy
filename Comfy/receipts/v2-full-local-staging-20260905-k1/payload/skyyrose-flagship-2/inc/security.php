<?php
/**
 * Front-end security headers for the V2 storefront.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

/**
 * Send the storefront CSP after platform and plugin defaults are registered.
 *
 * Three.js creates blob-backed textures and workers while decoding the mascot
 * GLB. Those URLs never leave the current browsing context, but they must be
 * explicitly allowed or the model renders without its verified textures.
 *
 * @return void
 */
function skyyrose2_security_headers() {
	if ( is_admin() && ! wp_doing_ajax() ) {
		return;
	}

	$directives = array(
		"default-src 'self'",
		"script-src 'self' 'unsafe-inline' 'wasm-unsafe-eval' blob: https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://cdn.babylonjs.com https://stats.wp.com https://widgets.wp.com https://s0.wp.com https://cdn.elementor.com https://ajax.googleapis.com https://unpkg.com https://connect.facebook.net https://js.stripe.com https://www.paypal.com https://www.paypalobjects.com",
		"style-src 'self' 'unsafe-inline' https://fonts.googleapis.com https://fonts-api.wp.com https://fonts.wp.com https://cdn.jsdelivr.net https://cdnjs.cloudflare.com https://s0.wp.com https://widgets.wp.com https://cdn.elementor.com",
		"font-src 'self' data: https://fonts.gstatic.com https://fonts-api.wp.com https://fonts.wp.com https://cdn.jsdelivr.net",
		"img-src 'self' data: blob: https:",
		"connect-src 'self' blob: https://cdn.jsdelivr.net https://stats.wp.com https://pixel.wp.com https://public-api.wordpress.com https://api.skyyrose.co https://devskyy.app https://www.facebook.com https://connect.facebook.net https://api.stripe.com https://js.stripe.com https://www.paypal.com https://www.paypalobjects.com",
		"frame-src 'self' https://www.youtube.com https://www.youtube-nocookie.com https://player.vimeo.com https://widgets.wp.com https://jetpack.wordpress.com https://wordpress.com https://js.stripe.com https://hooks.stripe.com https://www.paypal.com",
		"worker-src 'self' blob:",
		"child-src 'self' blob:",
		"object-src 'none'",
		"base-uri 'self'",
		"form-action 'self' https://www.paypal.com",
	);

	header( 'Content-Security-Policy: ' . implode( '; ', $directives ), true );
	header( 'X-Content-Type-Options: nosniff', true );
	header( 'Referrer-Policy: strict-origin-when-cross-origin', true );
}
add_action( 'send_headers', 'skyyrose2_security_headers', 99 );
