<?php
/**
 * Marketplace presentation registry.
 *
 * This registry owns install-time routes, editable starter copy, navigation,
 * and WooCommerce page shells. It never owns products, prices, stock, orders,
 * media, or payment state.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

/**
 * Convert titled sections into portable block markup.
 *
 * @param array<int,array{title:string,body:string}> $sections Content sections.
 * @return string
 */
function skyyrose2_marketplace_block_sections( $sections ) {
	$content = '';
	foreach ( $sections as $section ) {
		$content .= '<!-- wp:heading --><h2 class="wp-block-heading">' . esc_html( $section['title'] ) . '</h2><!-- /wp:heading -->';
		$content .= '<!-- wp:paragraph --><p>' . esc_html( $section['body'] ) . '</p><!-- /wp:paragraph -->';
	}
	return $content;
}

/**
 * Editable policy and service starter copy.
 *
 * The importer never overwrites existing merchant content. Marketplace buyers
 * must review these operational defaults for their jurisdiction and policies.
 *
 * @return array<string,string>
 */
function skyyrose2_marketplace_service_content() {
	return array(
		'faq'              => skyyrose2_marketplace_block_sections(
			array(
				array( 'title' => __( 'How do I choose a size?', 'skyyrose-flagship-2' ), 'body' => __( 'Use the measurements on the size guide and compare them with a piece you already own. Product-specific fit notes remain on the product page.', 'skyyrose-flagship-2' ) ),
				array( 'title' => __( 'When will a pre-order ship?', 'skyyrose-flagship-2' ), 'body' => __( 'The estimated fulfillment window is published before checkout and repeated in the order confirmation. Client Services shares updates if that window changes.', 'skyyrose-flagship-2' ) ),
				array( 'title' => __( 'How do I get order help?', 'skyyrose-flagship-2' ), 'body' => __( 'Send Client Services your order number and the email used at checkout. Never send payment-card details by email.', 'skyyrose-flagship-2' ) ),
			)
		),
		'shipping-returns' => skyyrose2_marketplace_block_sections(
			array(
				array( 'title' => __( 'Order processing', 'skyyrose-flagship-2' ), 'body' => __( 'In-stock orders enter processing after payment clears. Pre-order pieces follow the fulfillment window shown before purchase.', 'skyyrose-flagship-2' ) ),
				array( 'title' => __( 'Shipping updates', 'skyyrose-flagship-2' ), 'body' => __( 'Tracking is emailed when the carrier accepts the parcel. Carrier scans and delivery estimates remain the carrier’s live authority.', 'skyyrose-flagship-2' ) ),
				array( 'title' => __( 'Return requests', 'skyyrose-flagship-2' ), 'body' => __( 'Contact Client Services with the order number, item, and reason before sending anything back. Eligibility is confirmed against the policy and item condition.', 'skyyrose-flagship-2' ) ),
			)
		),
		'size-guide'       => skyyrose2_marketplace_block_sections(
			array(
				array( 'title' => __( 'Measure a piece you own', 'skyyrose-flagship-2' ), 'body' => __( 'Lay a similar garment flat without stretching it. Compare chest, body length, waist, rise, and inseam with the measurements published for the piece.', 'skyyrose-flagship-2' ) ),
				array( 'title' => __( 'Between sizes', 'skyyrose-flagship-2' ), 'body' => __( 'Choose based on the silhouette you want and the product-specific fit note. Client Services can help compare measurements before purchase.', 'skyyrose-flagship-2' ) ),
			)
		),
		'privacy-policy'   => skyyrose2_marketplace_block_sections(
			array(
				array( 'title' => __( 'Information collected', 'skyyrose-flagship-2' ), 'body' => __( 'The store processes information needed for orders, accounts, support, fraud prevention, and requested communications. This may include contact, billing, shipping, order, and device information.', 'skyyrose-flagship-2' ) ),
				array( 'title' => __( 'How information is used', 'skyyrose-flagship-2' ), 'body' => __( 'Information is used to operate the storefront, fulfill orders, provide support, protect customers, and improve service. Marketing is sent only where permission or applicable law allows it.', 'skyyrose-flagship-2' ) ),
				array( 'title' => __( 'Your choices', 'skyyrose-flagship-2' ), 'body' => __( 'Customers may request access, correction, or deletion where applicable, and may unsubscribe from marketing at any time.', 'skyyrose-flagship-2' ) ),
			)
		),
		'terms-of-service' => skyyrose2_marketplace_block_sections(
			array(
				array( 'title' => __( 'Store terms', 'skyyrose-flagship-2' ), 'body' => __( 'By using the storefront or placing an order, customers agree to provide current information and use the store lawfully.', 'skyyrose-flagship-2' ) ),
				array( 'title' => __( 'Orders and pricing', 'skyyrose-flagship-2' ), 'body' => __( 'Orders remain subject to acceptance, payment confirmation, and availability. Pricing or description errors may be corrected before fulfillment, with notice and an appropriate refund where required.', 'skyyrose-flagship-2' ) ),
				array( 'title' => __( 'Intellectual property', 'skyyrose-flagship-2' ), 'body' => __( 'Brand names, artwork, photography, product designs, and site content remain the property of their respective owners and licensors.', 'skyyrose-flagship-2' ) ),
			)
		),
		'accessibility'    => skyyrose2_marketplace_block_sections(
			array(
				array( 'title' => __( 'Our commitment', 'skyyrose-flagship-2' ), 'body' => __( 'SkyyRose works toward an inclusive shopping experience aligned with WCAG 2.2 Level AA, including keyboard access, readable contrast, meaningful structure, and reduced-motion support.', 'skyyrose-flagship-2' ) ),
				array( 'title' => __( 'Accessibility feedback', 'skyyrose-flagship-2' ), 'body' => __( 'If you encounter a barrier, contact Client Services with the page, task, device, browser, and assistive technology involved so an accessible alternative can be provided.', 'skyyrose-flagship-2' ) ),
			)
		),
	);
}

/**
 * Theme-owned page definitions for the one-click importer.
 *
 * @return array<string,array<string,mixed>>
 */
function skyyrose2_marketplace_pages() {
	$service = skyyrose2_marketplace_service_content();
	return array(
		'home'             => array( 'title' => __( 'Home', 'skyyrose-flagship-2' ), 'path' => 'home', 'template' => 'default', 'content' => '' ),
		'collections'      => array( 'title' => __( 'Collections', 'skyyrose-flagship-2' ), 'path' => 'collections', 'template' => 'default', 'content' => '' ),
		'signature'        => array( 'title' => __( 'Signature', 'skyyrose-flagship-2' ), 'path' => 'collections/signature', 'parent' => 'collections', 'template' => 'template-collection.php', 'content' => '' ),
		'black-rose'       => array( 'title' => __( 'Black Rose', 'skyyrose-flagship-2' ), 'path' => 'collections/black-rose', 'parent' => 'collections', 'template' => 'template-collection.php', 'content' => '' ),
		'love-hurts'       => array( 'title' => __( 'Love Hurts', 'skyyrose-flagship-2' ), 'path' => 'collections/love-hurts', 'parent' => 'collections', 'template' => 'template-collection.php', 'content' => '' ),
		'kids-capsule'     => array( 'title' => __( 'Kids Capsule', 'skyyrose-flagship-2' ), 'path' => 'collections/kids-capsule', 'parent' => 'collections', 'template' => 'template-collection.php', 'content' => '' ),
		'immersive-signature' => array( 'title' => __( 'Signature — Full Scene', 'skyyrose-flagship-2' ), 'path' => 'worlds/signature', 'template' => 'template-immersive-signature.php', 'content' => '' ),
		'immersive-black-rose' => array( 'title' => __( 'Black Rose — Full Scene', 'skyyrose-flagship-2' ), 'path' => 'worlds/black-rose', 'template' => 'template-immersive-black-rose.php', 'content' => '' ),
		'immersive-love-hurts' => array( 'title' => __( 'Love Hurts — Full Scene', 'skyyrose-flagship-2' ), 'path' => 'worlds/love-hurts', 'template' => 'template-immersive-love-hurts.php', 'content' => '' ),
		'immersive-kids-capsule' => array( 'title' => __( 'Kids Capsule — Full Scene', 'skyyrose-flagship-2' ), 'path' => 'worlds/kids-capsule', 'template' => 'template-immersive-kids-capsule.php', 'content' => '' ),
		'pre-order'        => array( 'title' => __( 'Pre-Order', 'skyyrose-flagship-2' ), 'path' => 'pre-order', 'template' => 'default', 'content' => '' ),
		'about'            => array( 'title' => __( 'About', 'skyyrose-flagship-2' ), 'path' => 'about', 'template' => 'default', 'content' => '' ),
		'contact'          => array( 'title' => __( 'Contact', 'skyyrose-flagship-2' ), 'path' => 'contact', 'template' => 'default', 'content' => '' ),
		'journal'          => array( 'title' => __( 'Journal', 'skyyrose-flagship-2' ), 'path' => 'journal', 'template' => 'default', 'content' => '' ),
		'wishlist'         => array( 'title' => __( 'Wishlist', 'skyyrose-flagship-2' ), 'path' => 'wishlist', 'template' => 'page-wishlist.php', 'content' => '' ),
		'faq'              => array( 'title' => __( 'Frequently Asked Questions', 'skyyrose-flagship-2' ), 'path' => 'faq', 'template' => 'default', 'content' => $service['faq'] ),
		'shipping-returns' => array( 'title' => __( 'Shipping + Returns', 'skyyrose-flagship-2' ), 'path' => 'shipping-returns', 'template' => 'default', 'content' => $service['shipping-returns'] ),
		'size-guide'       => array( 'title' => __( 'Size Guide', 'skyyrose-flagship-2' ), 'path' => 'size-guide', 'template' => 'default', 'content' => $service['size-guide'] ),
		'privacy-policy'   => array( 'title' => __( 'Privacy Policy', 'skyyrose-flagship-2' ), 'path' => 'privacy-policy', 'template' => 'default', 'content' => $service['privacy-policy'] ),
		'terms-of-service' => array( 'title' => __( 'Terms of Service', 'skyyrose-flagship-2' ), 'path' => 'terms-of-service', 'template' => 'default', 'content' => $service['terms-of-service'] ),
		'accessibility'    => array( 'title' => __( 'Accessibility Statement', 'skyyrose-flagship-2' ), 'path' => 'accessibility', 'template' => 'default', 'content' => $service['accessibility'] ),
	);
}

/**
 * WooCommerce classic-page shells. WooCommerce remains the state authority.
 *
 * @return array<string,array<string,string>>
 */
function skyyrose2_marketplace_woocommerce_pages() {
	return array(
		'shop'       => array( 'title' => __( 'Shop', 'skyyrose-flagship-2' ), 'content' => '', 'option' => 'woocommerce_shop_page_id' ),
		'cart'       => array( 'title' => __( 'Bag', 'skyyrose-flagship-2' ), 'content' => '<!-- wp:shortcode -->[woocommerce_cart]<!-- /wp:shortcode -->', 'option' => 'woocommerce_cart_page_id' ),
		'checkout'   => array( 'title' => __( 'Checkout', 'skyyrose-flagship-2' ), 'content' => '<!-- wp:shortcode -->[woocommerce_checkout]<!-- /wp:shortcode -->', 'option' => 'woocommerce_checkout_page_id' ),
		'my-account' => array( 'title' => __( 'My Account', 'skyyrose-flagship-2' ), 'content' => '<!-- wp:shortcode -->[woocommerce_my_account]<!-- /wp:shortcode -->', 'option' => 'woocommerce_myaccount_page_id' ),
		'track-order' => array( 'title' => __( 'Track Order', 'skyyrose-flagship-2' ), 'content' => '<!-- wp:shortcode -->[woocommerce_order_tracking]<!-- /wp:shortcode -->', 'option' => '' ),
	);
}

/**
 * Navigation definitions consumed by the importer.
 *
 * @return array<string,array<string,mixed>>
 */
function skyyrose2_marketplace_menus() {
	return array(
		'primary' => array(
			'name'  => 'SkyyRose Primary',
			'items' => array(
				array( 'title' => __( 'Collections', 'skyyrose-flagship-2' ), 'page' => 'collections', 'children' => array( 'signature', 'black-rose', 'love-hurts', 'kids-capsule' ) ),
				array( 'title' => __( 'Shop', 'skyyrose-flagship-2' ), 'page' => 'shop' ),
				array( 'title' => __( 'Pre-Order', 'skyyrose-flagship-2' ), 'page' => 'pre-order' ),
				array( 'title' => __( 'Journal', 'skyyrose-flagship-2' ), 'page' => 'journal' ),
				array( 'title' => __( 'About', 'skyyrose-flagship-2' ), 'page' => 'about' ),
				array( 'title' => __( 'Contact', 'skyyrose-flagship-2' ), 'page' => 'contact' ),
			),
		),
		'footer'  => array(
			'name'  => 'SkyyRose Footer',
			'items' => array(
				array( 'title' => __( 'Shipping + Returns', 'skyyrose-flagship-2' ), 'page' => 'shipping-returns' ),
				array( 'title' => __( 'Size Guide', 'skyyrose-flagship-2' ), 'page' => 'size-guide' ),
				array( 'title' => __( 'FAQ', 'skyyrose-flagship-2' ), 'page' => 'faq' ),
				array( 'title' => __( 'Privacy Policy', 'skyyrose-flagship-2' ), 'page' => 'privacy-policy' ),
				array( 'title' => __( 'Terms of Service', 'skyyrose-flagship-2' ), 'page' => 'terms-of-service' ),
				array( 'title' => __( 'Accessibility', 'skyyrose-flagship-2' ), 'page' => 'accessibility' ),
			),
		),
	);
}
