<?php
defined( 'ABSPATH' ) || exit;
/** Read-only concierge destinations and product discovery; Woo owns all sale facts. */
function skyyrose2_concierge_guide() {
	$contact = skyyrose2_marketplace_page_url( 'contact' );
	$shop = skyyrose2_shop_url();
	$guide = array(
		'greeting' => __( 'Welcome to SkyyRose. I’m Skyy, your digital house guide. This house is a father’s promise to his daughter, rooted in Oakland. I can help you explore a collection or find a piece by name or SKU.', 'skyyrose-flagship-2' ),
		'pages' => array(
			'shop' => array( 'label' => __( 'Shop the house', 'skyyrose-flagship-2' ), 'url' => $shop ),
			'contact' => array( 'label' => __( 'Contact the house', 'skyyrose-flagship-2' ), 'url' => $contact ),
		),
		'intents' => array(
			array( 'id' => 'sizing', 'patterns' => array( 'size', 'sizing', 'fit', 'measurements' ), 'answer' => __( 'Open the piece you’re considering to see its current size options and product details. If you need fit advice, contact the house with the product name or SKU.', 'skyyrose-flagship-2' ), 'link' => $contact, 'label' => __( 'Ask about fit', 'skyyrose-flagship-2' ) ),
			array( 'id' => 'shipping', 'patterns' => array( 'shipping', 'delivery', 'returns', 'order' ), 'answer' => __( 'Shipping and purchase details depend on your order. Please contact the house for help with delivery, returns or an existing order.', 'skyyrose-flagship-2' ), 'link' => $contact, 'label' => __( 'Contact the house', 'skyyrose-flagship-2' ) ),
			array( 'id' => 'legacy', 'patterns' => array( 'skyy', 'daughter', 'father', 'heir', 'story' ), 'answer' => __( 'SkyyRose is named after Skyy Rose, the founder’s daughter. The Heir carries that story forward: family, inheritance and a future built with care.', 'skyyrose-flagship-2' ), 'link' => skyyrose2_collection_url( 'kids-capsule' ), 'label' => __( 'Explore The Heir', 'skyyrose-flagship-2' ) ),
		),
		'products' => array(),
		'suggestions' => array( 'signature', 'black-rose', 'love-hurts', 'legacy', 'sizing' ),
	);
	foreach ( skyyrose2_collections() as $slug => $collection ) {
		$guide['intents'][] = array( 'id' => $slug, 'patterns' => array( $collection['name'], str_replace( '-', ' ', $slug ) ), 'answer' => $collection['headline'], 'link' => skyyrose2_collection_url( $slug ), 'label' => sprintf( __( 'Explore %s', 'skyyrose-flagship-2' ), $collection['name'] ) );
	}
	if ( function_exists( 'wc_get_products' ) ) {
		foreach ( wc_get_products( array( 'status' => 'publish', 'limit' => 100, 'orderby' => 'title', 'order' => 'ASC' ) ) as $item ) {
			if ( ! $item->is_visible() || post_password_required( $item->get_id() ) ) { continue; }
			$record = skyyrose2_product_presentation( $item );
			$guide['products'][] = array( 'name' => wp_strip_all_tags( $item->get_name() ), 'sku' => $item->get_sku(), 'url' => $item->get_permalink(), 'collection' => str_replace( '-', ' ', $record['collection'] ?? '' ) );
		}
	}
	return $guide;
}

