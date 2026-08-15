<?php
/**
 * WordPress starter-content bridge for fresh preview sites.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

/**
 * Register a lightweight preview set through WordPress core.
 *
 * The complete, nested commerce funnel is installed only through the explicit
 * one-click importer. Starter content remains intentionally non-destructive.
 *
 * @return void
 */
function skyyrose2_register_marketplace_starter_content() {
	add_theme_support(
		'starter-content',
		array(
			'posts'      => array(
				'home'        => array( 'post_type' => 'page', 'post_title' => __( 'Home', 'skyyrose-flagship-2' ), 'post_name' => 'home' ),
				'collections' => array( 'post_type' => 'page', 'post_title' => __( 'Collections', 'skyyrose-flagship-2' ), 'post_name' => 'collections' ),
				'pre-order'   => array( 'post_type' => 'page', 'post_title' => __( 'Pre-Order', 'skyyrose-flagship-2' ), 'post_name' => 'pre-order' ),
				'about'       => array( 'post_type' => 'page', 'post_title' => __( 'About', 'skyyrose-flagship-2' ), 'post_name' => 'about' ),
				'contact'     => array( 'post_type' => 'page', 'post_title' => __( 'Contact', 'skyyrose-flagship-2' ), 'post_name' => 'contact' ),
				'journal'     => array( 'post_type' => 'page', 'post_title' => __( 'Journal', 'skyyrose-flagship-2' ), 'post_name' => 'journal' ),
			),
			'options'    => array(
				'show_on_front'  => 'page',
				'page_on_front'   => '{{home}}',
				'page_for_posts'  => '{{journal}}',
			),
			'nav_menus'  => array(
				'primary' => array(
					'name'  => __( 'SkyyRose Primary', 'skyyrose-flagship-2' ),
					'items' => array( 'link_home', 'page_collections', 'page_pre-order', 'page_about', 'page_contact' ),
				),
			),
		)
	);
}
add_action( 'after_setup_theme', 'skyyrose2_register_marketplace_starter_content', 20 );
