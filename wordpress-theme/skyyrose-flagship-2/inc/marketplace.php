<?php
/**
 * Marketplace/fresh-install bootstrap.
 *
 * Include this file once from functions.php after the theme constants are
 * defined. Keeping the integration point singular prevents isolated worktrees
 * from reconnecting partial modules in a different order.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

require_once __DIR__ . '/presentation-registry.php';
require_once __DIR__ . '/starter-content.php';
require_once __DIR__ . '/demo-import.php';

/**
 * Register editor and feed support needed by a marketplace install.
 *
 * @return void
 */
function skyyrose2_marketplace_theme_support() {
	add_theme_support( 'editor-styles' );
	add_editor_style( 'editor-style.css' );
	add_theme_support( 'automatic-feed-links' );
	add_theme_support( 'customize-selective-refresh-widgets' );
}
add_action( 'after_setup_theme', 'skyyrose2_marketplace_theme_support', 25 );
