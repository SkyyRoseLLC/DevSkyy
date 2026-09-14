<?php
/** Standalone contract tests for inc/performance.php. */

define( 'ABSPATH', __DIR__ . '/' );
define( 'SKYYROSE2_DIR', dirname( __DIR__ ) );
define( 'SKYYROSE2_URI', 'https://example.test/wp-content/themes/skyyrose-flagship-2' );

$GLOBALS['sr2_hooks']       = array();
$GLOBALS['sr2_route']       = array( 'front' => false, 'page' => false, 'single' => false, 'singular' => '' );
$GLOBALS['sr2_templates']   = array();
$GLOBALS['sr2_styles']      = array();
$GLOBALS['sr2_scripts']     = array();
$GLOBALS['sr2_strategies']  = array();
$GLOBALS['sr2_post_slug']   = '';

function add_action( $hook, $callback, $priority = 10, $accepted_args = 1 ) {
	$GLOBALS['sr2_hooks'][] = compact( 'hook', 'callback', 'priority', 'accepted_args' );
}
function add_filter( $hook, $callback, $priority = 10, $accepted_args = 1 ) {
	$GLOBALS['sr2_hooks'][] = compact( 'hook', 'callback', 'priority', 'accepted_args' );
}
function remove_action() {}
function remove_filter() {}
function is_front_page() { return $GLOBALS['sr2_route']['front']; }
function is_page() { return $GLOBALS['sr2_route']['page']; }
function is_single() { return $GLOBALS['sr2_route']['single']; }
function is_singular( $type = '' ) { return $type ? $GLOBALS['sr2_route']['singular'] === $type : (bool) $GLOBALS['sr2_route']['singular']; }
function is_page_template( $templates ) { return (bool) array_intersect( (array) $templates, $GLOBALS['sr2_templates'] ); }
function is_user_logged_in() { return false; }
function wp_dequeue_style( $handle ) { $GLOBALS['sr2_styles'][] = $handle; }
function wp_dequeue_script( $handle ) { $GLOBALS['sr2_scripts'][] = $handle; }
function wp_script_add_data( $handle, $key, $value ) { $GLOBALS['sr2_strategies'][ $handle ][ $key ] = $value; }
function wp_parse_url( $url, $component = -1 ) { return parse_url( $url, $component ); }
function sanitize_title( $value ) { return strtolower( trim( preg_replace( '/[^a-z0-9]+/i', '-', $value ), '-' ) ); }
function get_post_field() { return $GLOBALS['sr2_post_slug']; }
function get_queried_object_id() { return 123; }
function wp_get_attachment_metadata() { return array( 'width' => 1200, 'height' => 1500 ); }
function skyyrose2_sot_asset_uri( $path ) { return SKYYROSE2_URI . '/assets/sot/' . ltrim( $path, '/' ); }
function skyyrose2_collections() {
	return array(
		'signature' => array(
			'hero'        => 'images/hero/responsive/signature-golden-gate-monuments-v2-1440w.webp',
			'hero_tablet' => 'images/hero/responsive/signature-golden-gate-monuments-v2-1024w.webp',
			'hero_mobile' => 'images/hero/responsive/signature-golden-gate-monuments-v2-640w.webp',
		),
	);
}

require dirname( __DIR__ ) . '/inc/performance.php';

function sr2_assert( $condition, $message ) {
	if ( ! $condition ) {
		fwrite( STDERR, "FAIL: {$message}\n" );
		exit( 1 );
	}
}

$registered = array_column( $GLOBALS['sr2_hooks'], 'callback' );
sr2_assert( in_array( 'skyyrose2_performance_preload_resources', $registered, true ), 'preload filter is registered' );
sr2_assert( in_array( 'skyyrose2_performance_defer_scripts', $registered, true ), 'defer policy is registered' );

$GLOBALS['sr2_route']['front'] = true;
$front                         = skyyrose2_performance_route_preloads();
sr2_assert( 3 === count( $front ), 'front page has three art-directed records' );
sr2_assert( 3 === count( array_unique( array_column( $front, 'media' ) ) ), 'front page media queries are mutually distinct' );
sr2_assert(
	3 === count(
		array_filter(
			$front,
			static function ( $item ) {
				return 'high' === $item['fetchpriority'];
			}
		)
	),
	'each active hero candidate is high priority'
);
foreach ( $front as $resource ) {
	$local = str_replace( SKYYROSE2_URI, SKYYROSE2_DIR, $resource['href'] );
	sr2_assert( is_readable( $local ), 'every front-page preload exists locally' );
}

$deduplicated = skyyrose2_performance_preload_resources( array( $front[0] ) );
sr2_assert( 3 === count( $deduplicated ), 'an existing hero href is not duplicated' );

$GLOBALS['sr2_route']['front'] = false;
$GLOBALS['sr2_templates']      = array( 'template-collection.php' );
$GLOBALS['sr2_post_slug']      = 'signature';
$collection                    = skyyrose2_performance_route_preloads();
sr2_assert( 3 === count( $collection ), 'collection route mirrors responsive hero art direction' );

$attachment = (object) array( 'ID' => 9 );
$attributes = skyyrose2_performance_image_attributes(
	array( 'fetchpriority' => 'high', 'loading' => 'lazy', 'srcset' => 'one.webp 1200w' ),
	$attachment
);
sr2_assert( ! isset( $attributes['loading'] ), 'high-priority image is not lazy' );
sr2_assert( 1200 === $attributes['width'] && 1500 === $attributes['height'], 'attachment dimensions are restored' );
sr2_assert( isset( $attributes['sizes'] ), 'responsive attachment receives sizes fallback' );

$GLOBALS['sr2_styles']  = array();
$GLOBALS['sr2_scripts'] = array();
skyyrose2_performance_dequeue_unused_assets();
sr2_assert( in_array( 'wp-block-library', $GLOBALS['sr2_styles'], true ), 'governed route removes block CSS' );
sr2_assert( in_array( 'wp-embed', $GLOBALS['sr2_scripts'], true ), 'governed route removes wp-embed' );

$GLOBALS['sr2_templates']      = array();
$GLOBALS['sr2_route']['page'] = true;
$GLOBALS['sr2_styles']         = array();
$GLOBALS['sr2_scripts']        = array();
skyyrose2_performance_dequeue_unused_assets();
sr2_assert( ! in_array( 'wp-block-library', $GLOBALS['sr2_styles'], true ), 'ordinary content retains block CSS' );
sr2_assert( ! in_array( 'wp-embed', $GLOBALS['sr2_scripts'], true ), 'ordinary content retains wp-embed' );

skyyrose2_performance_defer_scripts();
sr2_assert( 'defer' === $GLOBALS['sr2_strategies']['skyyrose2-theme']['strategy'], 'theme runtime is deferred' );
sr2_assert( 'defer' === $GLOBALS['sr2_strategies']['skyyrose2-immersive']['strategy'], 'late immersive runtime is deferred' );
sr2_assert( ! isset( $GLOBALS['sr2_strategies']['wc-add-to-cart'] ), 'WooCommerce purchase scripts are untouched' );

fwrite( STDOUT, "PASS performance contract\n" );
