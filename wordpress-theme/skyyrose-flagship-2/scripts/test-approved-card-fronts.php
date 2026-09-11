<?php
/** Standalone approved-front routing regression tests. */
define( 'ABSPATH', __DIR__ );
$fixture = sys_get_temp_dir() . '/sr2-card-front-' . bin2hex( random_bytes( 6 ) );
mkdir( $fixture . '/data', 0777, true );
mkdir( $fixture . '/assets/cards', 0777, true );
define( 'SKYYROSE2_DIR', $fixture );
define( 'SKYYROSE2_URI', 'https://example.test/theme-v2' );
class WC_Product {
	private $sku;
	public function __construct( $sku ) { $this->sku = $sku; }
	public function get_sku() { return $this->sku; }
}
function skyyrose2_front_assert( $condition, $message ) {
	if ( ! $condition ) { throw new RuntimeException( $message ); }
}
$front = array( 'src' => 'assets/cards/front.webp', 'width' => 1086, 'height' => 1448, 'alt' => 'Front view' );
file_put_contents( $fixture . '/assets/cards/front.webp', 'fixture' );
file_put_contents( $fixture . '/data/approved-card-fronts.json', json_encode( array( 'schema_version' => 1, 'products' => array(
	'br-003' => $front,
	'missing' => array_merge( $front, array( 'src' => 'assets/cards/missing.webp' ) ),
	'traversal' => array_merge( $front, array( 'src' => 'assets/../data/front.webp' ) ),
	'remote' => array_merge( $front, array( 'src' => 'https://example.test/front.webp' ) ),
	'zero' => array_merge( $front, array( 'width' => 0 ) ),
) ) ) );
require dirname( __DIR__ ) . '/inc/approved-card-fronts.php';
try {
	$resolved = skyyrose2_approved_card_front( new WC_Product( 'BR-003' ) );
	skyyrose2_front_assert( SKYYROSE2_URI . '/assets/cards/front.webp' === $resolved['src'], 'Exact SKU should resolve to V2 asset URL.' );
	foreach ( array( 'unknown', 'missing', 'traversal', 'remote', 'zero' ) as $sku ) {
		skyyrose2_front_assert( array() === skyyrose2_approved_card_front( new WC_Product( $sku ) ), 'Invalid record must fall back: ' . $sku );
	}
	skyyrose2_front_assert( array() === skyyrose2_approved_card_front( null ), 'Invalid product must fall back.' );
	$views = skyyrose2_card_front_reel_views( array( 10, 11, 12 ), 10, $resolved );
	skyyrose2_front_assert( $views === array( array( 'front' => $resolved ), array( 'id' => 11 ), array( 'id' => 12 ) ), 'Override must replace primary while retaining remaining views.' );
	skyyrose2_front_assert( skyyrose2_card_front_reel_views( array( 11, 12 ), 0, $resolved ) === $views, 'Gallery-only product must prepend approved front without dropping gallery.' );
	skyyrose2_front_assert( skyyrose2_card_front_reel_views( array( 10, 11 ), 10, array() ) === array( array( 'id' => 10 ), array( 'id' => 11 ) ), 'Unknown SKU must preserve WooCommerce view order.' );
	echo "Approved card front routing tests passed.\n";
} finally {
	unlink( $fixture . '/assets/cards/front.webp' );
	unlink( $fixture . '/data/approved-card-fronts.json' );
	rmdir( $fixture . '/assets/cards' );
	rmdir( $fixture . '/assets' );
	rmdir( $fixture . '/data' );
	rmdir( $fixture );
}
