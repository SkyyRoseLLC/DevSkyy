<?php
/**
 * Hash-verified product media resolution.
 *
 * Extracted from the flagship production architecture so V2 keeps commerce
 * concerns modular while retaining its SOT-first, skyyrose2-prefixed API.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

/**
 * Return WooCommerce gallery IDs that are safe to treat as garment views.
 *
 * The preview harness can attach a collection scene as a secondary fixture so
 * the editorial backdrop remains visible. It is never a product view. Live
 * WooCommerce galleries remain authoritative; this narrow path guard only
 * prevents known scene/branding buckets from entering the product reel.
 *
 * @param WC_Product $product Current WooCommerce product.
 * @return array<int,int>
 */
function skyyrose2_product_view_image_ids( $product ) {
	if ( ! $product || ! is_a( $product, 'WC_Product' ) ) {
		return array();
	}

	$ids = array();
	if ( method_exists( $product, 'get_image_id' ) ) {
		$ids[] = absint( $product->get_image_id() );
	}
	if ( method_exists( $product, 'get_gallery_image_ids' ) ) {
		$ids = array_merge( $ids, array_map( 'absint', (array) $product->get_gallery_image_ids() ) );
	}

	$ids = array_values( array_unique( array_filter( $ids ) ) );
	$primary = $ids ? array_shift( $ids ) : 0;
	$views   = $primary ? array( $primary ) : array();
	foreach ( $ids as $id ) {
		$url = function_exists( 'wp_get_attachment_url' ) ? (string) wp_get_attachment_url( $id ) : '';
		if ( ! $url || preg_match( '#/(?:scene|hero|immersive|branding|scroll-world)(?:/|[-_])#i', $url ) ) {
			continue;
		}
		$views[] = $id;
	}

	return array_slice( array_values( array_unique( $views ) ), 0, 3 );
}

/**
 * Load the approved, collection-owned opening product-media manifest.
 *
 * The manifest is deliberately separate from the presentation registry: it
 * attests to visual roles, while WooCommerce remains the authority for the
 * attachment IDs attached to a live product. A card can only use an image
 * when both sources agree.
 *
 * @return array<string,mixed>
 */
function skyyrose2_product_card_media_manifest() {
	static $manifest = null;
	if ( null !== $manifest ) {
		return $manifest;
	}

	$path = SKYYROSE2_DIR . '/data/opening-product-media.json';
	if ( ! is_readable( $path ) ) {
		$manifest = array();
		return $manifest;
	}

	$decoded  = json_decode( file_get_contents( $path ), true ); // phpcs:ignore WordPress.WP.AlternativeFunctions.file_get_contents_file_get_contents
	$manifest = is_array( $decoded ) ? $decoded : array();
	return $manifest;
}

/**
 * Return only attachment IDs that are both in the product gallery and named
 * in the approved SOT media manifest, ordered for a portal card.
 *
 * Product cards must lead with an on-model front view. Falling back to a
 * WooCommerce thumbnail, a collection scene, or a generic placeholder makes
 * a product look approved when its garment proof is not. If the on-model
 * proof cannot be reconciled, this intentionally returns an empty array.
 *
 * @param WC_Product $product Current WooCommerce product.
 * @return array<int,array{id:int,role:string}>
 */
function skyyrose2_product_verified_card_media( $product ) {
	if ( ! $product || ! is_a( $product, 'WC_Product' ) ) {
		return array();
	}

	$sku      = method_exists( $product, 'get_sku' ) ? sanitize_key( $product->get_sku() ) : '';
	$manifest = skyyrose2_product_card_media_manifest();
	$records  = isset( $manifest['products'] ) && is_array( $manifest['products'] ) ? $manifest['products'] : array();
	$record   = $sku && isset( $records[ $sku ] ) && is_array( $records[ $sku ] ) ? $records[ $sku ] : array();
	$views    = isset( $record['views'] ) && is_array( $record['views'] ) ? $record['views'] : array();
	if ( empty( $views ) || ! function_exists( 'wp_get_attachment_url' ) ) {
		return array();
	}

	$attachment_ids = array();
	if ( method_exists( $product, 'get_image_id' ) ) {
		$attachment_ids[] = absint( $product->get_image_id() );
	}
	if ( method_exists( $product, 'get_gallery_image_ids' ) ) {
		$attachment_ids = array_merge( $attachment_ids, array_map( 'absint', (array) $product->get_gallery_image_ids() ) );
	}
	$attachment_ids = array_values( array_unique( array_filter( $attachment_ids ) ) );
	if ( empty( $attachment_ids ) ) {
		return array();
	}

	$attachment_paths = array();
	foreach ( $attachment_ids as $attachment_id ) {
		$url  = (string) wp_get_attachment_url( $attachment_id );
		$path = (string) parse_url( $url, PHP_URL_PATH );
		if ( $path ) {
			$attachment_paths[ $attachment_id ] = basename( $path );
		}
	}

	$allowed_roles = array( 'on_model_front', 'on_model_back', 'mannequin_front', 'mannequin_back', 'render_3d', 'packshot' );
	$matched       = array();
	foreach ( $views as $view ) {
		$role = isset( $view['role'] ) ? strtolower( (string) $view['role'] ) : '';
		$role = preg_replace( '/[^a-z0-9_]/', '', $role );
		if ( ! in_array( $role, $allowed_roles, true ) || isset( $matched[ $role ] ) ) {
			continue;
		}

		$expected_paths = array_filter(
			array(
				isset( $view['source'] ) ? (string) $view['source'] : '',
				isset( $view['derivative'] ) ? (string) $view['derivative'] : '',
			)
		);
		$expected_files = array_map( 'basename', $expected_paths );
		foreach ( $attachment_paths as $attachment_id => $attachment_file ) {
			if ( in_array( $attachment_file, $expected_files, true ) ) {
				$matched[ $role ] = array(
					'id'   => (int) $attachment_id,
					'role' => $role,
				);
				break;
			}
		}
	}

	if ( empty( $matched['on_model_front'] ) ) {
		return array();
	}

	$ordered_roles = array( 'on_model_front', 'on_model_back', 'mannequin_front', 'mannequin_back', 'render_3d', 'packshot' );
	$ordered       = array();
	$used_ids      = array();
	foreach ( $ordered_roles as $role ) {
		if ( empty( $matched[ $role ] ) || isset( $used_ids[ $matched[ $role ]['id'] ] ) ) {
			continue;
		}
		$ordered[]                           = $matched[ $role ];
		$used_ids[ $matched[ $role ]['id'] ] = true;
	}

	return array_slice( $ordered, 0, 4 );
}
