<?php
/** Same-source PDP delivery; never grants media authority or changes attachments. */
defined( 'ABSPATH' ) || exit;

/** Limit the native AJAX presentation boundary to Woo's variation endpoint. */
function skyyrose2_pdp_gallery_request() {
	if ( function_exists( 'is_product' ) && is_product() ) {
		return true;
	}
	return function_exists( 'wp_doing_ajax' ) && wp_doing_ajax() && (
		'get_variation' === ( $_REQUEST['wc-ajax'] ?? '' ) ||
		'woocommerce_get_variation' === ( $_REQUEST['action'] ?? '' )
	);
}

/** Read the stable permission snapshot only for its owning parent product. */
function skyyrose2_pdp_media_context( $product ) {
	$context = $GLOBALS['skyyrose2_pdp_media_context'] ?? null;
	return is_a( $product, 'WC_Product' ) && is_array( $context ) && $context['product_id'] === $product->get_id() ? $context : null;
}

/** Capture before Woo's temporary gallery getters run; callers restore in finally. */
function skyyrose2_pdp_capture_media_context( $product ) {
	return array(
		'product_id' => $product->get_id(),
		'product' => $product,
		'media' => skyyrose2_product_commerce_media( $product ),
		'original_ids' => array_values( array_unique( array_filter( array_map( 'intval', array_merge( array( $product->get_image_id() ), $product->get_gallery_image_ids() ) ) ) ) ),
	);
}

/** Display-only attributes shared by every native gallery representation. */
function skyyrose2_pdp_gallery_attributes( $attributes, $attachment_id, $main_image, $product ) {
	$context = skyyrose2_pdp_media_context( $product );
	if ( ! $context || ! in_array( (int) $attachment_id, array_map( 'intval', $context['media']['ids'] ), true ) ) {
		return $attributes;
	}
	$attributes['loading'] = $main_image ? 'eager' : 'lazy';
	$attributes['fetchpriority'] = $main_image ? 'high' : 'auto';
	$attributes['decoding'] = 'async';
	$delivery = skyyrose2_pdp_media_delivery( $product, (int) $attachment_id );
	if ( $delivery ) {
		foreach ( array( 'src', 'srcset', 'sizes' ) as $key ) {
			$attributes[ $key ] = $delivery[ $key ];
		}
	}
	return $attributes;
}

/** Hash a readable local file once per request. Remote wrappers are never read. */
function skyyrose2_pdp_delivery_file_hash( $path ) {
	static $hashes = array();
	if ( ! is_string( $path ) || false !== strpos( $path, '://' ) ) {
		return '';
	}
	$local = realpath( $path );
	if ( ! $local || ! is_file( $local ) || ! is_readable( $local ) ) {
		return '';
	}
	if ( ! isset( $hashes[ $local ] ) ) {
		$hashes[ $local ] = hash_file( 'sha256', $local ) ?: '';
	}
	return $hashes[ $local ];
}

/**
 * Return responsive display attributes only after native PDP permission and byte proof.
 *
 * The accepted front's sha256 identifies the rendition source. Its source_sha256
 * identifies an earlier production input and is deliberately not interchangeable.
 * The caller retains native attachment IDs, alt text, full/lightbox and thumbnails.
 */
function skyyrose2_pdp_media_delivery( $product, $attachment_id ) {
	if ( ! is_a( $product, 'WC_Product' ) || ! $attachment_id ) {
		return array();
	}
	$context = skyyrose2_pdp_media_context( $product );
	$media = $context ? $context['media'] : skyyrose2_product_commerce_media( $product );
	if ( ! in_array( $media['state'] ?? '', array( 'commerce', 'editorial' ), true ) || ! in_array( (int) $attachment_id, array_map( 'intval', $media['ids'] ?? array() ), true ) ) {
		return array();
	}
	static $accepted = null;
	static $derived = null;
	if ( null === $accepted ) {
		$accepted_path = SKYYROSE2_DIR . '/data/approved-card-fronts.json';
		$derived_path  = SKYYROSE2_DIR . '/assets/derived/card-fronts/manifest.json';
		$accepted = is_readable( $accepted_path ) ? json_decode( file_get_contents( $accepted_path ), true ) : array(); // phpcs:ignore WordPress.WP.AlternativeFunctions.file_get_contents_file_get_contents
		$derived  = is_readable( $derived_path ) ? json_decode( file_get_contents( $derived_path ), true ) : array(); // phpcs:ignore WordPress.WP.AlternativeFunctions.file_get_contents_file_get_contents
	}
	$sku = strtolower( trim( (string) $product->get_sku() ) );
	$front = $accepted['products'][ $sku ] ?? array();
	$record = $derived['products'][ $sku ] ?? array();
	if ( ! is_array( $front ) || ! is_array( $record ) ) {
		return array();
	}
	$hash = $front['sha256'] ?? '';
	if ( 1 !== ( $accepted['schema_version'] ?? null ) || 'skyyrose.card-renditions.v1' !== ( $derived['schema'] ?? '' ) || 'FOUNDER_APPROVED_V2_CARD' !== ( $front['scene_status'] ?? '' ) || ! is_string( $hash ) || ! preg_match( '/^[a-f0-9]{64}$/D', $hash ) || ! preg_match( '/^[a-z0-9-]+$/D', $sku ) || ! is_array( $record['renditions'] ?? null ) || ( $record['source_sha256'] ?? '' ) !== $hash || ( $record['source'] ?? '' ) !== ( $front['src'] ?? '' ) ) {
		return array();
	}
	// Unfiltered local attachment metadata avoids trusting an offload URL as proof.
	if ( skyyrose2_pdp_delivery_file_hash( get_attached_file( $attachment_id, true ) ) !== $hash ) {
		return array();
	}
	$width = (int) ( $front['width'] ?? 0 );
	$height = (int) ( $front['height'] ?? 0 );
	if ( $width < 1 || $height < 1 ) {
		return array();
	}
	$srcset = array();
	$result = array();
	$root = realpath( SKYYROSE2_DIR . '/assets/derived/card-fronts' );
	foreach ( $record['renditions'] as $rendition ) {
		if ( ! is_array( $rendition ) ) {
			continue;
		}
		$rw = (int) ( $rendition['width'] ?? 0 );
		$rh = (int) ( $rendition['height'] ?? 0 );
		$rhash = $rendition['sha256'] ?? '';
		if ( ! is_string( $rhash ) || ! preg_match( '/^[a-f0-9]{64}$/D', $rhash ) ) {
			continue;
		}
		$relative = 'assets/derived/card-fronts/' . $sku . '-' . $rw . 'w.webp';
		$file = SKYYROSE2_DIR . '/' . $relative;
		$local = realpath( $file );
		if ( ! in_array( $rw, array( 320, 480, 768 ), true ) || $rh !== (int) round( $height * $rw / $width ) || ( $rendition['src'] ?? '' ) !== $relative || ! $root || ! $local || 0 !== strpos( $local, $root . DIRECTORY_SEPARATOR ) || is_link( $file ) || skyyrose2_pdp_delivery_file_hash( $file ) !== ( $rendition['sha256'] ?? '' ) ) {
			continue;
		}
		$dimensions = getimagesize( $local );
		if ( ! $dimensions || $rw !== $dimensions[0] || $rh !== $dimensions[1] ) {
			continue;
		}
		$url = SKYYROSE2_URI . '/' . $relative;
		$srcset[ $rw ] = $url . ' ' . $rw . 'w';
		if ( 480 === $rw ) {
			$result = array( 'src' => $url, 'width' => $rw, 'height' => $rh );
		}
	}
	if ( ! $result || ! $srcset ) {
		return array();
	}
	ksort( $srcset );
	$result['srcset'] = implode( ', ', $srcset );
	// Object-fit:contain paints within the existing bounded gallery height. Size
	// selection follows those pixels rather than the wider empty gallery box.
	$ratio = $width / $height;
	$result['sizes'] = sprintf( '(max-width: 47.99rem) clamp(%dpx, %.4fsvh, %dpx), clamp(%dpx, %.4fvw, %dpx)', (int) ceil( 208 * $ratio ), 30 * $ratio, (int) ceil( 272 * $ratio ), (int) ceil( 448 * $ratio ), 50 * $ratio, (int) ceil( 768 * $ratio ) );
	return $result;
}

/** Native variation JSON keeps original IDs, thumbnail and full/lightbox sources. */
function skyyrose2_pdp_variation_delivery( $data, $product, $variation ) {
	if ( ! is_a( $product, 'WC_Product' ) || ! is_a( $variation, 'WC_Product_Variation' ) || (int) $variation->get_parent_id() !== $product->get_id() || ( ! skyyrose2_pdp_media_context( $product ) && ! skyyrose2_pdp_gallery_request() ) ) {
		return $data;
	}
	$previous = $GLOBALS['skyyrose2_pdp_media_context'] ?? null;
	// Native get_available_variation has restored its getter overrides before
	// this filter runs, including the native AJAX get_variation path.
	$GLOBALS['skyyrose2_pdp_media_context'] = skyyrose2_pdp_media_context( $product ) ?: skyyrose2_pdp_capture_media_context( $product );
	try {
		$context = skyyrose2_pdp_media_context( $product );
		$permitted = array_map( 'intval', $context['media']['ids'] );
		if ( ! $permitted || 'rejected' === $context['media']['state'] ) {
			$data['image'] = array();
			$data['image_id'] = 0;
			$data['gallery_image_ids'] = array();
			$data['gallery_images_html'] = '';
			return $data;
		}
		$image_id = (int) ( $data['image_id'] ?? 0 );
		if ( $image_id && ! in_array( $image_id, $permitted, true ) ) {
			$data['image'] = array();
			$data['image_id'] = 0;
		}
		$candidates = array_values( array_unique( array_map( 'intval', $data['gallery_image_ids'] ?? array() ) ) );
		if ( $candidates ) {
			$data['gallery_image_ids'] = array_values( array_intersect( $candidates, $permitted ) );
			if ( $image_id && ! in_array( $image_id, $candidates, true ) ) {
				array_unshift( $candidates, $image_id );
			}
			$data['gallery_images_html'] = wc_get_product_gallery_html( $product, array_values( array_intersect( $candidates, $permitted ) ) );
		} else {
			$data['gallery_images_html'] = '';
		}
		$delivery = ! empty( $data['image_id'] ) && ! empty( $data['image'] ) ? skyyrose2_pdp_media_delivery( $product, $image_id ) : array();
		if ( $delivery ) {
			foreach ( array( 'src', 'srcset', 'sizes' ) as $key ) {
				$data['image'][ $key ] = $delivery[ $key ];
			}
			$data['image']['src_w'] = $delivery['width'];
			$data['image']['src_h'] = $delivery['height'];
		}
		return $data;
	} finally {
		if ( null === $previous ) {
			unset( $GLOBALS['skyyrose2_pdp_media_context'] );
		} else {
			$GLOBALS['skyyrose2_pdp_media_context'] = $previous;
		}
	}
}
add_filter( 'woocommerce_available_variation', 'skyyrose2_pdp_variation_delivery', 40, 3 );

/** Full product detail remains available on demand through native PhotoSwipe. */
function skyyrose2_pdp_hover_zoom_enabled( $enabled ) {
	return function_exists( 'is_product' ) && is_product() ? false : $enabled;
}
add_filter( 'woocommerce_single_product_zoom_enabled', 'skyyrose2_pdp_hover_zoom_enabled' );
