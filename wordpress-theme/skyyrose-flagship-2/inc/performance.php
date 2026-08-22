<?php
/**
 * Route-aware front-end performance policy.
 *
 * This module only changes delivery hints and removes assets that are unused on
 * fully theme-owned routes. It does not change the House of Roses markup,
 * WooCommerce purchase behavior, or the no-JavaScript presentation.
 *
 * Integration contract: require this file once from the theme root.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

/** Remove legacy emoji payloads from public pages. */
function skyyrose2_performance_disable_emojis() {
	remove_action( 'wp_head', 'print_emoji_detection_script', 7 );
	remove_action( 'wp_print_styles', 'print_emoji_styles' );
	remove_action( 'wp_enqueue_scripts', 'wp_enqueue_emoji_styles' );
	remove_filter( 'the_content_feed', 'wp_staticize_emoji' );
	remove_filter( 'comment_text_rss', 'wp_staticize_emoji' );
	remove_filter( 'wp_mail', 'wp_staticize_emoji_for_email' );
}
add_action( 'init', 'skyyrose2_performance_disable_emojis', 1 );

/**
 * Identify routes whose complete public markup is owned by this theme.
 *
 * Block styles and wp-embed remain available on ordinary editorial content,
 * WooCommerce archives, products, cart, checkout, and account routes.
 *
 * @return bool
 */
function skyyrose2_performance_is_governed_route() {
	if ( is_front_page() || is_page_template( 'template-collection.php' ) ) {
		return true;
	}

	$immersive_templates = array(
		'template-immersive-signature.php',
		'template-immersive-black-rose.php',
		'template-immersive-love-hurts.php',
		'template-immersive-kids-capsule.php',
	);

	return is_page_template( $immersive_templates );
}

/** Remove public assets that governed routes do not render. */
function skyyrose2_performance_dequeue_unused_assets() {
	if ( ! is_user_logged_in() ) {
		wp_dequeue_style( 'dashicons' );
	}

	if ( ! skyyrose2_performance_is_governed_route() ) {
		return;
	}

	$unused_styles = array(
		'wp-block-library',
		'wp-block-library-theme',
		'global-styles',
		'classic-theme-styles',
		'wc-blocks-style',
		'wc-blocks-packages-style',
		'wc-blocks-style-active-filters',
	);

	foreach ( $unused_styles as $handle ) {
		wp_dequeue_style( $handle );
	}

	wp_dequeue_script( 'wp-embed' );
}
add_action( 'wp_enqueue_scripts', 'skyyrose2_performance_dequeue_unused_assets', 100 );

/**
 * Prevent the disabled emoji service from retaining a DNS-prefetch hint.
 * Preserve every hint registered by active plugins and route-owned media.
 *
 * @param array<int,mixed> $urls          Hint entries.
 * @param string           $relation_type Hint relationship.
 * @return array<int,mixed>
 */
function skyyrose2_performance_resource_hints( $urls, $relation_type ) {
	if ( 'dns-prefetch' !== $relation_type ) {
		return $urls;
	}

	return array_values(
		array_filter(
			$urls,
			static function ( $entry ) {
				$url = is_array( $entry ) ? ( $entry['href'] ?? '' ) : $entry;
				return false === strpos( (string) $url, 's.w.org' );
			}
		)
	);
}
add_filter( 'wp_resource_hints', 'skyyrose2_performance_resource_hints', 10, 2 );

/**
 * Keep attachment images responsive and free of conflicting load hints.
 *
 * WordPress remains responsible for native lazy-loading heuristics and srcset
 * generation. This filter fills metadata gaps and ensures an explicitly high
 * priority image is never simultaneously marked lazy.
 *
 * @param array<string,mixed> $attr       Image attributes.
 * @param WP_Post             $attachment Attachment object.
 * @return array<string,mixed>
 */
function skyyrose2_performance_image_attributes( $attr, $attachment ) {
	if ( 'high' === ( $attr['fetchpriority'] ?? '' ) ) {
		unset( $attr['loading'] );
	}

	if ( empty( $attr['decoding'] ) ) {
		$attr['decoding'] = 'async';
	}

	if ( ( empty( $attr['width'] ) || empty( $attr['height'] ) ) && ! empty( $attachment->ID ) ) {
		$metadata = wp_get_attachment_metadata( $attachment->ID );
		if ( is_array( $metadata ) ) {
			if ( empty( $attr['width'] ) && ! empty( $metadata['width'] ) ) {
				$attr['width'] = (int) $metadata['width'];
			}
			if ( empty( $attr['height'] ) && ! empty( $metadata['height'] ) ) {
				$attr['height'] = (int) $metadata['height'];
			}
		}
	}

	if ( ! empty( $attr['srcset'] ) && empty( $attr['sizes'] ) && ! empty( $attr['width'] ) ) {
		$attr['sizes'] = sprintf( '(max-width: %1$dpx) 100vw, %1$dpx', (int) $attr['width'] );
	}

	return $attr;
}
add_filter( 'wp_get_attachment_image_attributes', 'skyyrose2_performance_image_attributes', 20, 2 );

/**
 * Limit Core's automatic eager-loading allowance to the first content image.
 * Explicit template attributes continue to take precedence.
 *
 * @return int
 */
function skyyrose2_performance_loading_threshold() {
	return 1;
}
add_filter( 'wp_omit_loading_attr_threshold', 'skyyrose2_performance_loading_threshold' );

/**
 * MIME type for a preloadable image URL.
 *
 * @param string $url Image URL.
 * @return string
 */
function skyyrose2_performance_image_mime( $url ) {
	$extension = strtolower( pathinfo( wp_parse_url( $url, PHP_URL_PATH ) ?: '', PATHINFO_EXTENSION ) );
	$types     = array(
		'avif' => 'image/avif',
		'gif'  => 'image/gif',
		'jpg'  => 'image/jpeg',
		'jpeg' => 'image/jpeg',
		'png'  => 'image/png',
		'webp' => 'image/webp',
	);

	return $types[ $extension ] ?? '';
}

/**
 * Build a preload record only for a bundled SOT asset that exists.
 *
 * @param string $path  Path beneath assets/sot.
 * @param string $media Optional media query.
 * @return array<string,string>
 */
function skyyrose2_performance_sot_preload( $path, $media = '' ) {
	$path       = ltrim( (string) $path, '/' );
	$local_path = SKYYROSE2_DIR . '/assets/sot/' . $path;
	if ( ! is_readable( $local_path ) ) {
		return array();
	}

	$url      = skyyrose2_sot_asset_uri( $path );
	$resource = array(
		'href'          => $url,
		'as'            => 'image',
		'fetchpriority' => 'high',
	);
	$type     = skyyrose2_performance_image_mime( $url );
	if ( $type ) {
		$resource['type'] = $type;
	}
	if ( $media ) {
		$resource['media'] = $media;
	}

	return $resource;
}

/**
 * Build mutually exclusive preloads for an art-directed hero picture.
 *
 * Exactly one record matches a viewport, mirroring the template breakpoints.
 *
 * @param string $desktop Desktop SOT path.
 * @param string $tablet  Tablet SOT path.
 * @param string $mobile  Mobile SOT path.
 * @return array<int,array<string,string>>
 */
function skyyrose2_performance_art_directed_preloads( $desktop, $tablet, $mobile ) {
	$specification = array(
		array( $mobile, '(max-width: 47.99em)' ),
		array( $tablet, '(min-width: 48em) and (max-width: 74.99em)' ),
		array( $desktop, '(min-width: 75em)' ),
	);
	$resources     = array();

	foreach ( $specification as $item ) {
		$resource = skyyrose2_performance_sot_preload( $item[0], $item[1] );
		if ( $resource ) {
			$resources[] = $resource;
		}
	}

	return $resources;
}

/**
 * Resolve the one route-owned LCP asset (or one art-directed set).
 *
 * @return array<int,array<string,string>>
 */
function skyyrose2_performance_route_preloads() {
	if ( is_front_page() ) {
		return skyyrose2_performance_art_directed_preloads(
			'images/hero/responsive/black-rose-bay-bridge-monuments-v4-1440w.webp',
			'images/hero/responsive/black-rose-bay-bridge-monuments-v4-1024w.webp',
			'images/hero/responsive/black-rose-bay-bridge-monuments-v4-640w.webp'
		);
	}

	if ( is_page_template( 'template-collection.php' ) && function_exists( 'skyyrose2_collections' ) ) {
		$slug        = sanitize_title( get_post_field( 'post_name', get_queried_object_id() ) );
		$collections = skyyrose2_collections();
		$collection  = $collections[ $slug ] ?? array();
		if ( ! empty( $collection['hero'] ) ) {
			return skyyrose2_performance_art_directed_preloads(
				$collection['hero'],
				$collection['hero_tablet'] ?? $collection['hero'],
				$collection['hero_mobile'] ?? $collection['hero']
			);
		}
	}

	$immersive_posters = array(
		'template-immersive-signature.php'    => 'images/hero/signature-golden-gate-monuments-v2.webp',
		'template-immersive-black-rose.php'   => 'images/hero/black-rose-bay-bridge-monuments-v4.webp',
		'template-immersive-love-hurts.php'   => 'images/hero/love-hurts-rose-aisle-monuments-v3.webp',
		'template-immersive-kids-capsule.php' => 'images/hero/kids-capsule-heir-throne-v3.webp',
	);
	foreach ( $immersive_posters as $template => $poster ) {
		if ( is_page_template( $template ) ) {
			$resource = skyyrose2_performance_sot_preload( $poster );
			return $resource ? array( $resource ) : array();
		}
	}

	if ( is_page() ) {
		$slug = sanitize_title( get_post_field( 'post_name', get_queried_object_id() ) );
		if ( 'collections' === $slug ) {
			$resource = skyyrose2_performance_sot_preload( 'branding/hero/signature-golden-gate-yacht-1280w.webp' );
			return $resource ? array( $resource ) : array();
		}
		if ( in_array( $slug, array( 'pre-order', 'preorder' ), true ) ) {
			return skyyrose2_performance_art_directed_preloads(
				'images/preorder/responsive/black-rose-salon-1440w.webp',
				'images/preorder/responsive/black-rose-salon-1024w.webp',
				'images/preorder/responsive/black-rose-salon-640w.webp'
			);
		}
		$page_heroes = array(
			'about'   => 'images/about/skyy-rose-founder-hero.webp',
			'contact' => 'images/immersive/scene-signature-oakland-atelier-gpt2.webp',
		);
		if ( isset( $page_heroes[ $slug ] ) ) {
			$resource = skyyrose2_performance_sot_preload( $page_heroes[ $slug ] );
			return $resource ? array( $resource ) : array();
		}
	}

	$image_id   = 0;
	$image_size = 'full';
	if ( is_singular( 'product' ) && function_exists( 'wc_get_product' ) ) {
		$product    = wc_get_product( get_queried_object_id() );
		$image_id   = $product ? (int) $product->get_image_id() : 0;
		$image_size = 'woocommerce_single';
	} elseif ( is_single() ) {
		$image_id = (int) get_post_thumbnail_id( get_queried_object_id() );
	}

	if ( $image_id ) {
		$source = wp_get_attachment_image_src( $image_id, $image_size );
		if ( $source && ! empty( $source[0] ) ) {
			$resource = array(
				'href'          => $source[0],
				'as'            => 'image',
				'fetchpriority' => 'high',
			);
			$type = skyyrose2_performance_image_mime( $source[0] );
			if ( $type ) {
				$resource['type'] = $type;
			}
			$srcset = wp_get_attachment_image_srcset( $image_id, $image_size );
			if ( $srcset ) {
				$resource['imagesrcset'] = $srcset;
				$resource['imagesizes']  = sprintf( '(max-width: %dpx) 100vw, %dpx', (int) $source[1], (int) $source[1] );
			}
			return array( $resource );
		}
	}

	return array();
}

/**
 * Add route hero preloads without duplicating an existing href.
 *
 * @param array<int,mixed> $resources Existing preload records.
 * @return array<int,mixed>
 */
function skyyrose2_performance_preload_resources( $resources ) {
	$existing = array();
	foreach ( $resources as $resource ) {
		if ( is_array( $resource ) && ! empty( $resource['href'] ) ) {
			$existing[ $resource['href'] ] = true;
		}
	}

	foreach ( skyyrose2_performance_route_preloads() as $resource ) {
		if ( empty( $resource['href'] ) || isset( $existing[ $resource['href'] ] ) ) {
			continue;
		}
		$resources[]                    = $resource;
		$existing[ $resource['href'] ] = true;
	}

	return $resources;
}
add_filter( 'wp_preload_resources', 'skyyrose2_performance_preload_resources' );

/**
 * Defer only theme-owned progressive-enhancement scripts.
 *
 * Footer placement, dependency order, reduced-motion handling, Save-Data
 * behavior, and DOM-first/no-JavaScript fallbacks remain unchanged.
 */
function skyyrose2_performance_defer_scripts() {
	$handles = array(
		'skyyrose2-theme',
		'skyyrose2-house-of-roses',
		'skyyrose2-kids-capsule-reveal',
		'skyyrose2-mascot-loader',
		'skyyrose2-immersive',
	);

	foreach ( $handles as $handle ) {
		wp_script_add_data( $handle, 'strategy', 'defer' );
	}
}
add_action( 'wp_enqueue_scripts', 'skyyrose2_performance_defer_scripts', 110 );
add_action( 'wp_footer', 'skyyrose2_performance_defer_scripts', 1 );
