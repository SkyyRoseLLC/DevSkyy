<?php
/**
 * SkyyRose Flagship 2 theme foundation.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

define( 'SKYYROSE2_VERSION', '2.4.4' );
define( 'SKYYROSE2_DIR', get_template_directory() );
define( 'SKYYROSE2_URI', get_template_directory_uri() );

/* Fresh-install, demo-import, and editor integration. */
require_once SKYYROSE2_DIR . '/inc/marketplace.php';
require_once SKYYROSE2_DIR . '/inc/product-catalog.php';
require_once SKYYROSE2_DIR . '/inc/product-media.php';
require_once SKYYROSE2_DIR . '/inc/product-cards.php';
require_once SKYYROSE2_DIR . '/inc/woocommerce.php';
require_once SKYYROSE2_DIR . '/inc/performance.php';
require_once SKYYROSE2_DIR . '/inc/seo-indexing.php';
require_once SKYYROSE2_DIR . '/inc/security.php';

/**
 * Resolve a theme-bundled, SOT-approved asset.
 *
 * @param string $path Path beneath assets/sot.
 * @return string
 */
function skyyrose2_sot_asset_uri( $path ) {
	return SKYYROSE2_URI . '/assets/sot/' . ltrim( $path, '/' );
}

/**
 * Resolve an exact snapshot from the current Scroll World sequence.
 *
 * @param string $path Snapshot filename.
 * @return string
 */
function skyyrose2_scroll_world_asset_uri( $path ) {
	return SKYYROSE2_URI . '/assets/scroll-world/' . ltrim( $path, '/' );
}

/**
 * Resolve the dedicated immersive story page for a collection.
 *
 * @param string $collection Collection slug.
 * @return string
 */
function skyyrose2_immersive_url( $collection ) {
	$collection = sanitize_title( $collection );
	if ( ! in_array( $collection, array( 'signature', 'black-rose', 'love-hurts', 'kids-capsule' ), true ) ) {
		return skyyrose2_marketplace_page_url( 'worlds' );
	}
	return skyyrose2_marketplace_page_url( 'immersive-' . $collection );
}

/**
 * Resolve collection world imagery from its declared source, never from a
 * filename guess. Scroll World files and V2 SOT files have distinct roots.
 *
 * @param array<string,mixed> $scene Declared collection world scene.
 * @return string
 */
function skyyrose2_collection_scene_uri( $scene ) {
	if ( empty( $scene['image'] ) ) {
		return '';
	}
	return isset( $scene['source'] ) && 'scroll-world' === $scene['source'] ? skyyrose2_scroll_world_asset_uri( $scene['image'] ) : skyyrose2_sot_asset_uri( $scene['image'] );
}

/** Theme supports and navigation slots. */
function skyyrose2_setup() {
	load_theme_textdomain( 'skyyrose-flagship-2', SKYYROSE2_DIR . '/languages' );
	add_theme_support( 'title-tag' );
	add_theme_support( 'post-thumbnails' );
	add_theme_support( 'responsive-embeds' );
	add_theme_support( 'align-wide' );
	add_theme_support(
		'html5',
		array( 'search-form', 'comment-form', 'comment-list', 'gallery', 'caption', 'style', 'script' )
	);
	add_theme_support( 'custom-logo', array( 'height' => 80, 'width' => 320, 'flex-height' => true, 'flex-width' => true ) );
	add_theme_support( 'woocommerce' );
	add_theme_support( 'wc-product-gallery-zoom' );
	add_theme_support( 'wc-product-gallery-lightbox' );
	add_theme_support( 'wc-product-gallery-slider' );
	register_nav_menus(
		array(
			'primary' => __( 'Primary Menu', 'skyyrose-flagship-2' ),
			'footer'  => __( 'Footer Menu', 'skyyrose-flagship-2' ),
		)
	);
}
add_action( 'after_setup_theme', 'skyyrose2_setup' );

/** Load self-hosted design system and progressive interactions. */
function skyyrose2_asset_suffix() {
	if ( defined( 'SCRIPT_DEBUG' ) && SCRIPT_DEBUG ) {
		return '';
	}

	/*
	 * Never enqueue a stale release bundle. The authored source remains the
	 * candidate authority until every core minified asset is present and newer
	 * than its source counterpart.
	 */
	$source_assets = array(
		'/assets/css/design-tokens.css' => '/assets/css/design-tokens.min.css',
		'/assets/css/theme.css'         => '/assets/css/theme.min.css',
		'/assets/js/theme.js'           => '/assets/js/theme.min.js',
	);
	foreach ( $source_assets as $source => $minified ) {
		$source_path   = SKYYROSE2_DIR . $source;
		$minified_path = SKYYROSE2_DIR . $minified;
		if ( ! file_exists( $minified_path ) || filemtime( $minified_path ) < filemtime( $source_path ) ) {
			return '';
		}
	}

	return '.min';
}

/**
 * Return a content-addressed version for a theme asset.
 *
 * A release number alone cannot invalidate a browser or edge cache when the
 * file changes between two V2 candidate builds. Hash the exact served asset so
 * an approved change always receives a new URL while unchanged files retain a
 * warm cache. The result is memoized for the duration of the request.
 *
 * @param string $relative_path Path beneath the theme directory.
 * @return string
 */
function skyyrose2_asset_version( $relative_path ) {
	static $versions = array();

	$relative_path = '/' . ltrim( (string) $relative_path, '/' );
	if ( isset( $versions[ $relative_path ] ) ) {
		return $versions[ $relative_path ];
	}

	$path = SKYYROSE2_DIR . $relative_path;
	if ( ! is_readable( $path ) ) {
		return SKYYROSE2_VERSION;
	}

	$hash = hash_file( 'sha256', $path );
	$versions[ $relative_path ] = $hash ? SKYYROSE2_VERSION . '-' . substr( $hash, 0, 12 ) : SKYYROSE2_VERSION;

	return $versions[ $relative_path ];
}

function skyyrose2_assets() {
	$suffix = skyyrose2_asset_suffix();
	$house_motion_suffix = $suffix && file_exists( SKYYROSE2_DIR . '/assets/js/house-of-roses-motion.min.js' ) ? '.min' : '';
	$kids_reveal_suffix = $suffix && file_exists( SKYYROSE2_DIR . '/assets/js/kids-capsule-reveal.min.js' ) ? '.min' : '';
	$tokens_asset = '/assets/css/design-tokens' . $suffix . '.css';
	$theme_asset  = '/assets/css/theme' . $suffix . '.css';
	$theme_script = '/assets/js/theme' . $suffix . '.js';
	$house_script = '/assets/js/house-of-roses-motion' . $house_motion_suffix . '.js';
	$kids_script  = '/assets/js/kids-capsule-reveal' . $kids_reveal_suffix . '.js';
	$mascot_style = '/assets/css/mascot' . $suffix . '.css';
	$loader_script = '/assets/js/mascot-loader' . $suffix . '.js';
	$mascot_script = '/assets/js/mascot' . $suffix . '.js';
	$three_script  = '/assets/js/skyy-3d' . $suffix . '.js';

	wp_enqueue_style( 'skyyrose2-tokens', SKYYROSE2_URI . $tokens_asset, array(), skyyrose2_asset_version( $tokens_asset ) );
	wp_enqueue_style( 'skyyrose2-theme', SKYYROSE2_URI . $theme_asset, array( 'skyyrose2-tokens' ), skyyrose2_asset_version( $theme_asset ) );
	wp_enqueue_script( 'skyyrose2-theme', SKYYROSE2_URI . $theme_script, array(), skyyrose2_asset_version( $theme_script ), true );
	wp_enqueue_script( 'skyyrose2-house-of-roses', SKYYROSE2_URI . $house_script, array( 'skyyrose2-theme' ), skyyrose2_asset_version( $house_script ), true );
	// Editorial collection and reservation pages render native Woo loop actions
	// outside WooCommerce's archive template. Load the same client runtime here
	// so an eligible simple product gets the normal AJAX confirmation, fragments,
	// and updated bag count; the anchor URL remains the no-JS cart fallback.
	if (
		function_exists( 'is_woocommerce' ) &&
		(
			is_page_template( 'template-collection.php' ) ||
			is_page_template( 'template-preorder.php' ) ||
			is_page_template( 'template-parts/v2-preorder.php' )
		)
	) {
		wp_enqueue_script( 'wc-add-to-cart' );
		wp_enqueue_script( 'wc-cart-fragments' );
	}
	if ( is_front_page() ) {
		wp_enqueue_script( 'skyyrose2-kids-capsule-reveal', SKYYROSE2_URI . $kids_script, array( 'skyyrose2-theme' ), skyyrose2_asset_version( $kids_script ), true );
	}
	if ( ! ( function_exists( 'is_checkout' ) && is_checkout() ) ) {
		wp_enqueue_style( 'skyyrose2-mascot', SKYYROSE2_URI . $mascot_style, array( 'skyyrose2-tokens' ), skyyrose2_asset_version( $mascot_style ) );
		wp_enqueue_script( 'skyyrose2-mascot-loader', SKYYROSE2_URI . $loader_script, array(), skyyrose2_asset_version( $loader_script ), true );
		wp_localize_script(
			'skyyrose2-mascot-loader',
			'SKYY_LOADER_CONFIG',
			array(
				'mascotUrl' => add_query_arg( 'ver', skyyrose2_asset_version( $mascot_script ), SKYYROSE2_URI . $mascot_script ),
				'skyy3dUrl' => add_query_arg( 'ver', skyyrose2_asset_version( $three_script ), SKYYROSE2_URI . $three_script ),
			)
		);
		wp_localize_script(
			'skyyrose2-mascot-loader',
			'SKYY_3D_CONFIG',
			array(
				'modelUrl'    => add_query_arg( 'ver', skyyrose2_asset_version( '/assets/models/skyy-mascot.glb' ), SKYYROSE2_URI . '/assets/models/skyy-mascot.glb' ),
				'decoderPath' => SKYYROSE2_URI . '/assets/js/lib/draco/',
			)
		);
	}
}
add_action( 'wp_enqueue_scripts', 'skyyrose2_assets' );

/**
 * Supply route-aware metadata when an SEO plugin is not the active authority.
 * Core continues to own canonical URLs; WooCommerce continues to own product facts.
 */
function skyyrose2_seo_context() {
	$context = array(
		'title'       => get_bloginfo( 'name' ),
		'description' => __( 'SkyyRose is independent luxury streetwear from Oakland, California.', 'skyyrose-flagship-2' ),
		'image'       => skyyrose2_sot_asset_uri( 'branding/hero/flagship-house-runway-gpt2.webp' ),
		'type'        => 'website',
	);

	if ( is_front_page() ) {
		$context['title']       = __( 'SkyyRose | Luxury Grows from Concrete', 'skyyrose-flagship-2' );
		$context['description'] = __( 'Enter SkyyRose: Oakland-rooted luxury streetwear, living collection worlds, limited pieces, and the stories behind the house.', 'skyyrose-flagship-2' );
	} elseif ( is_singular( 'product' ) && function_exists( 'wc_get_product' ) ) {
		$product = wc_get_product( get_queried_object_id() );
		if ( $product ) {
			$context['title']       = $product->get_name() . ' | ' . get_bloginfo( 'name' );
			$context['description'] = wp_strip_all_tags( $product->get_short_description() ?: $product->get_name() . ' · ' . __( 'SkyyRose collection piece.', 'skyyrose-flagship-2' ) );
			$context['image']       = $product->get_image_id() ? wp_get_attachment_image_url( $product->get_image_id(), 'full' ) : $context['image'];
			$context['type']        = 'product';
		}
	} elseif ( is_single() ) {
		$context['title']       = get_the_title() . ' | ' . get_bloginfo( 'name' );
		$context['description'] = wp_trim_words( wp_strip_all_tags( get_the_excerpt() ?: get_the_content() ), 30, '' );
		$context['image']       = get_the_post_thumbnail_url( get_queried_object_id(), 'full' ) ?: $context['image'];
		$context['type']        = 'article';
	} elseif ( is_page() ) {
		$page_slug = sanitize_title( get_post_field( 'post_name', get_queried_object_id() ) );
		$worlds    = skyyrose2_collections();
		if ( isset( $worlds[ $page_slug ] ) ) {
			$world             = $worlds[ $page_slug ];
			$context['title']       = $world['name'] . ' | SkyyRose';
			$context['description'] = wp_strip_all_tags( $world['line'] . ' ' . $world['manifesto'] );
			$context['image']       = skyyrose2_sot_asset_uri( $world['hero'] );
		} else {
			$context['title']       = get_the_title() . ' | ' . get_bloginfo( 'name' );
			$context['description'] = wp_trim_words( wp_strip_all_tags( get_the_excerpt() ?: get_the_content() ), 30, '' ) ?: $context['description'];
			$context['image']       = get_the_post_thumbnail_url( get_queried_object_id(), 'full' ) ?: $context['image'];
		}
	} elseif ( is_search() ) {
		$context['title']       = sprintf( __( 'Search “%s” | SkyyRose', 'skyyrose-flagship-2' ), get_search_query() );
		$context['description'] = __( 'Search SkyyRose products, collections, and journal stories.', 'skyyrose-flagship-2' );
	}

	return $context;
}

function skyyrose2_seo_head() {
	if ( defined( 'WPSEO_VERSION' ) || defined( 'RANK_MATH_VERSION' ) || defined( 'THE_SEO_FRAMEWORK_VERSION' ) ) {
		return;
	}
	$context = skyyrose2_seo_context();
	if ( empty( $context['description'] ) ) {
		return;
	}
	?>
	<meta name="description" content="<?php echo esc_attr( $context['description'] ); ?>">
	<meta property="og:type" content="<?php echo esc_attr( $context['type'] ); ?>">
	<meta property="og:title" content="<?php echo esc_attr( $context['title'] ); ?>">
	<meta property="og:description" content="<?php echo esc_attr( $context['description'] ); ?>">
	<meta property="og:url" content="<?php echo esc_url( is_singular() ? get_permalink() : home_url( add_query_arg( array(), $GLOBALS['wp']->request ?? '' ) ) ); ?>">
	<meta property="og:image" content="<?php echo esc_url( $context['image'] ); ?>">
	<meta name="twitter:card" content="summary_large_image">
	<?php
}
add_action( 'wp_head', 'skyyrose2_seo_head', 4 );

/**
 * Render schema facts owned by the theme.
 *
 * WooCommerce remains the single Product-schema authority.  The theme only
 * emits editorial schema; independently rebuilding a Product entity here
 * creates duplicate offers and inventory facts on product pages.
 */
function skyyrose2_schema_head() {
	if ( defined( 'WPSEO_VERSION' ) || defined( 'RANK_MATH_VERSION' ) || defined( 'THE_SEO_FRAMEWORK_VERSION' ) ) {
		return;
	}
	$schema = array();
	if ( is_single() && ! is_singular( 'product' ) ) {
		$schema = array(
			'@context'         => 'https://schema.org',
			'@type'            => 'Article',
			'headline'         => get_the_title(),
			'datePublished'    => get_the_date( DATE_W3C ),
			'dateModified'     => get_the_modified_date( DATE_W3C ),
			'mainEntityOfPage' => get_permalink(),
			'image'            => get_the_post_thumbnail_url( get_queried_object_id(), 'full' ) ?: null,
			'author'           => array( '@type' => 'Person', 'name' => get_the_author() ),
			'publisher'        => array( '@type' => 'Organization', 'name' => get_bloginfo( 'name' ) ),
		);
	}
	if ( $schema ) {
		echo '<script type="application/ld+json">' . wp_json_encode( $schema, JSON_UNESCAPED_SLASHES | JSON_UNESCAPED_UNICODE ) . '</script>';
	}
}
add_action( 'wp_head', 'skyyrose2_schema_head', 5 );

/** Keep the visible bag count synchronized with WooCommerce add-to-cart fragments. */
function skyyrose2_cart_fragment( $fragments ) {
	ob_start();
	?><span class="sr2-header__bag-count" aria-live="polite"><?php echo esc_html( skyyrose2_cart_count() ); ?></span><?php
	$fragments['.sr2-header__bag-count'] = ob_get_clean();
	return $fragments;
}
add_filter( 'woocommerce_add_to_cart_fragments', 'skyyrose2_cart_fragment' );

/** Store a one-time contact result and return its opaque browser token. */
function skyyrose2_contact_result_token( $status ) {
	$token = wp_generate_password( 24, false, false );
	set_transient( 'skyyrose2_contact_' . hash( 'sha256', $token ), sanitize_key( $status ), 5 * MINUTE_IN_SECONDS );
	return $token;
}

/** Return and consume the trusted contact result for this request. */
function skyyrose2_contact_result() {
	$token = isset( $_GET['contact_result'] ) ? sanitize_text_field( wp_unslash( $_GET['contact_result'] ) ) : '';
	if ( ! $token ) {
		return '';
	}
	$key    = 'skyyrose2_contact_' . hash( 'sha256', $token );
	$status = get_transient( $key );
	delete_transient( $key );
	return is_string( $status ) ? sanitize_key( $status ) : '';
}

/** Redirect back to Contact with a server-issued, one-time result. */
function skyyrose2_contact_redirect( $status ) {
	wp_safe_redirect(
		add_query_arg(
			'contact_result',
			rawurlencode( skyyrose2_contact_result_token( $status ) ),
			home_url( '/contact/' )
		)
	);
	exit;
}

/** Handle client-service messages without exposing raw mail endpoints. */
function skyyrose2_contact_form_submit() {
	if ( 'POST' !== ( $_SERVER['REQUEST_METHOD'] ?? '' ) || empty( $_POST['skyyrose2_contact_submit'] ) ) {
		return;
	}
	if ( ! isset( $_POST['skyyrose2_contact_nonce'] ) || ! wp_verify_nonce( sanitize_text_field( wp_unslash( $_POST['skyyrose2_contact_nonce'] ) ), 'skyyrose2_contact' ) ) {
		wp_die( esc_html__( 'Security check failed.', 'skyyrose-flagship-2' ), '', array( 'response' => 403 ) );
	}
	if ( ! empty( $_POST['contact_website'] ) ) {
		skyyrose2_contact_redirect( 'received' );
	}
	$rate_source = (string) ( $_SERVER['REMOTE_ADDR'] ?? '' ) . '|' . (string) ( $_SERVER['HTTP_USER_AGENT'] ?? '' );
	$rate_key    = 'skyyrose2_contact_rate_' . hash_hmac( 'sha256', $rate_source, wp_salt( 'nonce' ) );
	if ( get_transient( $rate_key ) ) {
		skyyrose2_contact_redirect( 'rate-limited' );
	}
	$name    = sanitize_text_field( wp_unslash( $_POST['contact_name'] ?? '' ) );
	$email   = sanitize_email( wp_unslash( $_POST['contact_email'] ?? '' ) );
	$subject = sanitize_text_field( wp_unslash( $_POST['contact_subject'] ?? 'Client inquiry' ) );
	$message = sanitize_textarea_field( wp_unslash( $_POST['contact_message'] ?? '' ) );
	$consent = ! empty( $_POST['contact_privacy'] );
	if ( ! $name || ! is_email( $email ) || ! $message || ! $consent || mb_strlen( $name ) > 120 || mb_strlen( $subject ) > 160 || mb_strlen( $message ) > 5000 ) {
		skyyrose2_contact_redirect( 'invalid' );
	}
	set_transient( $rate_key, 1, MINUTE_IN_SECONDS );
	$sent = wp_mail( get_option( 'admin_email' ), '[SkyyRose] ' . $subject, "Name: {$name}\nEmail: {$email}\n\n{$message}", array( 'Reply-To: ' . $email ) );
	skyyrose2_contact_redirect( $sent ? 'received' : 'delivery-error' );
}
add_action( 'init', 'skyyrose2_contact_form_submit' );

/**
 * Canonical collection presentation data derived from each generated SOT view.
 *
 * Product truth still comes from WooCommerce. This map owns narrative and
 * verified non-product visual assignments only.
 *
 * @return array<string,array<string,mixed>>
 */
function skyyrose2_collections() {
	return array(
		'signature'    => array(
			'name'       => __( 'Signature', 'skyyrose-flagship-2' ),
			'kicker'     => __( 'The House, Signed', 'skyyrose-flagship-2' ),
			'headline'   => __( 'The Bay is where the signature begins.', 'skyyrose-flagship-2' ),
			'line'       => __( 'A bronze signature answers the SR monogram across Golden Gate light. Oakland made the name; the water carries it forward.', 'skyyrose-flagship-2' ),
			'manifesto'  => __( 'Signature is the first promise of the house: build something worthy of a daughter, rooted in The Town, and finished with enough intention to travel far beyond it.', 'skyyrose-flagship-2' ),
			'world_heading' => __( 'Cross the water. Find the origin.', 'skyyrose-flagship-2' ),
			'world_intro' => __( 'Follow the two monuments from Golden Gate light back to Oakland—where a private mark became a house code.', 'skyyrose-flagship-2' ),
			'shop_kicker' => __( 'The Signature Edit', 'skyyrose-flagship-2' ),
			'shop_heading' => __( 'Carry the first mark forward.', 'skyyrose-flagship-2' ),
			'shop_intro' => __( 'The monument sets the mood. The clothes are gender-neutral by design: choose the shape, size, and story that feel like yours. Each published piece below holds the live product, price, and availability decision.', 'skyyrose-flagship-2' ),
			'card_story' => __( 'Oakland origin, carried without a label. The first mark is made for the person who reaches for it.', 'skyyrose-flagship-2' ),
			'hero_cta' => __( 'Shop Signature', 'skyyrose-flagship-2' ),
			'world_cta' => __( 'Cross into the story', 'skyyrose-flagship-2' ),
			'invitation_title' => __( 'Make the origin yours.', 'skyyrose-flagship-2' ),
			'invitation' => __( 'Choose the piece that carries your name with the same calm certainty.', 'skyyrose-flagship-2' ),
			'invitation_cta' => __( 'Choose Signature', 'skyyrose-flagship-2' ),
			'hero'       => 'images/hero/responsive/signature-golden-gate-monuments-v2-1440w.webp',
			'hero_tablet' => 'images/hero/responsive/signature-golden-gate-monuments-v2-1024w.webp',
			'hero_mobile' => 'images/hero/responsive/signature-golden-gate-monuments-v2-640w.webp',
			'portrait'   => 'scene-1-signature.webp',
			'portrait_source' => 'scroll-world',
			'lockup'     => 'images/lockups/signature-lockup.webp',
			'artifact'   => 'images/logos/sr-monogram-rose-gold.webp',
			'atmosphere' => 'images/logos/rose-gold-rose.webp',
			'lookbook'   => 'images/lookbook/lb-rose-hoodie-beanie-960w.webp',
			'lookbook_mobile' => 'images/lookbook/lb-rose-hoodie-beanie-480w.webp',
			'world'      => array(
				array( 'image' => 'images/immersive/scene-signature-golden-gate.webp', 'label' => __( 'Golden Gate Salon', 'skyyrose-flagship-2' ), 'copy' => __( 'Gold light turns the bridge into a threshold: one side holds the signature, the other holds the rose.', 'skyyrose-flagship-2' ) ),
				array( 'image' => 'branding/hero/signature-golden-gate-yacht-1280w.webp', 'label' => __( 'Bay Atelier', 'skyyrose-flagship-2' ), 'copy' => __( 'The water is not a backdrop. It is the distance the house was built to cross.', 'skyyrose-flagship-2' ) ),
				array( 'image' => 'images/immersive/scene-signature-oakland-atelier-gpt2.webp', 'label' => __( 'The First Atelier', 'skyyrose-flagship-2' ), 'copy' => __( 'Back in Oakland, the polish has a source: work, faith, and the decision to make a name mean something.', 'skyyrose-flagship-2' ) ),
			),
		),
		'black-rose'   => array(
			'name'       => __( 'Black Rose', 'skyyrose-flagship-2' ),
			'kicker'     => __( 'Beauty Without Permission', 'skyyrose-flagship-2' ),
			'headline'   => __( 'Bloom under a Bay Bridge moon.', 'skyyrose-flagship-2' ),
			'line'       => __( 'The Black Rose name holds one side of the room. Its star-and-rose monument holds the other. Between them: the Bay after dark.', 'skyyrose-flagship-2' ),
			'manifesto'  => __( 'Black is not absence. It is depth, protection, and elegance with its guard up. The rose survives because it knows its own value—and never asks a shadow for permission to bloom.', 'skyyrose-flagship-2' ),
			'world_heading' => __( 'Enter after dark.', 'skyyrose-flagship-2' ),
			'world_intro' => __( 'Move from the monument room into a moonlit Oakland court, where every black surface keeps its own light.', 'skyyrose-flagship-2' ),
			'shop_kicker' => __( 'Black Rose / Night Edition', 'skyyrose-flagship-2' ),
			'shop_heading' => __( 'Wear the protection beautifully.', 'skyyrose-flagship-2' ),
			'shop_intro' => __( 'The world is the armor. These pieces are designed gender-neutral, with room for each wearer to make the silhouette their own. WooCommerce remains the authority for each current piece, its price, size, and availability.', 'skyyrose-flagship-2' ),
			'card_story' => __( 'Protection, polish, and enough room to define the fit yourself. The rose does not ask permission to take up space.', 'skyyrose-flagship-2' ),
			'hero_cta' => __( 'Shop Black Rose', 'skyyrose-flagship-2' ),
			'world_cta' => __( 'Enter the moon court', 'skyyrose-flagship-2' ),
			'invitation_title' => __( 'Keep your light protected.', 'skyyrose-flagship-2' ),
			'invitation' => __( 'Choose the piece that lets you show up fully without giving the room all of you.', 'skyyrose-flagship-2' ),
			'invitation_cta' => __( 'Choose Black Rose', 'skyyrose-flagship-2' ),
			// Founder-directed Bay Bridge two-monument salon. Lake Merritt remains an approved alternate.
			'hero'       => 'images/hero/responsive/black-rose-bay-bridge-monuments-v4-1440w.webp',
			'hero_tablet' => 'images/hero/responsive/black-rose-bay-bridge-monuments-v4-1024w.webp',
			'hero_mobile' => 'images/hero/responsive/black-rose-bay-bridge-monuments-v4-640w.webp',
			'portrait'   => 'scene-2-black-rose.webp',
			'portrait_source' => 'scroll-world',
			'lockup'     => 'images/lockups/black-rose-lockup.webp',
			'artifact'   => 'images/lockups/black-rose-star-graphic.webp',
			'atmosphere' => 'images/logos/black-roses-cloud-cluster.webp',
			'lookbook'   => 'images/lookbook/lb-black-rose-football-960w.webp',
			'lookbook_mobile' => 'images/lookbook/lb-black-rose-football-480w.webp',
			'world'      => array(
				array( 'image' => 'scene-2-black-rose.webp', 'source' => 'scroll-world', 'label' => __( 'Forbidden Garden', 'skyyrose-flagship-2' ), 'copy' => __( 'The flower appears where it was never expected to. That is the point.', 'skyyrose-flagship-2' ) ),
				array( 'image' => 'branding/hero/forbidden-midnight-1280w.webp', 'label' => __( 'Midnight House', 'skyyrose-flagship-2' ), 'copy' => __( 'Concrete, chrome, black petals: a house built for quiet strength, not quietness.', 'skyyrose-flagship-2' ) ),
				array( 'image' => 'images/immersive/scene-black-rose-moon-court-gpt2.webp', 'label' => __( 'The Silver Moon Court', 'skyyrose-flagship-2' ), 'copy' => __( 'Under a hard moon, beauty is neither hidden nor explained. It simply holds the court.', 'skyyrose-flagship-2' ) ),
			),
		),
		'love-hurts'   => array(
			'name'       => __( 'Love Hurts', 'skyyrose-flagship-2' ),
			'kicker'     => __( 'The Beast Speaks', 'skyyrose-flagship-2' ),
			'headline'   => __( 'Every wound protects a rose.', 'skyyrose-flagship-2' ),
			'line'       => __( 'Love Hurts stands in scarlet script opposite the cracked-heart star. Down the aisle, the Beast faces the rose he could not forget.', 'skyyrose-flagship-2' ),
			'manifesto'  => __( 'Love Hurts is the part of the story told from the other side of the door: not a fairytale ending, but the distance between damage and devotion. What cracked still protects what mattered.', 'skyyrose-flagship-2' ),
			'world_heading' => __( 'Walk the aisle without looking away.', 'skyyrose-flagship-2' ),
			'world_intro' => __( 'Crimson glass, black thorns, a back-turned Beast, and one rose under glass. This is the story before the ending.', 'skyyrose-flagship-2' ),
			'shop_kicker' => __( 'Love Hurts / The Rose Remains', 'skyyrose-flagship-2' ),
			'shop_heading' => __( 'Take the mark with you.', 'skyyrose-flagship-2' ),
			'shop_intro' => __( 'The story is a frame, never a substitute for the garment. Love Hurts is gender-neutral by design—softness, armor, and fit are yours to define. Current details, sizes, prices, and availability are shown on every published piece.', 'skyyrose-flagship-2' ),
			'card_story' => __( 'Softness and armor live in the same piece. Wear the story your own way; the rose belongs to no one type of body.', 'skyyrose-flagship-2' ),
			'hero_cta' => __( 'Shop Love Hurts', 'skyyrose-flagship-2' ),
			'world_cta' => __( 'Walk the aisle', 'skyyrose-flagship-2' ),
			'invitation_title' => __( 'Keep what mattered.', 'skyyrose-flagship-2' ),
			'invitation' => __( 'Choose the piece that holds the tenderness and the edge in the same hand.', 'skyyrose-flagship-2' ),
			'invitation_cta' => __( 'Choose Love Hurts', 'skyyrose-flagship-2' ),
			// Founder-directed cathedral monuments. V1 Beauty and the Beast remains a world chapter below.
			'hero'       => 'images/hero/responsive/love-hurts-rose-aisle-monuments-v3-1440w.webp',
			'hero_tablet' => 'images/hero/responsive/love-hurts-rose-aisle-monuments-v3-1024w.webp',
			'hero_mobile' => 'images/hero/responsive/love-hurts-rose-aisle-monuments-v3-640w.webp',
			'portrait'   => 'scene-3-love-hurts.webp',
			'portrait_source' => 'scroll-world',
			'lockup'     => 'images/lockups/love-hurts-lockup.webp',
			'artifact'   => 'images/lockups/love-hurts-star-heart-graphic.webp',
			'atmosphere' => 'images/logos/heart-rose-composite.webp',
			// V1's original 960px editorial frame is the visual baseline V2 must exceed.
			'lookbook'   => 'images/lookbook/lb-love-hurts-varsity-960w.webp',
			'lookbook_mobile' => 'images/lookbook/lb-love-hurts-varsity-480w.webp',
			'world'      => array(
				array( 'image' => 'images/immersive/scene-love-hurts-cathedral.webp', 'label' => __( 'Cathedral of Thorns', 'skyyrose-flagship-2' ), 'copy' => __( 'The room makes space for rage, reverence, and a red rose no one gets to touch.', 'skyyrose-flagship-2' ) ),
				array( 'image' => 'branding/hero/beauty-and-beast-1280w.webp', 'label' => __( 'The Beast’s Chamber', 'skyyrose-flagship-2' ), 'copy' => __( 'He faces the chamber with his back to us. The rose—not a throne—holds the room.', 'skyyrose-flagship-2' ) ),
				array( 'image' => 'images/immersive/scene-love-hurts-cracked-rose-gpt2.webp', 'label' => __( 'The Protected Wound', 'skyyrose-flagship-2' ), 'copy' => __( 'A cracked heart is not empty. It is evidence that something stayed worth guarding.', 'skyyrose-flagship-2' ) ),
			),
		),
		'kids-capsule' => array(
			'name'       => __( 'Kids Capsule', 'skyyrose-flagship-2' ),
			'kicker'     => __( 'The Heir', 'skyyrose-flagship-2' ),
			'headline'   => __( 'The throne is already hers.', 'skyyrose-flagship-2' ),
			'line'       => __( 'The Skyy mascot sits front and center, not waiting to inherit the house—already imagining what she will build inside it.', 'skyyrose-flagship-2' ),
			'manifesto'  => __( 'Kids Capsule is not a smaller version of the house. It is the first room built for the next generation: confidence, imagination, and enough space to picture themselves at the center of the story.', 'skyyrose-flagship-2' ),
			'world_heading' => __( 'Enter the heir’s room.', 'skyyrose-flagship-2' ),
			'world_intro' => __( 'The throne comes first. Then the playroom, the runway, and every big idea still waiting for its name.', 'skyyrose-flagship-2' ),
			'shop_kicker' => __( 'Kids Capsule / The Heir Edit', 'skyyrose-flagship-2' ),
			'shop_heading' => __( 'Dress the next chapter.', 'skyyrose-flagship-2' ),
			'shop_intro' => __( 'Explore the published Kids Capsule pieces below. Built for expression without a script, every piece makes room for play, confidence, and personality. Product pages provide the current price, availability, size, and care information.', 'skyyrose-flagship-2' ),
			'card_story' => __( 'Built for imagination before anyone hands out roles. The next chapter belongs to whoever is ready to write it.', 'skyyrose-flagship-2' ),
			'hero_cta' => __( 'Shop Kids Capsule', 'skyyrose-flagship-2' ),
			'world_cta' => __( 'Enter her world', 'skyyrose-flagship-2' ),
			'invitation_title' => __( 'Give imagination a place at the table.', 'skyyrose-flagship-2' ),
			'invitation' => __( 'Choose the piece that lets the next generation arrive as themselves—before anyone tells them to play small.', 'skyyrose-flagship-2' ),
			'invitation_cta' => __( 'Choose Kids Capsule', 'skyyrose-flagship-2' ),
			'hero'       => 'images/hero/responsive/kids-capsule-heir-throne-v3-1440w.webp',
			'hero_tablet' => 'images/hero/responsive/kids-capsule-heir-throne-v3-1024w.webp',
			'hero_mobile' => 'images/hero/responsive/kids-capsule-heir-throne-v3-640w.webp',
			'portrait'   => 'scene-4-kids-capsule.webp',
			'portrait_source' => 'scroll-world',
			'lockup'     => 'images/logos/sr-monogram-rose-gold.webp',
			'artifact'   => 'images/mascot/skyy-canonical-v2.png',
			'atmosphere' => 'images/logos/sr-monogram-rose-gold.webp',
			// Preserve V1's full editorial frame for an honest Kids Capsule baseline.
			'lookbook'   => 'images/lookbook/lb-kid-black-rose-960w.webp',
			'lookbook_mobile' => 'images/lookbook/lb-kid-black-rose-480w.webp',
			'world'      => array(
				array( 'image' => 'images/immersive/scene-kids-capsule-playroom.webp', 'label' => __( 'After-Dark Playroom', 'skyyrose-flagship-2' ), 'copy' => __( 'A room where imagination is treated like an heirloom, not an interruption.', 'skyyrose-flagship-2' ) ),
				array( 'image' => 'images/immersive/scene-kids-capsule-runway.webp', 'label' => __( 'First Runway', 'skyyrose-flagship-2' ), 'copy' => __( 'No borrowed confidence. Just the first walk into a story that already has room for them.', 'skyyrose-flagship-2' ) ),
				array( 'image' => 'images/immersive/scene-kids-capsule-heir-runway-gpt2.webp', 'label' => __( 'The Heir’s Salon', 'skyyrose-flagship-2' ), 'copy' => __( 'The house does not hand down a script. It hands down the room to write one.', 'skyyrose-flagship-2' ) ),
			),
		),
	);
}

/**
 * Return the canonical press corpus used by About and Journal fallbacks.
 *
 * These are source records, not invented editorial posts. A live WordPress
 * post always takes precedence in the Journal; this fallback keeps the four
 * founder-approved V1 articles discoverable on a fresh install.
 *
 * @return array<int,array<string,string>>
 */
function skyyrose2_press_features() {
	return array(
		array(
			'source'  => 'Maxim',
			'date'    => 'February 15, 2023',
			'title'   => '14 Game-Changing Entrepreneurs To Watch In 2023',
			'excerpt' => "Corey Foster is an innovative entrepreneur and artist who has created a truly unique clothing line. The Skyy Rose Collection blends fashion with streetwear, creating high quality and beautiful designs people can't help but admire.",
			'url'     => 'https://www.maxim.com/partner/14-game-changing-entrepreneurs-to-watch-in-2023/',
		),
		array(
			'source'  => 'San Francisco Post',
			'date'    => 'August 23, 2024',
			'title'   => "From Oakland's Streets to Fashion Heights",
			'excerpt' => 'The Skyy Rose Collection, a trailblazing gender-neutral clothing brand, is redefining fashion in the Bay Area and beyond. Established by Corey Foster, a single father with a dream, this Oakland-based brand embodies resilience, creativity, and the power of turning adversity into a catalyst for success.',
			'url'     => 'https://sanfranciscopost.com/the-skyy-rose-collection-from-oaklands-streets-to-fashion-heights/',
		),
		array(
			'source'  => 'Best of Best Review',
			'date'    => 'August 20, 2024',
			'title'   => 'Best Bay Area Clothing Line Award 2024',
			'excerpt' => "The Skyy Rose Collection has been honored with the prestigious Best Bay Area Clothing Line Award 2024 by Best of Best Review. This recognition underscores the brand's exceptional contribution to the fashion industry, particularly in the realm of high-end, gender-neutral clothing that transcends age and gender boundaries.",
			'url'     => 'https://bestofbestreview.com/awards/the-skyy-rose-collection-best-bay-area-clothing-line-award-2024',
		),
		array(
			'source'  => 'CEO Weekly',
			'date'    => 'October 22, 2024',
			'title'   => 'The Unyielding Journey of a Single Father and Entrepreneur',
			'excerpt' => 'Despite facing numerous setbacks, such as failed website attempts and deceitful manufacturers that promised much yet delivered little, this indomitable spirit refused to be quenched. The Skyy Rose Collection represents hope, hard work, and the relentless spirit of never giving up.',
			'url'     => 'https://ceoweekly.com/the-unyielding-journey-of-a-single-father-and-entrepreneur/',
		),
	);
}

/**
 * Build the collection monument reel shown inside the global header.
 *
 * Each founder-approved collection hero remains one continuous room containing
 * its typographic and graphic monuments. This avoids duplicating a scene to
 * simulate an unapproved eight-frame set or treating a transparent lockup as a
 * finished statue photograph.
 *
 * @return array<int,array<string,string>>
 */
function skyyrose2_header_world_frames() {
	$frames = array();
	foreach ( skyyrose2_collections() as $slug => $collection ) {
		$combined_labels = array(
			'signature'    => __( 'Signature typography and SR monogram monuments', 'skyyrose-flagship-2' ),
			'black-rose'   => __( 'Black Rose typography and star-and-rose monuments', 'skyyrose-flagship-2' ),
			'love-hurts'   => __( 'Love Hurts graffiti and cracked-heart monuments', 'skyyrose-flagship-2' ),
			'kids-capsule' => __( 'Skyy, the heir, on her throne', 'skyyrose-flagship-2' ),
		);
		$frames[] = array(
			'collection' => $slug,
			'name'       => $collection['name'],
			'label'      => isset( $combined_labels[ $slug ] ) ? $combined_labels[ $slug ] : $collection['name'],
			'image'      => skyyrose2_sot_asset_uri( $collection['hero'] ),
			'url'        => skyyrose2_collection_url( $slug ),
		);
	}
	return $frames;
}

/**
 * Resolve a theme-owned page from the imported hierarchy before falling back
 * to the documented route. This keeps runtime links aligned with the actual
 * page binding instead of assuming a hard-coded permalink shape.
 *
 * @param string $page_key Marketplace page registry key.
 * @return string
 */
function skyyrose2_marketplace_page_url( $page_key ) {
	$pages = function_exists( 'skyyrose2_marketplace_pages' ) ? skyyrose2_marketplace_pages() : array();
	$key   = sanitize_key( $page_key );
	$page  = isset( $pages[ $key ] ) && is_array( $pages[ $key ] ) ? $pages[ $key ] : array();
	$path  = ! empty( $page['path'] ) ? trim( (string) $page['path'], '/' ) : '';

	if ( $path && function_exists( 'get_page_by_path' ) ) {
		$resolved = get_page_by_path( $path, OBJECT, 'page' );
		if ( $resolved instanceof WP_Post ) {
			return get_permalink( $resolved );
		}
	}

	return $path ? home_url( '/' . $path . '/' ) : home_url( '/' );
}

/**
 * Resolve the WooCommerce shop using its configured page first. A fallback
 * remains available during initial setup before that page exists.
 *
 * @return string
 */
function skyyrose2_shop_url() {
	if ( function_exists( 'wc_get_page_permalink' ) ) {
		$url = wc_get_page_permalink( 'shop' );
		if ( is_string( $url ) && '' !== $url ) {
			return $url;
		}
	}
	return home_url( '/shop/' );
}

/** @param string $slug Collection slug. @return string */
function skyyrose2_collection_url( $slug ) {
	$slug = sanitize_title( $slug );
	if ( ! array_key_exists( $slug, skyyrose2_collections() ) ) {
		return skyyrose2_marketplace_page_url( 'collections' );
	}
	return skyyrose2_marketplace_page_url( $slug );
}

/** Render collection runway with accessible native horizontal scroll. */
function skyyrose2_render_collection_rail() {
	$collections = skyyrose2_collections();
	?>
	<section class="sr2-worlds" aria-labelledby="sr2-worlds-title" data-horizontal-world>
		<header class="sr2-section-head sr2-section-head--split">
			<div><p><?php esc_html_e( 'Choose Your World', 'skyyrose-flagship-2' ); ?></p><h2 id="sr2-worlds-title"><?php esc_html_e( 'Four stories. One house.', 'skyyrose-flagship-2' ); ?></h2></div>
			<div class="sr2-rail-controls"><button type="button" data-rail-prev aria-label="<?php esc_attr_e( 'Previous collection', 'skyyrose-flagship-2' ); ?>">←</button><span data-rail-count>01 / 04</span><button type="button" data-rail-next aria-label="<?php esc_attr_e( 'Next collection', 'skyyrose-flagship-2' ); ?>">→</button></div>
		</header>
		<div class="sr2-worlds__stage" data-scroll-world-stage>
		<div class="sr2-worlds__rail" tabindex="0" aria-label="<?php esc_attr_e( 'Collection worlds. Scroll horizontally.', 'skyyrose-flagship-2' ); ?>" data-horizontal-rail>
			<?php foreach ( $collections as $index => $collection ) : ?>
				<?php
				if ( isset( $collection['portrait_source'] ) && 'scroll-world' === $collection['portrait_source'] ) {
					$portrait_uri = skyyrose2_scroll_world_asset_uri( $collection['portrait'] );
				} else {
					$portrait_uri = skyyrose2_sot_asset_uri( $collection['portrait'] );
				}
				?>
				<a class="sr2-world" data-collection="<?php echo esc_attr( $index ); ?>" href="<?php echo esc_url( skyyrose2_collection_url( $index ) ); ?>">
					<img src="<?php echo esc_url( $portrait_uri ); ?>" alt="" width="1920" height="1275" loading="<?php echo 0 === array_search( $index, array_keys( $collections ), true ) ? 'eager' : 'lazy'; ?>" decoding="async">
					<span class="sr2-world__shade" aria-hidden="true"></span>
					<span class="sr2-world__copy"><small><?php echo esc_html( sprintf( '%02d · %s', array_search( $index, array_keys( $collections ), true ) + 1, $collection['kicker'] ) ); ?></small><strong><?php echo esc_html( $collection['name'] ); ?></strong><em><?php echo esc_html( $collection['line'] ); ?></em><b><?php esc_html_e( 'Enter world', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span></b></span>
				</a>
			<?php endforeach; ?>
		</div>
		<div class="sr2-rail-progress" aria-hidden="true"><span data-rail-progress></span></div>
		</div>
	</section>
	<?php
}

/** Render site header. */
function skyyrose2_header() {
	$bag_url = function_exists( 'wc_get_cart_url' ) ? wc_get_cart_url() : home_url( '/cart/' );
	$frames  = skyyrose2_header_world_frames();
	?>
	<header class="sr2-header" data-site-header>
		<button class="sr2-header__menu" type="button" aria-label="<?php esc_attr_e( 'Open site menu', 'skyyrose-flagship-2' ); ?>" aria-controls="sr2-menu" aria-expanded="false" data-sr2-menu><span></span><span><?php esc_html_e( 'Menu', 'skyyrose-flagship-2' ); ?></span></button>
		<a class="sr2-header__brand" href="<?php echo esc_url( home_url( '/' ) ); ?>" aria-label="<?php esc_attr_e( 'SkyyRose home', 'skyyrose-flagship-2' ); ?>"><img class="sr2-header__brand-mark" src="<?php echo esc_url( skyyrose2_sot_asset_uri( 'brand/skyyrose-logo-still-384w.webp' ) ); ?>" data-brand-animation="<?php echo esc_url( skyyrose2_sot_asset_uri( 'brand/skyyrose-logo-animated-384w.webp' ) ); ?>" width="384" height="216" decoding="async" alt="" aria-hidden="true"><span class="screen-reader-text"><?php esc_html_e( 'SkyyRose', 'skyyrose-flagship-2' ); ?></span></a>
		<a class="sr2-header__bag" href="<?php echo esc_url( $bag_url ); ?>"><?php esc_html_e( 'Bag', 'skyyrose-flagship-2' ); ?> <span class="sr2-header__bag-count" aria-live="polite" aria-label="<?php esc_attr_e( 'items in bag', 'skyyrose-flagship-2' ); ?>"><?php echo esc_html( skyyrose2_cart_count() ); ?></span></a>
		<nav id="sr2-menu" class="sr2-header__nav" aria-label="<?php esc_attr_e( 'Primary navigation', 'skyyrose-flagship-2' ); ?>" data-sr2-nav>
			<div class="sr2-header__nav-main">
				<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'collections' ) ); ?>"><span>01</span><?php esc_html_e( 'Collections', 'skyyrose-flagship-2' ); ?></a>
				<a href="<?php echo esc_url( skyyrose2_shop_url() ); ?>"><span>02</span><?php esc_html_e( 'Shop', 'skyyrose-flagship-2' ); ?></a>
				<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'pre-order' ) ); ?>"><span>03</span><?php esc_html_e( 'Pre-Order', 'skyyrose-flagship-2' ); ?></a>
				<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'journal' ) ); ?>"><span>04</span><?php esc_html_e( 'Journal', 'skyyrose-flagship-2' ); ?></a>
				<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'about' ) ); ?>"><span>05</span><?php esc_html_e( 'About', 'skyyrose-flagship-2' ); ?></a>
				<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'contact' ) ); ?>"><span>06</span><?php esc_html_e( 'Contact', 'skyyrose-flagship-2' ); ?></a>
				<button type="button" data-search-open aria-haspopup="dialog" aria-controls="sr2-search-dialog"><span>07</span><?php esc_html_e( 'Search', 'skyyrose-flagship-2' ); ?></button>
				<?php if ( function_exists( 'wc_get_page_permalink' ) ) : ?><a href="<?php echo esc_url( wc_get_page_permalink( 'myaccount' ) ); ?>"><span>08</span><?php esc_html_e( 'Account', 'skyyrose-flagship-2' ); ?></a><?php endif; ?>
			</div>
			<div class="sr2-header-worlds" data-house-header-reel data-house-header-interval="6500" data-house-header-start="0">
				<header class="sr2-header-worlds__head">
					<div><span><?php esc_html_e( 'The Living Archive', 'skyyrose-flagship-2' ); ?></span><strong><?php esc_html_e( 'Scroll the collection monuments', 'skyyrose-flagship-2' ); ?></strong></div>
					<div class="sr2-header-worlds__controls">
						<button type="button" aria-label="<?php esc_attr_e( 'Previous collection scene', 'skyyrose-flagship-2' ); ?>" data-house-header-prev>←</button>
						<span data-house-header-count>01 / 04</span>
						<button type="button" aria-label="<?php esc_attr_e( 'Next collection scene', 'skyyrose-flagship-2' ); ?>" data-house-header-next>→</button>
						<button type="button" aria-pressed="false" data-house-header-toggle><?php esc_html_e( 'Pause collection scenes', 'skyyrose-flagship-2' ); ?></button>
					</div>
				</header>
				<p class="screen-reader-text" aria-live="polite" data-house-header-status></p>
				<div class="sr2-header-worlds__viewport" tabindex="0" aria-label="<?php esc_attr_e( 'Collection typography and graphic monuments. Scroll horizontally.', 'skyyrose-flagship-2' ); ?>" data-house-header-track>
					<div class="sr2-header-worlds__rail">
						<?php foreach ( $frames as $index => $frame ) : ?>
							<a class="sr2-header-worlds__frame" data-house-header-slide data-house-header-label="<?php echo esc_attr( $frame['label'] ); ?>" data-collection="<?php echo esc_attr( $frame['collection'] ); ?>" href="<?php echo esc_url( $frame['url'] ); ?>" aria-label="<?php echo esc_attr( sprintf( __( 'Enter %1$s from the %2$s frame', 'skyyrose-flagship-2' ), $frame['name'], $frame['label'] ) ); ?>">
								<img src="<?php echo esc_url( $frame['image'] ); ?>" alt="" width="1920" height="1080" loading="<?php echo 0 === $index ? 'eager' : 'lazy'; ?>" decoding="async" aria-hidden="true">
								<span><small><?php echo esc_html( sprintf( '%02d', $index + 1 ) ); ?></small><b><?php echo esc_html( $frame['name'] ); ?></b><em><?php echo esc_html( $frame['label'] ); ?></em></span>
							</a>
						<?php endforeach; ?>
					</div>
				</div>
				<div class="sr2-header-worlds__progress" aria-hidden="true"><span></span></div>
			</div>
		</nav>
	</header>
	<?php
}

/** Render site footer. */
function skyyrose2_footer() {
	?>
	<footer class="sr2-footer">
		<nav class="sr2-footer__nav sr2-footer__nav--house" aria-label="<?php esc_attr_e( 'Explore SkyyRose', 'skyyrose-flagship-2' ); ?>">
			<a href="<?php echo esc_url( skyyrose2_shop_url() ); ?>"><?php esc_html_e( 'Shop', 'skyyrose-flagship-2' ); ?></a>
			<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'collections' ) ); ?>"><?php esc_html_e( 'Collections', 'skyyrose-flagship-2' ); ?></a>
			<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'pre-order' ) ); ?>"><?php esc_html_e( 'Pre-Order', 'skyyrose-flagship-2' ); ?></a>
			<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'journal' ) ); ?>"><?php esc_html_e( 'Journal', 'skyyrose-flagship-2' ); ?></a>
		</nav>
		<div class="sr2-footer__identity"><a class="sr2-footer__brand" href="<?php echo esc_url( home_url( '/' ) ); ?>" aria-label="<?php esc_attr_e( 'SkyyRose home', 'skyyrose-flagship-2' ); ?>"><img src="<?php echo esc_url( skyyrose2_sot_asset_uri( 'brand/skyyrose-logo-still-384w.webp' ) ); ?>" data-brand-animation="<?php echo esc_url( skyyrose2_sot_asset_uri( 'brand/skyyrose-logo-animated-384w.webp' ) ); ?>" data-brand-animation-mode="viewport" alt="" width="384" height="216" loading="lazy" decoding="async" aria-hidden="true"><span class="screen-reader-text"><?php esc_html_e( 'SkyyRose', 'skyyrose-flagship-2' ); ?></span></a><p><?php esc_html_e( 'Oakland, California · Independent luxury fashion.', 'skyyrose-flagship-2' ); ?></p></div>
		<nav class="sr2-footer__nav sr2-footer__nav--service" aria-label="<?php esc_attr_e( 'Customer care', 'skyyrose-flagship-2' ); ?>">
			<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'faq' ) ); ?>"><?php esc_html_e( 'FAQ', 'skyyrose-flagship-2' ); ?></a>
			<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'shipping-returns' ) ); ?>"><?php esc_html_e( 'Shipping + Returns', 'skyyrose-flagship-2' ); ?></a>
			<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'returns-exchanges' ) ); ?>"><?php esc_html_e( 'Returns + Exchanges', 'skyyrose-flagship-2' ); ?></a>
			<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'size-guide' ) ); ?>"><?php esc_html_e( 'Size Guide', 'skyyrose-flagship-2' ); ?></a>
			<a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'contact' ) ); ?>"><?php esc_html_e( 'Support', 'skyyrose-flagship-2' ); ?></a>
		</nav>
		<div class="sr2-footer__legal"><p>© <?php echo esc_html( gmdate( 'Y' ) ); ?> <?php esc_html_e( 'The Skyy Rose Collection LLC', 'skyyrose-flagship-2' ); ?></p><nav aria-label="<?php esc_attr_e( 'Legal', 'skyyrose-flagship-2' ); ?>"><a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'privacy-policy' ) ); ?>"><?php esc_html_e( 'Privacy', 'skyyrose-flagship-2' ); ?></a><a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'terms-of-service' ) ); ?>"><?php esc_html_e( 'Terms', 'skyyrose-flagship-2' ); ?></a><a href="<?php echo esc_url( skyyrose2_marketplace_page_url( 'accessibility' ) ); ?>"><?php esc_html_e( 'Accessibility', 'skyyrose-flagship-2' ); ?></a></nav></div>
	</footer>
	<?php
}

/** Route collection child pages without manual template assignment. */
function skyyrose2_collection_template( $template ) {
	if ( ! is_page() ) {
		return $template;
	}
	$page_id = get_queried_object_id();
	$slug    = sanitize_title( get_post_field( 'post_name', $page_id ) );
	if ( array_key_exists( $slug, skyyrose2_collections() ) ) {
		$expected_path = 'collections/' . $slug;
		if ( function_exists( 'get_page_uri' ) && $expected_path !== trim( (string) get_page_uri( $page_id ), '/' ) ) {
			return $template;
		}
		$collection_template = SKYYROSE2_DIR . '/template-collection.php';
		if ( file_exists( $collection_template ) ) {
			return $collection_template;
		}
	}
	return $template;
}
add_filter( 'template_include', 'skyyrose2_collection_template', 20 );
