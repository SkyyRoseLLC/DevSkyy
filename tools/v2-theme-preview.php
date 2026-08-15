<?php
/**
 * Local V2 template renderer.
 *
 * Preview-only: loads the actual V2 PHP route templates and theme assets without
 * WordPress or staging. WooCommerce records below are visual fixtures only; live
 * price, stock, variation, cart, payment, and journal data remain WordPress-owned.
 */
declare(strict_types=1);

$theme_dir = realpath( __DIR__ . '/../wordpress-theme/skyyrose-flagship-2' );
if ( ! $theme_dir ) { http_response_code( 500 ); exit( 'V2 theme missing.' ); }
define( 'ABSPATH', __DIR__ . '/' );
define( 'OBJECT', 'OBJECT' );
$route = preg_replace( '/[^a-z0-9-]/i', '', $_GET['route'] ?? 'home' ) ?: 'home';
$preview_state = preg_replace( '/[^a-z-]/', '', $_GET['state'] ?? '' );
$preview_sku = strtolower( preg_replace( '/[^a-z0-9-]/i', '', $_GET['sku'] ?? 'sg-005' ) ) ?: 'sg-005';
$preview_routes = array( 'home', 'collections', 'signature', 'black-rose', 'love-hurts', 'kids-capsule', 'immersive-signature', 'immersive-black-rose', 'immersive-love-hurts', 'immersive-kids-capsule', 'pre-order', 'about', 'contact', 'journal', 'wishlist', 'faq', 'shipping-returns', 'size-guide', 'privacy-policy', 'terms-of-service', 'accessibility', 'cart', 'checkout', 'account', 'order-tracking', 'shop', 'product', '404' );
if ( ! in_array( $route, $preview_routes, true ) ) { $route = 'home'; }
$preview_posts = array();
$preview_cursor = 0;

function __( $text ) { return $text; }
function esc_html__( $text ) { return htmlspecialchars( $text, ENT_QUOTES, 'UTF-8' ); }
function esc_attr__( $text ) { return htmlspecialchars( $text, ENT_QUOTES, 'UTF-8' ); }
function esc_html_e( $text ) { echo esc_html( $text ); }
function esc_attr_e( $text ) { echo esc_attr( $text ); }
function esc_html( $value ) { return htmlspecialchars( (string) $value, ENT_QUOTES, 'UTF-8' ); }
function esc_attr( $value ) { return htmlspecialchars( (string) $value, ENT_QUOTES, 'UTF-8' ); }
function esc_url( $value ) { return htmlspecialchars( (string) $value, ENT_QUOTES, 'UTF-8' ); }
function wp_kses_post( $value ) { return $value; }
function wp_strip_all_tags( $value ) { return trim( strip_tags( (string) $value ) ); }
function wp_trim_words( $text, $num_words = 55, $more = null ) {
	$words = preg_split( '/\s+/', trim( wp_strip_all_tags( $text ) ) );
	if ( count( $words ) <= (int) $num_words ) {
		return implode( ' ', $words );
	}
	return implode( ' ', array_slice( $words, 0, (int) $num_words ) ) . ( null === $more ? ' …' : $more );
}
function sanitize_title( $value ) { return strtolower( preg_replace( '/[^a-z0-9]+/', '-', trim( (string) $value ) ) ); }
function sanitize_key( $value ) { return sanitize_title( $value ); }
function sanitize_html_class( $value ) { return sanitize_title( $value ); }
function sanitize_email( $value ) { return filter_var( (string) $value, FILTER_SANITIZE_EMAIL ); }
function is_email( $value ) { return false !== filter_var( (string) $value, FILTER_VALIDATE_EMAIL ); }
function absint( $value ) { return abs( (int) $value ); }
function add_action() {}
function add_filter() {}
function has_nav_menu() { return false; }
function wp_enqueue_style() {}
function wp_enqueue_script() {}
function wp_generate_uuid4() { return 'preview-00000000-0000-4000-8000-000000000000'; }
function wp_json_encode( $value, $flags = 0 ) { return json_encode( $value, $flags ); }
function do_action( $hook ) {
	global $product, $preview_sku;
	if ( 'woocommerce_before_single_product_summary' === $hook && $product instanceof WC_Product ) {
		$main_id = $product->get_image_id();
		echo '<div class="woocommerce-product-gallery sr2-preview-gallery">';
		echo '<figure class="woocommerce-product-gallery__wrapper"><div class="woocommerce-product-gallery__image">' . wp_get_attachment_image( $main_id, 'full', false, array( 'class' => 'wp-post-image' ) ) . '</div></figure>';
		echo '<div class="sr2-preview-gallery__thumbs">';
		foreach ( $product->get_gallery_image_ids() as $gallery_id ) {
			echo wp_get_attachment_image( $gallery_id, 'thumbnail', false, array( 'class' => 'sr2-preview-gallery__thumb' ) );
		}
		echo '</div></div>';
	} elseif ( 'woocommerce_single_product_summary' === $hook && $product instanceof WC_Product ) {
		echo '<h1 class="product_title entry-title">' . esc_html( $product->get_name() ) . '</h1>';
		echo '<p class="price">' . wp_kses_post( $product->get_price_html() ) . '</p>';
		echo '<div class="woocommerce-product-details__short-description"><p>' . esc_html( $product->get_short_description() ) . '</p></div>';
		echo '<form class="cart variations_form"><table class="variations"><tbody><tr><th><label for="preview-size">Size</label></th><td><select id="preview-size"><option>Choose a size</option><option>XS</option><option>S</option><option>M</option><option>L</option><option>XL</option><option>2XL</option></select></td></tr></tbody></table><div class="quantity"><label for="preview-qty">Quantity</label><input id="preview-qty" type="number" min="1" value="1"></div><button class="single_add_to_cart_button button alt" type="button">Secure this piece</button></form>';
		$collection_labels = array( 'signature' => 'Signature', 'black-rose' => 'Black Rose', 'love-hurts' => 'Love Hurts', 'kids-capsule' => 'Kids Capsule', 'jersey-series' => 'Jersey Series / Black Rose release' );
		$collection_terms  = wp_get_post_terms( $product->get_id(), 'product_cat', array( 'fields' => 'slugs' ) );
		$collection_slug   = $collection_terms[0] ?? 'signature';
		$collection_label   = $collection_labels[ $collection_slug ] ?? 'SkyyRose';
		echo '<div class="product_meta"><span class="sku_wrapper">SKU: <span class="sku">' . esc_html( strtoupper( $product->get_sku() ) ) . '</span></span><span class="posted_in">Collection: ' . esc_html( $collection_label ) . '</span></div>';
		echo '<ul class="sr2-preview-promises"><li>Limited archive release</li><li>Tracked delivery</li><li>Secure checkout</li></ul>';
	} elseif ( 'woocommerce_after_single_product_summary' === $hook ) {
		echo '<section class="woocommerce-tabs sr2-preview-tabs"><h2>Built to carry the story</h2><p>Published construction, care, fit, shipping, and return details remain attached to the live WooCommerce product record.</p><div class="sr2-preview-tabs__grid"><article><h3>Construction</h3><p>Material and finishing details from the product SOT.</p></article><article><h3>Fit</h3><p>Size guidance and model measurements beside the purchase decision.</p></article><article><h3>Delivery</h3><p>Real fulfillment status and preorder timing—never fabricated.</p></article></div></section><section class="related products"><h2>Continue through the house</h2></section>';
	}
}
function apply_filters( $value ) { return $value; }
function is_wp_error() { return false; }
function get_template_directory() { global $theme_dir; return $theme_dir; }
function get_template_directory_uri() { return '/wordpress-theme/skyyrose-flagship-2'; }
function get_theme_file_uri( $path = '' ) { return get_template_directory_uri() . '/' . ltrim( $path, '/' ); }
function home_url( $path = '/' ) {
	$path = trim( $path, '/' );
	$routes = array(
		'collections/signature' => 'signature',
		'collections/black-rose' => 'black-rose',
		'collections/love-hurts' => 'love-hurts',
		'collections/kids-capsule' => 'kids-capsule',
		'worlds/signature' => 'immersive-signature',
		'worlds/black-rose' => 'immersive-black-rose',
		'worlds/love-hurts' => 'immersive-love-hurts',
		'worlds/kids-capsule' => 'immersive-kids-capsule',
	);
	return '/tools/v2-theme-preview.php?route=' . rawurlencode( $routes[ $path ] ?? ( $path ?: 'home' ) );
}
function add_query_arg( $args, $url = '' ) { return $url; }
function wp_verify_nonce() { return true; }
function wp_safe_redirect() { return true; }
function wp_get_referer() { return '/tools/v2-theme-preview.php?route=contact'; }
function wp_mail() { return true; }
function wp_die( $message ) { throw new RuntimeException( (string) $message ); }
function wp_generate_password( $length = 12 ) { return str_repeat( 'p', (int) $length ); }
function set_transient() { return true; }
function get_transient() { return false; }
function delete_transient() { return true; }
function wp_salt() { return 'preview-only-salt'; }
function sanitize_text_field( $value ) { return trim( strip_tags( (string) $value ) ); }
function sanitize_textarea_field( $value ) { return trim( strip_tags( (string) $value ) ); }
function wp_unslash( $value ) { return $value; }
function language_attributes() { echo 'lang="en-US"'; }
function bloginfo( $field ) { echo 'charset' === $field ? 'UTF-8' : 'SkyyRose'; }
function get_bloginfo( $field = '' ) { return 'name' === $field ? 'SkyyRose' : ''; }
function is_home() { global $route; return 'journal' === $route; }
function wp_get_document_title() { return get_the_title(); }
function body_class() { echo 'class="sr2-preview woocommerce"'; }
function wp_body_open() { echo '<div class="sr2-preview-banner">LOCAL V2 TEMPLATE PREVIEW · no staging writes · WooCommerce data is visual-fixture only</div>'; }
function wp_head() {
	global $route;
	$title       = 'home' === $route ? 'SkyyRose — House of Roses' : 'SkyyRose — ' . ucwords( str_replace( '-', ' ', $route ) );
	$description = 'SkyyRose is an Oakland luxury streetwear house where every collection opens a distinct story world.';
	echo '<title>' . esc_html( $title ) . '</title><meta name="description" content="' . esc_attr( $description ) . '"><link rel="stylesheet" href="/wordpress-theme/skyyrose-flagship-2/assets/css/design-tokens.css"><link rel="stylesheet" href="/wordpress-theme/skyyrose-flagship-2/assets/css/theme.css"><style>.sr2-preview-banner{position:fixed;z-index:1000;right:12px;bottom:12px;padding:8px 10px;background:#e2b6a7;color:#100e0c;font:600 10px/1.2 sans-serif;letter-spacing:.08em}.sr2-preview-gallery img{display:block;width:100%;height:auto}.sr2-preview-gallery__thumbs{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:8px;margin-top:8px}.sr2-preview-gallery__thumb{border:1px solid var(--sr2-line);aspect-ratio:1;object-fit:cover}.sr2-pdp-product--portal .quantity{display:grid;gap:4px;flex:0 0 7rem}.sr2-pdp-product--portal .quantity input{width:100%;min-height:48px}.sr2-pdp-product--portal .product_meta{display:grid;gap:6px;margin-top:24px;color:var(--sr2-muted);font:600 .7rem/1.4 var(--sr2-font-ui);letter-spacing:.08em;text-transform:uppercase}.sr2-preview-promises{display:grid;gap:7px;margin-top:24px;padding-left:1.2rem}.sr2-preview-tabs__grid{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:24px;margin-top:32px}@media(max-width:640px){.sr2-preview-banner{right:8px;left:8px}.sr2-preview-tabs__grid{grid-template-columns:1fr}}</style>';
}
function wp_footer() { echo '<script src="/wordpress-theme/skyyrose-flagship-2/assets/js/theme.js" defer></script><script src="/wordpress-theme/skyyrose-flagship-2/assets/js/house-of-roses-motion.js" defer></script><script src="/wordpress-theme/skyyrose-flagship-2/assets/js/kids-capsule-reveal.js" defer></script>'; }
function get_header() { require get_template_directory() . '/header.php'; }
function get_footer() { require get_template_directory() . '/footer.php'; }
function get_template_part( $slug, $name = null, $args = array() ) {
	if ( 'template-parts/skyy-mascot' === $slug ) { return; }
	$file = get_template_directory() . '/' . $slug . ( $name ? '-' . $name : '' ) . '.php';
	if ( is_file( $file ) ) { require $file; }
}
function get_queried_object_id() { return 1; }
function get_the_ID() {
	global $product;
	return $product instanceof WC_Product ? $product->get_id() : 1;
}
function the_ID() { echo esc_attr( get_the_ID() ); }
function get_post_field( $field ) { global $route; return 'post_name' === $field ? $route : ''; }
function get_the_title() { global $route; return ucwords( str_replace( '-', ' ', $route ) ); }
function the_title() { echo esc_html( get_the_title() ); }
function get_the_content() { return ''; }
function the_content() {
	global $route;
	$service_copy = array(
		'faq'              => 'Answers on fit, pre-order timing, order support, and the path to Client Services.',
		'shipping-returns' => 'In-stock and pre-order fulfillment guidance, tracking expectations, and return-request steps.',
		'size-guide'       => 'Measure a piece you own flat, compare the published measurements, and contact Client Services between sizes.',
		'privacy-policy'   => 'The store uses order, support, security, and requested communication data according to the published policy.',
		'terms-of-service' => 'Orders remain subject to acceptance, payment confirmation, availability, and the merchant terms.',
		'accessibility'    => 'SkyyRose works toward a WCAG 2.2 AA shopping experience with keyboard, contrast, and reduced-motion support.',
	);
	if ( isset( $service_copy[ $route ] ) ) {
		echo '<p>' . esc_html( $service_copy[ $route ] ) . '</p><p>Live policy copy is supplied by the WordPress page record after import.</p>';
		return;
	}
	echo '<p>Preview route content is rendered from this V2 template. WordPress supplies the live page record after staging deployment.</p>';
}
function have_posts() { global $preview_posts, $preview_cursor; return $preview_cursor < count( $preview_posts ); }
function the_post() { global $preview_posts, $preview_cursor, $product; $product = $preview_posts[ $preview_cursor++ ]; }
function is_account_page() { return false; }
function get_option( $name ) { return 'admin_email' === $name ? 'hello@skyyrose.com' : ''; }
function wp_nonce_field() { echo '<input type="hidden" value="preview">'; }
function get_page_by_path() { return null; }
function get_permalink( $item = null ) {
	if ( $item instanceof WC_Product ) {
		return '/tools/v2-theme-preview.php?route=product&sku=' . rawurlencode( $item->get_sku() );
	}
	return '/tools/v2-theme-preview.php?route=' . ( $item ? 'product' : 'shop' );
}
function wc_get_page_permalink( $page ) {
	$routes = array( 'shop' => 'shop', 'cart' => 'cart', 'checkout' => 'checkout', 'myaccount' => 'account', 'order-tracking' => 'order-tracking' );
	return '/tools/v2-theme-preview.php?route=' . ( $routes[ $page ] ?? 'home' );
}
function wc_get_cart_url() { return wc_get_page_permalink( 'cart' ); }
function wc_get_checkout_url() { return wc_get_page_permalink( 'checkout' ); }
function wc_get_product_category_list( $id ) { return 'SkyyRose · Collection'; }
function wp_get_post_terms( $product_id = 0, $taxonomy = '', $args = array() ) {
	global $route, $preview_product;
	$fixture_terms = array( 1 => 'signature', 2 => 'black-rose', 3 => 'love-hurts', 4 => 'kids-capsule', 5 => 'kids-capsule' );
	if ( isset( $fixture_terms[ (int) $product_id ] ) ) { return array( $fixture_terms[ (int) $product_id ] ); }
	if ( 'product' === $route && $preview_product instanceof WC_Product && $preview_product->get_id() === (int) $product_id ) {
		$sku = strtolower( $preview_product->get_sku() );
		$jersey_ids = array( 103, 108, 109, 110, 111, 112, 114, 115 );
		return in_array( (int) $product_id, $jersey_ids, true ) || str_starts_with( $sku, 'br-' ) ? array( 'black-rose' ) : ( str_starts_with( $sku, 'lh-' ) ? array( 'love-hurts' ) : ( str_starts_with( $sku, 'kids-' ) ? array( 'kids-capsule' ) : array( 'signature' ) ) );
	}
	return in_array( $route, array( 'signature', 'black-rose', 'love-hurts', 'kids-capsule' ), true ) ? array( $route ) : array();
}
function wp_get_attachment_image( $id, $size = 'thumbnail', $icon = false, $attrs = array() ) {
	$images = array(
		1 => 'images/home/on-model/signature-sg-005.webp',
		2 => 'images/home/on-model/black-rose-br-004.webp',
		3 => 'images/home/on-model/love-hurts-lh-004.webp',
		4 => 'images/products/kids-001-product-proof-400.webp',
		7 => 'images/products/kids-002-product-proof-400.webp',
		5 => 'images/immersive/scene-signature-golden-gate.webp',
		6 => 'images/home/on-model/signature-sg-005.webp',
	);
	$class = ! empty( $attrs['class'] ) ? $attrs['class'] : '';
	return '<img class="' . esc_attr( $class ) . '" src="' . esc_url( get_template_directory_uri() . '/assets/sot/' . ( $images[ $id ] ?? $images[1] ) ) . '" alt="SkyyRose published product view">';
}
function wp_get_attachment_image_url( $id, $size = 'thumbnail' ) {
	$images = array(
		1 => 'images/home/on-model/signature-sg-005.webp',
		2 => 'images/home/on-model/black-rose-br-004.webp',
		3 => 'images/home/on-model/love-hurts-lh-004.webp',
		4 => 'images/products/kids-001-product-proof-400.webp',
		7 => 'images/products/kids-002-product-proof-400.webp',
		5 => 'images/immersive/scene-signature-golden-gate.webp',
		6 => 'images/home/on-model/signature-sg-005.webp',
	);
	return get_template_directory_uri() . '/assets/sot/' . ( $images[ (int) $id ] ?? $images[1] );
}
function woocommerce_page_title() { return 'The House Edit'; }
function woocommerce_product_loop() { return true; }
function woocommerce_product_loop_start() { echo '<ul class="products">'; }
function woocommerce_product_loop_end() { echo '</ul>'; }
function wc_get_template_part() { require get_template_directory() . '/woocommerce/content-product.php'; }
function wc_product_class( $class = '', $product = null ) { echo 'class="' . esc_attr( $class ) . '"'; }
function wc_get_loop_prop( $prop, $default = 0 ) { static $loop = 0; return 'loop' === $prop ? ++$loop : $default; }
function wc_get_stock_html( $product ) { return '<p class="stock in-stock">Available for preorder</p>'; }
function woocommerce_template_loop_add_to_cart() {
	global $product;
	$url = $product instanceof WC_Product ? get_permalink( $product ) : '/tools/v2-theme-preview.php?route=product&sku=sg-005';
	echo '<a class="button product_type_simple add_to_cart_button" href="' . esc_url( $url ) . '">Enter the scene</a>';
}
function post_password_required() { return false; }
function get_the_password_form() { return ''; }
function has_term( $term, $taxonomy = '', $post = null ) {
	global $product;
	$product_id = $post instanceof WC_Product ? $post->get_id() : ( is_numeric( $post ) ? (int) $post : ( $product instanceof WC_Product ? $product->get_id() : 0 ) );
	$requested  = is_array( $term ) ? array_map( 'sanitize_title', $term ) : array( sanitize_title( $term ) );
	return (bool) array_intersect( $requested, wp_get_post_terms( $product_id, $taxonomy, array( 'fields' => 'slugs' ) ) );
}
function _n( $single, $plural, $number ) { return 1 === (int) $number ? $single : $plural; }

class WC_Product {
	private $id; private $name; private $image; private $sku;
	public function __construct( $id, $name, $image, $sku = '' ) { $this->id = $id; $this->name = $name; $this->image = $image; $this->sku = $sku ?: 'preview-' . $id; }
	public function get_id() { return $this->id; }
	public function get_name() { return $this->name; }
	public function get_image_id() { return $this->image; }
	public function get_gallery_image_ids() { return array( 5 ); }
	public function get_price_html() { return '<span class="woocommerce-Price-amount amount">$128.00</span>'; }
	public function is_in_stock() { return true; }
	public function is_visible() { return true; }
	public function is_purchasable() { return true; }
	public function get_type() { return 'variable'; }
	public function get_sku() { return $this->sku; }
	public function get_status() { return 'publish'; }
	public function get_permalink() { return '/tools/v2-theme-preview.php?route=product&sku=' . rawurlencode( $this->sku ); }
	public function get_short_description() { return 'A limited SkyyRose archive piece rooted in Oakland and built to carry the Bay with you.'; }
}
function wc_get_products( $args = array() ) {
	$fixtures = array(
		new WC_Product( 1, 'Bay Bridge Shirt', 1, 'sg-005' ),
		new WC_Product( 2, 'Black Rose Hoodie', 2, 'br-004' ),
		new WC_Product( 3, 'Love Hurts Bomber Jacket', 3, 'lh-004' ),
		new WC_Product( 4, 'Kids Colorblock Hoodie Set — Red/Black', 4, 'kids-001' ),
		new WC_Product( 5, 'Kids Colorblock Hoodie Set — Purple/Black', 7, 'kids-002' ),
	);
	$categories = isset( $args['category'] ) ? array_map( 'sanitize_title', (array) $args['category'] ) : array();
	if ( $categories ) {
		$fixtures = array_values(
			array_filter(
				$fixtures,
				static function ( $fixture ) use ( $categories ) {
					return (bool) array_intersect( $categories, wp_get_post_terms( $fixture->get_id(), 'product_cat', array( 'fields' => 'slugs' ) ) );
				}
			)
		);
	}
	$limit = isset( $args['limit'] ) ? (int) $args['limit'] : count( $fixtures );
	return $limit < 0 ? $fixtures : array_slice( $fixtures, 0, max( 0, $limit ) );
}
function preview_jersey_products() {
	return array(
		new WC_Product( 103, 'Baseball Classic Black', 0, 'br-003' ),
		new WC_Product( 108, 'SF Inspired Football', 0, 'br-008' ),
		new WC_Product( 109, 'Last Oakland Football', 0, 'br-009' ),
		new WC_Product( 110, 'The Bay Basketball', 0, 'br-010' ),
		new WC_Product( 111, 'The Rose Hockey', 0, 'br-011' ),
		new WC_Product( 112, 'Last Oakland Baseball', 0, 'br-012' ),
		new WC_Product( 114, 'Giants Baseball', 0, 'br-014' ),
		new WC_Product( 115, 'Baseball Classic White', 0, 'br-015' )
	);
}
function preview_product_for_sku( $sku ) {
	$wanted = strtolower( (string) $sku );
	foreach ( array_merge( wc_get_products(), preview_jersey_products() ) as $candidate ) {
		if ( strtolower( $candidate->get_sku() ) === $wanted ) {
			return $candidate;
		}
	}
	return null;
}
function wc_get_product( $id ) { foreach ( array_merge( wc_get_products(), preview_jersey_products() ) as $product ) { if ( $product->get_id() === (int) $id ) { return $product; } } return null; }
function wc_get_product_id_by_sku( $sku = '' ) { global $preview_state; if ( 'kids-missing-purple' === $preview_state && 'kids-002' === strtolower( (string) $sku ) ) { return 0; } foreach ( array_merge( wc_get_products(), preview_jersey_products() ) as $product ) { if ( strtolower( $product->get_sku() ) === strtolower( (string) $sku ) ) { return $product->get_id(); } } return 0; }

require $theme_dir . '/functions.php';

if ( in_array( $route, array( 'signature', 'black-rose', 'love-hurts', 'kids-capsule' ), true ) ) {
	require $theme_dir . '/template-collection.php';
} elseif ( 'immersive-signature' === $route ) {
	require $theme_dir . '/template-immersive-signature.php';
} elseif ( 'immersive-black-rose' === $route ) {
	require $theme_dir . '/template-immersive-black-rose.php';
} elseif ( 'immersive-love-hurts' === $route ) {
	require $theme_dir . '/template-immersive-love-hurts.php';
} elseif ( 'immersive-kids-capsule' === $route ) {
	require $theme_dir . '/template-immersive-kids-capsule.php';
} elseif ( 'home' === $route ) {
	require $theme_dir . '/front-page.php';
} elseif ( 'shop' === $route ) {
	$preview_posts = wc_get_products();
	require $theme_dir . '/woocommerce/archive-product.php';
} elseif ( 'product' === $route ) {
	$preview_product = preview_product_for_sku( $preview_sku ) ?: preview_product_for_sku( 'sg-005' );
	$preview_posts = array( $preview_product );
	$GLOBALS['preview_product'] = $preview_product;
	require $theme_dir . '/woocommerce/single-product.php';
} elseif ( 'journal' === $route ) {
	$preview_posts = array();
	require $theme_dir . '/index.php';
} elseif ( '404' === $route ) {
	require $theme_dir . '/404.php';
} elseif ( in_array( $route, array( 'cart', 'checkout', 'account', 'order-tracking' ), true ) ) {
	require $theme_dir . '/template-parts/v2-commerce-preview.php';
} else {
	// WordPress would populate The Loop for every Page route.  Give the real V2
	// page template one inert fixture so local previews render page content too.
	$preview_posts = array( (object) array( 'ID' => 1 ) );
	require $theme_dir . '/page.php';
}
