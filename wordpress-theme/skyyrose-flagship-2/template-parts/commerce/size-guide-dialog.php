<?php
/**
 * Product-level fit and size guide dialog.
 *
 * Product measurements remain owned by the WooCommerce product record. This
 * dialog gives shoppers a fast, accessible decision aid without inventing
 * measurements or replacing the product page's authoritative fit notes.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;
?>
<dialog id="sr2-size-guide-dialog" class="sr2-size-guide-dialog" aria-labelledby="sr2-size-guide-title">
	<form method="dialog" class="sr2-size-guide-dialog__close-form">
		<button type="submit" class="sr2-size-guide-dialog__close" aria-label="<?php esc_attr_e( 'Close fit and size guide', 'skyyrose-flagship-2' ); ?>">×</button>
	</form>
	<div class="sr2-size-guide-dialog__panel">
		<p class="sr2-eyebrow"><?php esc_html_e( 'Fit / House Guide', 'skyyrose-flagship-2' ); ?></p>
		<h2 id="sr2-size-guide-title"><?php esc_html_e( 'Choose the silhouette you want to live in.', 'skyyrose-flagship-2' ); ?></h2>
		<div class="sr2-size-guide-dialog__steps">
			<article><span>01</span><h3><?php esc_html_e( 'Start with the piece', 'skyyrose-flagship-2' ); ?></h3><p><?php esc_html_e( 'Read its fit note and published measurements first. They are the product-specific source of truth.', 'skyyrose-flagship-2' ); ?></p></article>
			<article><span>02</span><h3><?php esc_html_e( 'Measure what you own', 'skyyrose-flagship-2' ); ?></h3><p><?php esc_html_e( 'Lay a comparable garment flat. Compare chest, length, waist, rise, and inseam without stretching it.', 'skyyrose-flagship-2' ); ?></p></article>
			<article><span>03</span><h3><?php esc_html_e( 'Ask before the order', 'skyyrose-flagship-2' ); ?></h3><p><?php esc_html_e( 'If you are between sizes or buying a made-to-order piece, Client Services can help before checkout.', 'skyyrose-flagship-2' ); ?></p></article>
		</div>
		<div class="sr2-size-guide-dialog__actions">
			<a class="sr2-button sr2-button--fill" href="<?php echo esc_url( home_url( '/size-guide/' ) ); ?>"><?php esc_html_e( 'Open full size guide', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span></a>
			<a class="sr2-button" href="<?php echo esc_url( home_url( '/contact/' ) ); ?>"><?php esc_html_e( 'Ask Client Services', 'skyyrose-flagship-2' ); ?></a>
		</div>
	</div>
</dialog>
