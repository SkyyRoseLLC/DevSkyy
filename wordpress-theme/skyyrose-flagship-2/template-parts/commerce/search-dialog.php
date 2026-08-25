<?php
/**
 * Global storefront search dialog.
 *
 * Search remains WordPress-owned: this dialog only provides a fast, accessible
 * entry point from the global navigation and submits to the canonical search
 * route. It deliberately does not fabricate predictive results or catalog facts.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;
?>
<dialog id="sr2-search-dialog" class="sr2-search-dialog" aria-labelledby="sr2-search-dialog-title">
	<form method="dialog" class="sr2-search-dialog__close-form">
		<button type="submit" class="sr2-search-dialog__close" aria-label="<?php esc_attr_e( 'Close search', 'skyyrose-flagship-2' ); ?>">×</button>
	</form>
	<div class="sr2-search-dialog__panel">
		<p class="sr2-eyebrow"><?php esc_html_e( 'The Living Archive', 'skyyrose-flagship-2' ); ?></p>
		<h2 id="sr2-search-dialog-title"><?php esc_html_e( 'Search the house.', 'skyyrose-flagship-2' ); ?></h2>
		<p><?php esc_html_e( 'Find a piece, collection, journal story, or service answer.', 'skyyrose-flagship-2' ); ?></p>
		<form class="sr2-search-dialog__form" role="search" method="get" action="<?php echo esc_url( home_url( '/' ) ); ?>">
			<label for="sr2-global-search"><?php esc_html_e( 'Search SkyyRose', 'skyyrose-flagship-2' ); ?></label>
			<div>
				<input id="sr2-global-search" name="s" type="search" inputmode="search" autocomplete="off" placeholder="<?php esc_attr_e( 'Try “Black Rose”', 'skyyrose-flagship-2' ); ?>" required data-search-input>
				<button class="sr2-button sr2-button--fill" type="submit"><?php esc_html_e( 'Search', 'skyyrose-flagship-2' ); ?></button>
			</div>
		</form>
		<nav class="sr2-search-dialog__links" aria-label="<?php esc_attr_e( 'Search shortcuts', 'skyyrose-flagship-2' ); ?>">
			<a href="<?php echo esc_url( home_url( '/collections/signature/' ) ); ?>"><?php esc_html_e( 'Signature', 'skyyrose-flagship-2' ); ?></a>
			<a href="<?php echo esc_url( home_url( '/collections/black-rose/' ) ); ?>"><?php esc_html_e( 'Black Rose', 'skyyrose-flagship-2' ); ?></a>
			<a href="<?php echo esc_url( home_url( '/collections/love-hurts/' ) ); ?>"><?php esc_html_e( 'Love Hurts', 'skyyrose-flagship-2' ); ?></a>
			<a href="<?php echo esc_url( home_url( '/collections/kids-capsule/' ) ); ?>"><?php esc_html_e( 'Kids Capsule', 'skyyrose-flagship-2' ); ?></a>
		</nav>
	</div>
</dialog>
