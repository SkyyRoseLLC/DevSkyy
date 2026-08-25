<?php
/**
 * Founder-approved press archive shown until the live Journal has posts.
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

$press_features = function_exists( 'skyyrose2_press_features' ) ? skyyrose2_press_features() : array();
if ( empty( $press_features ) ) {
	return;
}
?>
<section class="sr2-journal-fallback" aria-labelledby="sr2-journal-archive-title">
	<p class="sr2-eyebrow"><?php esc_html_e( 'Source archive', 'skyyrose-flagship-2' ); ?></p>
	<h2 id="sr2-journal-archive-title"><?php esc_html_e( 'The work has already been documented.', 'skyyrose-flagship-2' ); ?></h2>
	<p><?php esc_html_e( 'Fresh installs begin with the founder-approved press record below. Published WordPress stories replace this fallback automatically when the Journal is populated.', 'skyyrose-flagship-2' ); ?></p>
	<div class="sr2-journal-fallback__grid">
		<?php foreach ( $press_features as $feature ) : ?>
			<article>
				<p class="sr2-eyebrow"><?php echo esc_html( $feature['source'] . ' · ' . $feature['date'] ); ?></p>
				<h3><?php echo esc_html( $feature['title'] ); ?></h3>
				<p><?php echo esc_html( $feature['excerpt'] ); ?></p>
				<a class="sr2-text-link" href="<?php echo esc_url( $feature['url'] ); ?>" target="_blank" rel="noopener noreferrer">
					<?php esc_html_e( 'Read the original', 'skyyrose-flagship-2' ); ?> <span aria-hidden="true">↗</span>
				</a>
			</article>
		<?php endforeach; ?>
	</div>
</section>
