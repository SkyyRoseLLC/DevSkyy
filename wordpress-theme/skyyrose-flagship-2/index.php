<?php
/**
 * Primary fallback template.
 *
 * @package SkyyRose_Flagship_2
 */

defined( 'ABSPATH' ) || exit;

get_header();
?>
<main id="primary" class="sr2-section sr2-page-shell">
	<header class="sr2-page-hero">
		<p class="sr2-eyebrow"><?php esc_html_e( 'SkyyRose Journal', 'skyyrose-flagship-2' ); ?></p>
		<h1><?php echo esc_html( is_home() ? get_bloginfo( 'name' ) : wp_get_document_title() ); ?></h1>
	</header>
	<div class="sr2-content">
		<?php if ( have_posts() ) : ?>
			<?php while ( have_posts() ) : ?>
				<?php the_post(); ?>
				<article <?php post_class( 'sr2-journal-entry' ); ?>>
					<p class="sr2-eyebrow"><?php echo esc_html( get_the_date() ); ?></p>
					<h2><a href="<?php the_permalink(); ?>"><?php the_title(); ?></a></h2>
					<?php the_excerpt(); ?>
				</article>
			<?php endwhile; ?>
			<?php the_posts_pagination(); ?>
		<?php else : ?>
			<section class="sr2-journal-fallback" aria-labelledby="sr2-journal-archive-title">
				<p class="sr2-eyebrow"><?php esc_html_e( 'Source archive', 'skyyrose-flagship-2' ); ?></p>
				<h2 id="sr2-journal-archive-title"><?php esc_html_e( 'The work has already been documented.', 'skyyrose-flagship-2' ); ?></h2>
				<p><?php esc_html_e( 'Fresh installs begin with the founder-approved press record below. Published WordPress stories replace this fallback automatically when the Journal is populated.', 'skyyrose-flagship-2' ); ?></p>
				<div class="sr2-journal-fallback__grid">
					<?php foreach ( skyyrose2_press_features() as $feature ) : ?>
						<article><p class="sr2-eyebrow"><?php echo esc_html( $feature['source'] . ' · ' . $feature['date'] ); ?></p><h3><?php echo esc_html( $feature['title'] ); ?></h3><p><?php echo esc_html( $feature['excerpt'] ); ?></p><a class="sr2-text-link" href="<?php echo esc_url( $feature['url'] ); ?>" target="_blank" rel="noopener noreferrer"><?php esc_html_e( 'Read the original', 'skyyrose-flagship-2' ); ?> ↗</a></article>
					<?php endforeach; ?>
				</div>
			</section>
		<?php endif; ?>
	</div>
</main>
<?php
get_footer();
