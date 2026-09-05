<?php
/** Collection identity and its existing full-frame monument. @package SkyyRoseFlagship2 */
defined( 'ABSPATH' ) || exit;
$arrival = $args['collection'];
$arrival_presentation = $args['presentation'];
$arrival_media = skyyrose2_collection_arrival_media( $arrival );
?>
<header class="sr2-world-arrival">
	<div class="sr2-world-arrival__identity">
		<p class="sr2-world-index"><span aria-hidden="true"><?php echo esc_html( $arrival_presentation['ordinal'] ); ?> / </span><?php echo esc_html( $arrival['kicker'] ); ?></p>
		<h1><?php echo esc_html( $arrival['name'] ); ?></h1>
		<p class="sr2-world-arrival__line"><?php echo esc_html( $arrival['headline'] ); ?></p>
		<div class="sr2-world-arrival__actions">
			<a class="sr2-control sr2-control--primary" href="#shop"><?php echo esc_html( $arrival['hero_cta'] ); ?></a>
			<a class="sr2-world-text-link" href="#origin"><?php echo esc_html( $arrival['world_cta'] ); ?> <span aria-hidden="true">↓</span></a>
		</div>
	</div>
	<figure class="sr2-world-arrival__monument">
		<img src="<?php echo esc_url( $arrival_media['src'] ); ?>" srcset="<?php echo esc_attr( $arrival_media['srcset'] ); ?>" sizes="<?php echo esc_attr( $arrival_media['sizes'] ); ?>" width="1440" height="810" alt="<?php echo esc_attr( $arrival_presentation['arrival_alt'] ); ?>" fetchpriority="high" loading="eager" decoding="async">
		<figcaption><span><?php echo esc_html( $arrival_presentation['location'] ); ?></span><span><?php echo esc_html( $arrival['name'] ); ?></span></figcaption>
	</figure>
</header>
