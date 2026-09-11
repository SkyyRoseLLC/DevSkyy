<?php
/**
 * Template Name: Immersive — Signature City Tour
 * Template Post Type: page
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

get_template_part(
	'template-parts/immersive/world',
	null,
	array(
		'collection_slug' => 'signature',
		'collection_name' => __( 'Signature', 'skyyrose-flagship-2' ),
		'world_name'      => __( 'The City Signs Back', 'skyyrose-flagship-2' ),
		'world_type'      => 'city-tour',
		'kicker'          => __( 'Oakland → Golden Gate → Oakland', 'skyyrose-flagship-2' ),
		'direction'       => __( 'A city tour told in gold light, water, and one mark that learned how to travel.', 'skyyrose-flagship-2' ),
		'manifesto'       => __( 'Signature is the first promise of the house. The route begins in Oakland, crosses the water, and comes home with the confidence of a name built to mean something.', 'skyyrose-flagship-2' ),
		'poster'          => 'images/hero/signature-golden-gate-monuments-v2.webp',
		'accent'          => '#d4af37',
		'accent_rgb'      => '212, 175, 55',
		'scene'           => array( 'cameraPath' => 'bridge-crossing', 'atmosphere' => 'golden-hour', 'instances' => 64 ),
		'chapters'        => array(
			array(
				'id'       => 'threshold',
				'label'    => __( 'The Threshold', 'skyyrose-flagship-2' ),
				'image'    => 'images/immersive/scene-signature-golden-gate.webp',
				'alt'      => __( 'Signature monuments facing the Golden Gate Bridge in gold light', 'skyyrose-flagship-2' ),
				'copy'     => __( 'Two monuments hold the entrance: the signature that made the promise and the SR monogram that made it a house. The bridge is not scenery. It is permission to move.', 'skyyrose-flagship-2' ),
				'aside'    => __( 'Scene cue: follow the cable line toward the first piece.', 'skyyrose-flagship-2' ),
				'hotspots' => array( array( 'sku' => 'sg-005', 'left' => 70, 'top' => 55 ) ),
			),
			array(
				'id'       => 'crossing',
				'label'    => __( 'The Water Carries It', 'skyyrose-flagship-2' ),
				'image'    => 'branding/hero/signature-golden-gate-yacht-1280w.webp',
				'alt'      => __( 'A Signature collection passage across the San Francisco Bay', 'skyyrose-flagship-2' ),
				'copy'     => __( 'From the water, the skyline changes without losing its source. Signature turns Oakland memory into movement: quiet construction, clean finish, no need to announce every mile.', 'skyyrose-flagship-2' ),
				'hotspots' => array( array( 'sku' => 'sg-001', 'left' => 34, 'top' => 60 ) ),
			),
			array(
				'id'    => 'atelier',
				'label' => __( 'The First Atelier', 'skyyrose-flagship-2' ),
				'image' => 'images/immersive/scene-signature-oakland-atelier-gpt2.webp',
				'alt'   => __( 'The Oakland atelier where the Signature story returns home', 'skyyrose-flagship-2' ),
				'copy'  => __( 'The route returns to the workroom. Every refined line still carries the hands, the family, and the city that taught the house how to build from concrete.', 'skyyrose-flagship-2' ),
			),
		),
		'exit_title'      => __( 'Sign your place in the route.', 'skyyrose-flagship-2' ),
		'exit_copy'       => __( 'The city tour ends where the garment begins: fit, material, and the piece you choose to carry forward.', 'skyyrose-flagship-2' ),
	)
);
