<?php
/**
 * Template Name: Immersive — Love Hurts Castle
 * Template Post Type: page
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

get_template_part(
	'template-parts/immersive/world',
	null,
	array(
		'collection_slug' => 'love-hurts',
		'collection_name' => __( 'Love Hurts', 'skyyrose-flagship-2' ),
		'world_name'      => __( 'The Rose Remains', 'skyyrose-flagship-2' ),
		'world_type'      => 'romantic-castle',
		'kicker'          => __( 'A story from the other side of the door', 'skyyrose-flagship-2' ),
		'direction'       => __( 'Follow the crimson aisle toward the Beast, the glass, and the wound that still guards a rose.', 'skyyrose-flagship-2' ),
		'manifesto'       => __( 'Love Hurts is not the fairytale ending. It is the distance between damage and devotion — the room where tenderness keeps its armor and what cracked still protects what mattered.', 'skyyrose-flagship-2' ),
		'poster'          => 'images/hero/love-hurts-rose-aisle-monuments-v3.webp',
		'accent'          => '#dc143c',
		'accent_rgb'      => '220, 20, 60',
		'scene'           => array( 'cameraPath' => 'crimson-aisle', 'atmosphere' => 'ember-mist', 'instances' => 72 ),
		'chapters'        => array(
			array(
				'id'       => 'cathedral',
				'label'    => __( 'Cathedral of Thorns', 'skyyrose-flagship-2' ),
				'image'    => 'images/immersive/scene-love-hurts-cathedral.webp',
				'alt'      => __( 'A crimson cathedral aisle framed by roses and thorns', 'skyyrose-flagship-2' ),
				'copy'     => __( 'The aisle makes room for rage and reverence at the same time. Every thorn points inward. Every red window insists the feeling is still alive.', 'skyyrose-flagship-2' ),
				'hotspots' => array( array( 'sku' => 'lh-004', 'left' => 69, 'top' => 58 ) ),
			),
			array(
				'id'    => 'beast-chamber',
				'label' => __( 'The Beast’s Chamber', 'skyyrose-flagship-2' ),
				'image' => 'branding/hero/beauty-and-beast-1280w.webp',
				'alt'   => __( 'The Beast facing the chamber with an enchanted rose holding the room', 'skyyrose-flagship-2' ),
				'copy'  => __( 'His back is turned toward us and his attention stays on the room. The enchanted rose — not an empty throne — is the center of gravity.', 'skyyrose-flagship-2' ),
			),
			array(
				'id'       => 'protected-wound',
				'label'    => __( 'The Protected Wound', 'skyyrose-flagship-2' ),
				'image'    => 'images/immersive/scene-love-hurts-cracked-rose-gpt2.webp',
				'alt'      => __( 'A cracked heart and rose protected in a dark red chamber', 'skyyrose-flagship-2' ),
				'copy'     => __( 'A crack is not an empty space. It is proof of impact — and evidence that something remained worth guarding after the collision.', 'skyyrose-flagship-2' ),
				'hotspots' => array( array( 'sku' => 'lh-005', 'left' => 37, 'top' => 62 ) ),
			),
		),
		'exit_title'      => __( 'Keep what mattered.', 'skyyrose-flagship-2' ),
		'exit_copy'       => __( 'Move from the castle into the collection to choose the published piece that holds softness and armor in the same hand.', 'skyyrose-flagship-2' ),
	)
);
