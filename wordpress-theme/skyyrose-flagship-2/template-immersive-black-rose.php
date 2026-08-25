<?php
/**
 * Template Name: Immersive — Black Rose Cathedral
 * Template Post Type: page
 *
 * @package SkyyRoseFlagship2
 */

defined( 'ABSPATH' ) || exit;

get_template_part(
	'template-parts/immersive/world',
	null,
	array(
		'collection_slug' => 'black-rose',
		'collection_name' => __( 'Black Rose', 'skyyrose-flagship-2' ),
		'world_name'      => __( 'The Moon Keeps the Court', 'skyyrose-flagship-2' ),
		'world_type'      => 'gothic-cathedral',
		'kicker'          => __( 'Oakland after midnight', 'skyyrose-flagship-2' ),
		'direction'       => __( 'Walk the nave between protection and beauty. Neither one has to apologize for the other.', 'skyyrose-flagship-2' ),
		'manifesto'       => __( 'Black is not absence. It is depth, guard, and elegance that knows what it survived. The rose blooms here without asking the room for permission.', 'skyyrose-flagship-2' ),
		'poster'          => 'images/hero/black-rose-bay-bridge-monuments-v4.webp',
		'accent'          => '#c0c0c0',
		'accent_rgb'      => '192, 192, 192',
		'scene'           => array( 'cameraPath' => 'cathedral-nave', 'atmosphere' => 'silver-fog', 'instances' => 96 ),
		'chapters'        => array(
			array(
				'id'     => 'forbidden-garden',
				'label'  => __( 'The Forbidden Garden', 'skyyrose-flagship-2' ),
				'image'  => 'scene-2-black-rose.webp',
				'source' => 'scroll-world',
				'alt'    => __( 'A black rose emerging inside the original Scroll World garden', 'skyyrose-flagship-2' ),
				'copy'   => __( 'The first bloom appears where the world said it did not belong. The garden does not soften it. The darkness lets every edge become visible.', 'skyyrose-flagship-2' ),
			),
			array(
				'id'       => 'midnight-house',
				'label'    => __( 'Midnight House', 'skyyrose-flagship-2' ),
				'image'    => 'branding/hero/forbidden-midnight-1280w.webp',
				'alt'      => __( 'Black Rose architecture under an Oakland midnight sky', 'skyyrose-flagship-2' ),
				'copy'     => __( 'Concrete becomes sanctuary. Silver edges pull shape from shadow. The room is quiet because it is protected, not because it has nothing to say.', 'skyyrose-flagship-2' ),
				'aside'    => __( 'Jersey Series pieces are deliberately reserved for their own Town Line reveal.', 'skyyrose-flagship-2' ),
				'hotspots' => array( array( 'sku' => 'br-004', 'left' => 68, 'top' => 58 ) ),
			),
			array(
				'id'       => 'moon-court',
				'label'    => __( 'The Silver Moon Court', 'skyyrose-flagship-2' ),
				'image'    => 'images/immersive/scene-black-rose-moon-court-gpt2.webp',
				'alt'      => __( 'A moonlit Black Rose court of black marble and silver light', 'skyyrose-flagship-2' ),
				'copy'     => __( 'At the end of the nave, the rose holds court on its own terms. Protection has become polish. Survival has become posture.', 'skyyrose-flagship-2' ),
				'hotspots' => array( array( 'sku' => 'br-006', 'left' => 31, 'top' => 62 ) ),
			),
		),
		'exit_title'      => __( 'Wear the protection beautifully.', 'skyyrose-flagship-2' ),
		'exit_copy'       => __( 'Leave the cathedral for the collection where the published pieces, exact fit, and live availability are waiting.', 'skyyrose-flagship-2' ),
	)
);
