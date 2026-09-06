<?php
/** Canonical Skyy character and progressively enhanced house concierge. */
defined( 'ABSPATH' ) || exit;
$mascot = skyyrose2_sot_asset_uri( 'images/mascot/skyy-canonical-v2-512w.webp' );
?>
<dialog id="skyy-ask-dialog" class="skyy-ask-dialog" aria-labelledby="skyy-ask-dialog-title" aria-describedby="skyy-ask-description">
	<header class="skyy-ask-dialog__head"><div><p class="skyy-ask-dialog__eyebrow">THE HOUSE CONCIERGE</p><h2 id="skyy-ask-dialog-title">Ask Skyy</h2></div><button id="skyy-ask-cancel" type="button" aria-label="Close Ask Skyy">×</button></header>
	<div class="skyy-ask-dialog__body">
		<div id="skyy-dialog-stage">
		<div id="skyyrose-mascot" class="skyy-concierge-stage" data-state="hidden" data-renderer="static">
			<div id="skyyrose-mascot-trigger" class="skyyrose-mascot__character" aria-hidden="true"><img class="skyyrose-mascot__image" src="<?php echo esc_url( $mascot ); ?>" alt="" width="220" height="340" loading="lazy" decoding="async"><canvas id="skyy-3d-canvas" class="skyy-3d-canvas" width="220" height="340" aria-hidden="true" hidden></canvas></div>
			<button id="skyy-motion-toggle" type="button" aria-pressed="false" hidden>Pause character</button>
			<div class="skyy-hero-actions"><button id="skyy-hero-chat" type="button" aria-haspopup="dialog" aria-controls="skyy-ask-dialog">Ask Skyy</button><button id="skyy-hero-dismiss" type="button">Dismiss Skyy</button></div>
		</div>
		</div>
		<div class="skyy-ask-dialog__conversation"><p id="skyy-ask-description">Explore the house, find a piece, or ask about sizing. Answers use our site guide and current catalog; this is not a live support chat.</p>
			<div id="skyy-conversation" class="skyy-conversation" role="log" aria-label="Conversation with Skyy" aria-live="polite" aria-relevant="additions"></div>
			<div id="skyy-chips" class="skyy-chips" role="group" aria-label="Suggested questions"></div>
			<form id="skyy-ask-form" class="skyy-ask-form"><label for="skyy-ask-input">Your question</label><div class="skyy-ask-form__row"><input id="skyy-ask-input" name="question" type="text" autocomplete="off" maxlength="300" required placeholder="A product, collection, or question…"><button type="submit">Ask</button></div></form>
		</div>
	</div>
</dialog>
