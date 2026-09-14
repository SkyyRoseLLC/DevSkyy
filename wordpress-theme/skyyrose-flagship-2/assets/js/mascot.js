/**
 * SkyyRose Brand Mascot — Skyy Living Character
 *
 * State machine: dormant → walking-in → greeting → idle ↔ speaking → exiting
 *
 * Design rules (binding — see tasks/mascot-design-rulebook.md):
 * - NO pop-in: she always WALKS in (animated entrance), never materializes.
 *   First entrance of a session fires FIRST_ENTRY_DELAY_MS after load —
 *   host-greets-arrival semantics (founder directive 2026-07-08; idle-only
 *   gating meant actively-browsing visitors never saw her). Re-entries
 *   only after a randomized 10–30s idle window, reset by activity.
 * - Hard cap of MAX_PROACTIVE_APPEARANCES proactive appearances per session.
 * - A dismissal (minimize/ESC) is remembered for the rest of the session —
 *   Skyy never walks on again unsolicited after that; the recall pill stays
 *   available for the visitor to bring her back manually.
 * - The same scripted line never resurfaces twice in one session.
 * - Motion for the character image is opt-in via CSS
 *   (`prefers-reduced-motion: no-preference` in mascot.css/skyy-walk.css) —
 *   `prefersReducedMotion` below only affects JS-side *timing* (skipping
 *   artificial delays before showing already-final states), not whether
 *   CSS keyframes run.
 * - The minimize button (always visible + keyboard-reachable once idle or
 *   speaking) is the SC 2.2.2 pause/stop control for the looping idle
 *   breathing animation.
 * - Routine speech uses the non-modal speech bubble (role="status"). The
 *   "Ask Skyy" free-text lookup is the one deliberate, user-initiated panel
 *   interaction, and uses a native <dialog> + showModal() with focus
 *   restore (SC 2.1.2) instead of hijacking the bubble.
 *
 * Features:
 * - Contextual speech bubbles with typewriter text
 * - Choice chips drive conversation trees
 * - Tier 1 guide brain: matches free text against data/site-guide.json
 *   intents (deterministic keyword match, no external lib, no network call)
 * - Tier 2 (optional, Customizer-gated): falls back to the existing AJAX
 *   seam when the Tier 1 matcher finds nothing
 * - IntersectionObserver reacts to product grids entering view
 * - Product hover triggers excited reaction
 * - Minimize (walk off) + recall pill
 * - Keyboard accessible (Escape to dismiss/minimize)
 * - Respects prefers-reduced-motion
 *
 * @package SkyyRose_Flagship
 * @since   4.0.0
 */
(function () {
	'use strict';

	// -------------------------------------------------------------------------
	// Constants
	// -------------------------------------------------------------------------

	var IDLE_TRIGGER_MIN_MS      = 10000;
	var IDLE_TRIGGER_MAX_MS      = 30000;
	var ACTIVITY_THROTTLE_MS     = 250;
	/* First entrance of a session fires this long after load REGARDLESS of
	 * activity — she is the site host and greets arrivals. Only re-entries
	 * use the idle-trigger window above (an actively scrolling visitor never
	 * accumulates 10-30s of stillness, so idle-only gating means engaged
	 * visitors never meet her — the exact bug shipped in v1.9.2). */
	var FIRST_ENTRY_DELAY_MS     = 4500;
	var TYPEWRITER_SPEED         = 28;
	var BUBBLE_AUTO_HIDE         = 8000;
	var MAX_PROACTIVE_APPEARANCES = 3;

	var SESSION_KEY_GREETED   = 'skyy_greeted';
	var SESSION_KEY_DISMISSED = 'skyy_dismissed';
	var SESSION_KEY_APPEARANCES = 'skyy_appearances';
	var SESSION_KEY_SHOWN     = 'skyy_shown_prompts';

	// -------------------------------------------------------------------------
	// DOM References
	// -------------------------------------------------------------------------

	var mascotEl    = document.getElementById('skyyrose-mascot');
	if (!mascotEl) return;

	var triggerBtn   = document.getElementById('skyyrose-mascot-trigger');
	var minimizeBtn  = document.getElementById('skyyrose-mascot-minimize');
	var recallBtn    = document.getElementById('skyyrose-mascot-recall');
	var bubble       = document.getElementById('skyy-bubble');
	var bubbleText   = document.getElementById('skyy-bubble-text');
	var chipsEl      = document.getElementById('skyy-chips');
	var askTrigger   = document.getElementById('skyy-ask-trigger');
	var askDialog    = document.getElementById('skyy-ask-dialog');
	var askForm      = document.getElementById('skyy-ask-form');
	var askInput     = document.getElementById('skyy-ask-input');
	var askCancelBtn = document.getElementById('skyy-ask-cancel');

	if (!triggerBtn || !minimizeBtn || !recallBtn || !bubble || !bubbleText || !chipsEl) return;

	var prefersReducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
	var context = mascotEl.getAttribute('data-context') || 'default';

	var guideData = ( window.SKYY_GUIDE_DATA && window.SKYY_GUIDE_DATA.intents ) ? window.SKYY_GUIDE_DATA : { pages: {}, intents: [] };
	var guidePromise = null;
	var mascotConfig = window.SKYY_MASCOT_CONFIG || { pageTip: '', llmEnabled: false };

	// State: dormant | walking-in | greeting | idle | speaking | exiting
	var state = 'dormant';

	// Bridge to skyy-3d.js: every state transition (and gesture moment) is
	// broadcast as a `skyy:*` CustomEvent so the 3D loader can switch
	// animation clips. skyy-3d.js listens; nothing else needs these.
	function emitSkyy(name) {
		mascotEl.dataset.state = name;
		document.dispatchEvent(new CustomEvent('skyy:' + name));
	}

	// Pending greeting delay (wave-then-speak). Tracked so any user action
	// inside the gap — manual trigger, chip advance, minimize, ESC — cancels
	// the deferred greeting instead of letting it clobber the new state.
	var greetTimer = null;

	function clearGreetTimer() {
		if (greetTimer !== null) {
			clearTimeout(greetTimer);
			greetTimer = null;
		}
	}

	var typewriterTimer  = null;
	var autoDismissTimer = null;
	var idleTimer        = null;
	var lastActivityReset = 0;
	var askDialogOpener   = null;
	var pendingQuestion   = null;

	// -------------------------------------------------------------------------
	// Session Memory — appearances, dismissal, shown-prompt dedupe
	// -------------------------------------------------------------------------

	function getShownPrompts() {
		try {
			var raw = sessionStorage.getItem(SESSION_KEY_SHOWN);
			var parsed = raw ? JSON.parse(raw) : [];
			return Array.isArray(parsed) ? parsed : [];
		} catch (e) {
			return [];
		}
	}

	function markPromptShown(id) {
		if (!id) return;
		var shown = getShownPrompts();
		if (shown.indexOf(id) === -1) {
			shown.push(id);
			try {
				sessionStorage.setItem(SESSION_KEY_SHOWN, JSON.stringify(shown));
			} catch (e) { /* sessionStorage unavailable — degrade silently */ }
		}
	}

	function wasPromptShown(id) {
		return getShownPrompts().indexOf(id) !== -1;
	}

	function getAppearanceCount() {
		var raw = sessionStorage.getItem(SESSION_KEY_APPEARANCES);
		var n = raw ? parseInt(raw, 10) : 0;
		return isNaN(n) ? 0 : n;
	}

	function recordProactiveAppearance() {
		try {
			sessionStorage.setItem(SESSION_KEY_APPEARANCES, String(getAppearanceCount() + 1));
		} catch (e) { /* degrade silently */ }
	}

	function proactiveAppearancesExhausted() {
		return getAppearanceCount() >= MAX_PROACTIVE_APPEARANCES;
	}

	function isDismissedThisSession() {
		return sessionStorage.getItem(SESSION_KEY_DISMISSED) === '1';
	}

	function markDismissed() {
		try {
			sessionStorage.setItem(SESSION_KEY_DISMISSED, '1');
		} catch (e) { /* degrade silently */ }
	}

	// -------------------------------------------------------------------------
	// Conversation Scripts
	// -------------------------------------------------------------------------

	var SCRIPTS = {
		homepage: {
			greeting: {
				text: 'Hey! I’m Skyy 👋 Welcome to SkyyRose — Luxury Grows from Concrete.',
				chips: [
					{ id: 'new_drops', label: 'What’s new?',    next: 'new_drops' },
					{ id: 'preorder',  label: 'Pre-order drops',     next: 'preorder'  },
					{ id: 'story',     label: 'Our story',           next: 'story'     }
				]
			},
			new_drops: {
				text: 'The Black Rose and Love Hurts collections are live.',
				chips: [
					{ id: 'shop', label: 'Shop now →', action: '/shop/', next: null },
					{ id: 'world', label: 'Enter the world', action: '/collections-world/', next: null }
				]
			},
			preorder: {
				text: 'Most of the collection is pre-order — made for you, no overstock. 🌹',
				chips: [{ id: 'preorder_page', label: 'See pre-orders', action: '/pre-order/', next: null }]
			},
			story: {
				text: 'SkyyRose was born in Oakland. Corey Foster built this from the concrete up. Real luxury, real roots.',
				chips: [{ id: 'about', label: 'Read our story', action: '/about/', next: null }]
			}
		},
		'black-rose': {
			greeting: {
				text: 'Oakland in your DNA? 🖤 Black Rose was made for the ones who built something from nothing.',
				chips: [
					{ id: 'jerseys', label: 'Show me the jerseys',  next: 'jerseys'                            },
					{ id: 'br_shop', label: 'Browse collection',    action: '/collections/black-rose/', next: null }
				]
			},
			jerseys: {
				text: 'SF inspired, Last Oakland, The Bay, The Rose — each its own story.',
				chips: [{ id: 'br_shop', label: 'Browse →', action: '/collections/black-rose/', next: null }]
			}
		},
		'love-hurts': {
			greeting: {
				text: 'Love Hurts is for the ones who feel everything deeply. ❤️',
				chips: [
					{ id: 'lh_fannie', label: 'What’s The Fannie?', next: 'fannie'                              },
					{ id: 'lh_browse', label: 'Browse collection',       action: '/collections/love-hurts/', next: null }
				]
			},
			fannie: {
				text: 'The Fannie is a statement piece — small, bold, SkyyRose. Love Hurts edition. 🌹',
				chips: [{ id: 'lh_shop', label: 'See it →', action: '/collections/love-hurts/', next: null }]
			}
		},
		signature: {
			greeting: {
				text: 'SF from the soul. 🌉 The Signature collection is Golden Gate, fog, and Bay Area energy.',
				chips: [
					{ id: 'sg_bridge', label: 'The Bay Bridge drops?', next: 'bridge'                               },
					{ id: 'sg_browse', label: 'Browse collection',      action: '/collections/signature/', next: null }
				]
			},
			bridge: {
				text: 'Bay Bridge Shorts + Shirt — the drop that defines the collection.',
				chips: [{ id: 'sg_shop', label: 'See it →', action: '/collections/signature/', next: null }]
			}
		},
		'kids-capsule': {
			greeting: {
				text: 'Little rebels need luxury too. 👑 Kids Capsule brings the SkyyRose story down a size.',
				chips: [{ id: 'kc_browse', label: 'Browse Kids Capsule', action: '/collections/kids-capsule/', next: null }]
			}
		},
		preorder: {
			greeting: {
				text: 'Pre-ordering means we make it for you — no overstock, no waste. Pure luxury.',
				chips: [{ id: 'po_browse', label: 'See all pre-orders', action: '/pre-order/', next: null }]
			}
		},
		'404': {
			greeting: {
				text: 'That page doesn’t exist — let me point you somewhere real. 🌹',
				chips: [
					{ id: '404_home', label: 'Take me home',      action: '/',      next: null },
					{ id: '404_shop', label: 'Shop collections',  action: '/shop/', next: null }
				]
			}
		},
		'default': {
			greeting: {
				text: 'Hey! I’m Skyy 👋 Need help finding something?',
				chips: [
					{ id: 'def_shop', label: 'Shop now',  action: '/shop/',     next: null },
					{ id: 'def_help', label: 'Get help',  action: '/contact/',  next: null }
				]
			}
		}
	};

	// -------------------------------------------------------------------------
	// Typewriter
	// -------------------------------------------------------------------------

	function typewrite(el, text, onDone) {
		clearInterval(typewriterTimer);
		el.textContent = '';
		if (prefersReducedMotion) {
			el.textContent = text;
			if (onDone) onDone();
			return;
		}
		var i = 0;
		typewriterTimer = setInterval(function () {
			el.textContent += text[i++];
			if (i >= text.length) {
				clearInterval(typewriterTimer);
				if (onDone) onDone();
			}
		}, TYPEWRITER_SPEED);
	}

	// -------------------------------------------------------------------------
	// Speech Bubble
	// -------------------------------------------------------------------------

	function speak(text, chips) {
		clearGreetTimer();
		clearTimeout(autoDismissTimer);
		state = 'speaking';
		emitSkyy('speaking');
		mascotEl.classList.add('skyy--speaking');
		triggerBtn.setAttribute('aria-expanded', 'true');

		// Clear previous chips
		chipsEl.replaceChildren();

		// Show bubble
		bubble.removeAttribute('hidden');
		// Force reflow so transition fires
		void bubble.offsetWidth;
		bubble.classList.add('skyy-bubble--visible');

		if (askTrigger) {
			askTrigger.hidden = false;
		}

		// Typewrite, then render chips
		typewrite(bubbleText, text, function () {
			if (chips && chips.length) {
				chips.forEach(function (chip) {
					var btn = document.createElement('button');
					btn.type = 'button';
					btn.className = 'skyy-chip';
					btn.textContent = chip.label;
					btn.dataset.chipId = chip.id;
					btn.dataset.action = chip.action || '';
					btn.dataset.next   = chip.next   || '';
					chipsEl.appendChild(btn);
				});
			} else {
				// No chips — auto-dismiss after delay
				autoDismissTimer = setTimeout(dismissBubble, BUBBLE_AUTO_HIDE);
			}
		});
	}

	function dismissBubble() {
		clearTimeout(autoDismissTimer);
		clearInterval(typewriterTimer);
		bubble.classList.remove('skyy-bubble--visible');
		mascotEl.classList.remove('skyy--speaking');
		triggerBtn.setAttribute('aria-expanded', 'false');
		setTimeout(function () {
			if (!bubble.classList.contains('skyy-bubble--visible')) {
				bubble.setAttribute('hidden', '');
				bubbleText.textContent = '';
				chipsEl.replaceChildren();
				if (askTrigger) {
					askTrigger.hidden = true;
				}
			}
			if (state === 'speaking' && !(askDialog && askDialog.open)) {
				state = 'idle';
				emitSkyy('idle');
			}
		}, 300);
	}

	// -------------------------------------------------------------------------
	// Chip Handling
	// -------------------------------------------------------------------------

	chipsEl.addEventListener('click', function (e) {
		var btn = e.target.closest('.skyy-chip');
		if (!btn) return;

		var action = btn.dataset.action;
		var next   = btn.dataset.next;

		// Visitor picked a destination — Skyy points the way.
		emitSkyy('point');

		if (action) {
			window.location.href = action;
			return;
		}

		if (next) {
			var script = SCRIPTS[context] || SCRIPTS['default'];
			var node   = script[next];
			if (node) {
				speak(node.text, node.chips || null);
			} else {
				dismissBubble();
			}
		}
	});

	// -------------------------------------------------------------------------
	// Staging house guide, now carried by the walk-on character.
 var data = window.SKYY_GUIDE_DATA || {};
 var intents = Array.isArray(data.intents) ? data.intents : [];
 var products = Array.isArray(data.products) ? data.products : [];
 var guideAvailable = Array.isArray(data.products) && Array.isArray(data.intents);
 var log = document.getElementById('skyy-conversation');
 var chatStage = document.getElementById('skyy-chat-stage');
 var characterHome = mascotEl.parentNode;
 var chatMotionTimer;
 var exitTimer;
 var chatClosing = false;
  function normalize(value) {
    return String(value || '')
      .toLowerCase()
      .replace(/[^\p{L}\p{N}]+/gu, ' ')
      .trim();
  }
  function includes(query, phrase) {
    return phrase && (' ' + query + ' ').includes(' ' + normalize(phrase) + ' ');
  }
  function safeUrl(value) {
    if (typeof value !== 'string' || !value.trim()) return '';
    try {
      var url = new URL(value, location.href);
      return url.origin === location.origin && /^https?:$/.test(url.protocol) ? url.href : '';
    } catch (_) {
      return '';
    }
  }
  function add(text, speaker, links) {
    var firstMessage = log.children.length === 0;
    var entry = document.createElement('div');
    entry.className = 'skyy-message skyy-message--' + speaker;
    var label = document.createElement('strong');
    label.textContent = speaker === 'visitor' ? 'You' : 'Skyy';
    var paragraph = document.createElement('p');
    paragraph.textContent = text;
    entry.append(label, paragraph);
    (links || []).forEach(function (item) {
      var href = safeUrl(item.url || item.link);
      if (!href) return;
      var anchor = document.createElement('a');
      anchor.href = href;
      anchor.textContent = item.label || item.name || 'Explore';
      entry.appendChild(anchor);
    });
    log.appendChild(entry);
    while (log.children.length > 20) log.firstElementChild.remove();
    log.scrollTop = firstMessage ? 0 : log.scrollHeight;
  }
  function answer(question) {
    var query = normalize(question);
    if (!query) return;
    mascotEl.dataset.conversation = 'thinking';
    document.dispatchEvent(new CustomEvent('skyy:thinking'));
    add(question, 'visitor');
    if (!guideAvailable) {
      mascotEl.dataset.conversation = 'chat-failure';
      add('The house guide is unavailable right now. You can still browse the shop or contact the house.', 'skyy', [{ url: (data.pages && data.pages.contact ? data.pages.contact.url : '/contact/'), label: 'Contact the house' }]);
      emitSkyy('idle');
      return;
    }
    var exact = products.filter(function (p) {
      return includes(query, p.sku) || includes(query, p.name);
    });
    var found = exact.length
      ? exact
      : products.filter(function (p) {
          var title = normalize(p.name + ' ' + (p.collection || ''));
          var words = query.split(' ').filter(function (word) {
            return (
              word.length > 2 &&
              !['the', 'show', 'find', 'for', 'with', 'have', 'want', 'some', 'please', 'products', 'product'].includes(
                word
              )
            );
          });
          return (
            words.length > 0 &&
            words.every(function (word) {
              return includes(title, word);
            })
          );
        });
    var match = intents
      .map(function (intent) {
        var patterns = Array.isArray(intent.patterns) ? intent.patterns : [];
        return {
          intent: intent,
          score: patterns.reduce(function (score, pattern) {
            return includes(query, pattern) ? Math.max(score, normalize(pattern).length) : score;
          }, 0),
        };
      })
      .sort(function (a, b) {
        return b.score - a.score;
      })[0];
    if (found.length) {
      add(
        'Here are matching pieces from our catalog. Open a product for its current price, available options, and purchase details.',
        'skyy',
        found.slice(0, 4).map(function (p) {
          return { url: p.url, label: p.name + (p.sku ? ' · ' + p.sku : '') };
        })
      );
    } else if (match && match.score) {
      var intent = match.intent;
      add(String(intent.answer || ''), 'skyy', intent.link ? [{ url: intent.link, label: intent.label }] : []);
    } else {
      add(
        'I can help you find a piece by name or SKU, explore a collection, or point you to our site information. For a question about an order, please contact the house.',
        'skyy',
        Object.values(data.pages || {})
          .filter(function (p) {
            return p && /contact|shop/i.test(p.label || '');
          })
          .slice(0, 2)
      );
    }
    mascotEl.dataset.conversation = found.length ? 'gesture' : /^(hi|hello|hey)( skyy)?$/.test(query) ? 'greeting' : 'talking';
    emitSkyy(found.length ? 'joy' : /^(hi|hello|hey)( skyy)?$/.test(query) ? 'wave' : 'speaking');
    clearInterval(chatMotionTimer);
    if (mascotEl.dataset.conversation === 'talking') {
      var visibleTalkTime = 0;
      chatMotionTimer = setInterval(function () {
        if (!askDialog.open) { clearInterval(chatMotionTimer); return; }
        if (document.hidden || window.skyyRoseMascot3D?.isLoading()) return;
        visibleTalkTime += 250;
        if (visibleTalkTime >= 3200) { clearInterval(chatMotionTimer); emitSkyy('idle'); }
      }, 250);
    }
  }

 function openChat() {
   if (!askDialog || chatClosing || askDialog.open || !log || typeof askDialog.showModal !== 'function') return;
   if (document.querySelector('dialog[open]')) return;
   clearTimeout(exitTimer);
   recallBtn.style.display = 'none'; recallBtn.setAttribute('aria-hidden', 'true');
   askDialogOpener = document.activeElement;
   clearGreetTimer(); clearTimeout(autoDismissTimer); clearTimeout(idleTimer);
   dismissBubble();
   state = 'speaking';
   mascotEl.classList.remove('skyyrose-mascot--hidden', 'skyyrose-mascot--entering', 'skyyrose-mascot--exiting');
   mascotEl.classList.add('skyyrose-mascot--idle');
   mascotEl.setAttribute('aria-hidden', 'false');
   chatStage.appendChild(mascotEl);
   askDialog.showModal();
   triggerBtn.setAttribute('aria-expanded', 'true');
   emitSkyy('walking-in'); emitSkyy('idle');
   if (!log.children.length) add(data.greeting || 'I’m Skyy. Which piece or collection would you like to explore?', 'skyy');
   askInput.focus({ preventScroll: true });
 }
 if (askTrigger && askDialog && askForm && askInput && log) {
   askTrigger.addEventListener('click', openChat);
   askForm.addEventListener('submit', function (event) {
     event.preventDefault();
     var question = askInput.value.trim().slice(0, 300);
     if (!question) return;
     askInput.value = ''; answer(question); askInput.focus({ preventScroll: true });
   });
   askInput.addEventListener('input', function () { clearInterval(chatMotionTimer); emitSkyy('idle'); });
   function closeChat() { chatClosing = true; askDialog.close(); }
   askCancelBtn.addEventListener('click', closeChat);
   askDialog.addEventListener('cancel', function () { chatClosing = true; });
   askDialog.addEventListener('close', function () {
     chatClosing = false;
     clearInterval(chatMotionTimer);
     characterHome.appendChild(mascotEl);
     state = 'idle'; emitSkyy('idle');
     triggerBtn.setAttribute('aria-expanded', 'false');
     triggerBtn.focus({ preventScroll: true });
   });
 }
 window.skyyRoseConcierge = Object.freeze({ open: openChat });

	// -------------------------------------------------------------------------
	// Walk-On / Walk-Off
	// -------------------------------------------------------------------------

	function proactiveSpeak(promptId, text, chips) {
		if (wasPromptShown(promptId)) return false;
		markPromptShown(promptId);
		speak(text, chips);
		return true;
	}

	function walkOn(isProactive) {
		if (state === 'walking-in' || state === 'idle' || state === 'speaking') return;
		var threeReady = !!window.skyyRoseMascot3D;
		var walkDuration = prefersReducedMotion ? 0 : (threeReady ? 2200 : 1400);
		mascotEl.style.setProperty('--skyy-walk-duration', walkDuration + 'ms');
		state = 'walking-in';
		emitSkyy('walking-in');
		mascotEl.setAttribute('aria-hidden', 'false');
		mascotEl.classList.remove('skyyrose-mascot--hidden', 'skyyrose-mascot--exiting');
		mascotEl.classList.add('skyyrose-mascot--entering');

		setTimeout(function () {
			if (state !== 'walking-in') return;
			mascotEl.classList.remove('skyyrose-mascot--entering');
			mascotEl.classList.add('skyyrose-mascot--idle');
			state = 'idle';
			emitSkyy('idle');

			// First visit = one contextual nudge, never a tour. Greet once
			// per session; never resurface the same greeting after that.
			// A founder-authored page tip (Customizer, empty by default)
			// takes the place of the generic greeting line when configured —
			// that IS the contextual nudge for this page.
			if (isProactive && !sessionStorage.getItem(SESSION_KEY_GREETED)) {
				try {
					sessionStorage.setItem(SESSION_KEY_GREETED, '1');
				} catch (e) { /* degrade silently */ }
				var script = SCRIPTS[context] || SCRIPTS['default'];
				var greetingText = mascotConfig.pageTip ? mascotConfig.pageTip : script.greeting.text;
				// Wave first, speak after — emitting both synchronously makes
				// the talk clip instantly override the wave in the 3D layer.
				emitSkyy('wave');
				var waveDuration = window.skyyRoseMascot3D && window.skyyRoseMascot3D.getActionDuration
					? (window.skyyRoseMascot3D.getActionDuration('skyy_wave') * 1000 || 2800) : 0;
				greetTimer = setTimeout(function deliverGreeting() {
					greetTimer = null;
					// User may have acted during the gap — only greet an
					// idle, undismissed character.
					if (state !== 'idle' || isDismissedThisSession()) return;
					// Background tabs pause the mixer; wall-clock time alone
					// must not interrupt a greeting that is still playing.
					var bridge = window.skyyRoseMascot3D;
					if (document.hidden || (!prefersReducedMotion && bridge &&
						((bridge.isLoading && bridge.isLoading()) || bridge.getCurrentAction() === 'Skyy_Wave'))) {
						greetTimer = setTimeout(deliverGreeting, 200);
						return;
					}
					proactiveSpeak('greeting:' + context, greetingText, script.greeting.chips);
				}, prefersReducedMotion ? 0 : Math.max(1400, waveDuration + 150));
				recordProactiveAppearance();
			}
		}, walkDuration);
	}

	function walkOff(onDone) {
		clearGreetTimer();
		var threeReady = !!window.skyyRoseMascot3D;
		var exitDuration = prefersReducedMotion ? 0 : (threeReady ? 2200 : 600);
		mascotEl.style.setProperty('--skyy-exit-duration', exitDuration + 'ms');
		state = 'exiting';
		emitSkyy('exiting');
		dismissBubble();
		clearTimeout(idleTimer);
		mascotEl.classList.remove('skyyrose-mascot--idle', 'skyyrose-mascot--entering');
		mascotEl.classList.add('skyyrose-mascot--exiting');
		mascotEl.setAttribute('aria-hidden', 'true');

		clearTimeout(exitTimer);
		exitTimer = setTimeout(function () {
			if (state !== 'exiting') return;
			mascotEl.classList.add('skyyrose-mascot--hidden');
			mascotEl.classList.remove('skyyrose-mascot--exiting');
			state = 'dormant';
			// Stop the renderer once the walk-off completes; keeping this an
			// explicit state event avoids spending GPU time while hidden.
			emitSkyy('hidden');
			if (onDone) onDone();
		}, exitDuration);
	}

	// -------------------------------------------------------------------------
	// Entrance scheduling
	//
	// First appearance per session: fixed FIRST_ENTRY_DELAY_MS after load,
	// NOT reset by activity — she is the site host and greets arrivals.
	// Re-entries (after she has appeared at least once): only after the
	// visitor has gone quiet for a randomized 10–30s window, activity resets
	// that timer. Capped at MAX_PROACTIVE_APPEARANCES per session and skipped
	// entirely once the visitor has explicitly dismissed her this session.
	// -------------------------------------------------------------------------

	function scheduleIdleEntrance() {
		clearTimeout(idleTimer);
		if (isDismissedThisSession() || proactiveAppearancesExhausted()) {
			return;
		}
		// First appearance of the session: short fixed delay, host-greets-
		// arrival semantics. Re-entries: randomized idle window.
		var threshold = getAppearanceCount() === 0
			? FIRST_ENTRY_DELAY_MS
			: IDLE_TRIGGER_MIN_MS + Math.random() * (IDLE_TRIGGER_MAX_MS - IDLE_TRIGGER_MIN_MS);
		idleTimer = setTimeout(function () {
			if (state === 'dormant' && !isDismissedThisSession() && !proactiveAppearancesExhausted()) {
				walkOn(true);
			}
		}, threshold);
	}

	function onActivity() {
		var now = Date.now();
		if (now - lastActivityReset < ACTIVITY_THROTTLE_MS) return;
		lastActivityReset = now;
		// Re-arm only for idle-triggered RE-entries. The first-entrance timer
		// must survive scrolling/mouse movement — resetting it on activity
		// means engaged visitors never see her at all.
		if (state === 'dormant' && getAppearanceCount() > 0) {
			scheduleIdleEntrance();
		}
	}

	['scroll', 'pointerdown', 'pointermove', 'keydown', 'touchstart'].forEach(function (eventName) {
		window.addEventListener(eventName, onActivity, { passive: true });
	});

	// React when product grid enters viewport — a Skyy-initiated aside while
	// she's already present is not a new "proactive appearance."
	if ('IntersectionObserver' in window) {
		var productObserved = false;
		var productObs = new IntersectionObserver(function (entries) {
			entries.forEach(function (entry) {
				if (entry.isIntersecting && state === 'idle' && !productObserved) {
					productObserved = true;
					mascotEl.classList.add('skyy--glancing');
					setTimeout(function () {
						mascotEl.classList.remove('skyy--glancing');
					}, 2000);
					if (proactiveSpeak('product-glance', 'These just landed. 👀', null)) {
						emitSkyy('excited');
					}
				}
			});
		}, { threshold: 0.3 });

		document.querySelectorAll('.products, .wc-block-grid, .collection-hero').forEach(function (el) {
			productObs.observe(el);
		});
	}

	// React when hovering product cards. Throttled separately from the CSS
	// class toggle below — rapid multi-card hovering must not thrash the 3D
	// layer's clip-switching (skyy-3d.js listens for skyy:excited).
	var HOVER_EXCITE_THROTTLE_MS = 2000;
	var lastHoverExciteAt = 0;
	document.addEventListener('mouseover', function (e) {
		if (state !== 'idle') return;
		if (e.target.closest('.product, .wc-block-grid__product')) {
			mascotEl.classList.add('skyy--excited');
			setTimeout(function () {
				mascotEl.classList.remove('skyy--excited');
			}, 1200);

			var now = Date.now();
			if (now - lastHoverExciteAt > HOVER_EXCITE_THROTTLE_MS) {
				lastHoverExciteAt = now;
				emitSkyy('excited');
			}
		}
	});

	// -------------------------------------------------------------------------
	// Minimize / Recall
	// -------------------------------------------------------------------------

	function minimize(isDismissal) {
		if (isDismissal) {
			markDismissed();
		}
		walkOff(function () {
			recallBtn.style.display = 'flex';
			recallBtn.setAttribute('aria-hidden', 'false');
			// Dismissal must not strand keyboard focus on <body> — the recall
			// pill is the continuation control.
			recallBtn.focus();
		});
	}

	function recall() {
		recallBtn.style.display = 'none';
		recallBtn.setAttribute('aria-hidden', 'true');
		// User-initiated — doesn't count against the proactive-appearance cap
		// and doesn't re-greet (greeting already marked shown this session).
		walkOn(false);
	}

	// -------------------------------------------------------------------------
	// Event Listeners
	// -------------------------------------------------------------------------

	triggerBtn.addEventListener('click', function (e) {
		e.stopPropagation();
		if (!askDialog.open && (state === 'speaking' || state === 'idle')) openChat();
	});

	minimizeBtn.addEventListener('click', function (e) {
		e.stopPropagation();
		if (askDialog.open) askDialog.close();
		minimize(true);
	});

	recallBtn.addEventListener('click', function (e) {
		e.preventDefault();
		e.stopPropagation();
		recall();
	});

	// Close bubble on outside click
	document.addEventListener('click', function (e) {
		if (!askDialog.open && state === 'speaking' && !mascotEl.contains(e.target) && !recallBtn.contains(e.target)) {
			dismissBubble();
		}
	});

	// Keyboard accessibility
	document.addEventListener('keydown', function (e) {
		if (askDialog.open) return;
		if (e.key === 'Escape') {
			if (state === 'speaking') {
				dismissBubble();
				triggerBtn.focus();
			} else if (state === 'idle') {
				minimize(true);
			}
		}
	});

	// -------------------------------------------------------------------------
	// Initialize
	// -------------------------------------------------------------------------

	if (isDismissedThisSession()) {
		// Respect the earlier dismissal — surface only the recall pill, no
		// unsolicited walk-on for the rest of this session.
		recallBtn.style.display = 'flex';
		recallBtn.setAttribute('aria-hidden', 'false');
	} else {
		scheduleIdleEntrance();
	}
})();
