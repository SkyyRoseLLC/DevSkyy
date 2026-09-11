# Nine-scene finalization — local candidate

Source worktree: `/Users/theceo/.codex/worktrees/7116/DevSkyy`, branch `codex/v2-cinematic-ooda-20260906`, starting/ending HEAD `aabd2bffdc1e322862acd05a5640c14cf00f3acf` (uncommitted candidate). Baseline was the immutable theme served on 18417; current theme served on 18416. No assets, product cast, approval records, or Town Line source were changed. No deployment, promotion, paid calls, or new creative assets.

## Result

The 90-case individual scene browser matrix passed: nine scenes each had matched reduced-motion captures at 390/1440, normal playback/keyboard/CTA/handoff at 390/1440, Save-Data, no JavaScript, failed video, pending video, lifecycle/resize, and reverse/rapid scroll. Eighteen fresh review recordings show individual scene arrival, motion progression, native product CTA focus, and the native #shop handoff. Video and still identity are bound to the existing approved manifest; the separate 19-test authority/27-media-hash suite passed.

This is a scoped engineering and visual review result, not whole-theme certification. Real device, screen-reader, actual browser BFcache eligibility, and per-scene GPU/frame-cost certification remain unexercised here. Root owns cross-engine and route performance evidence.

## Changes

- `hero-composed-scene.php`: strict deferred approved poster URL/srcset, with a native noscript image and unchanged dimensions/alt/cast. A page no longer relies on the browser's broad native-lazy distance heuristic. The no-JS placeholder is hidden and the native fallback image remains visible.
- `collection-scene-motion.js`: prepare current/nearby poster at 240px approach; prepare one next poster when a scene becomes active. Only an actually visible scene (15% threshold) receives a video URL. Never prepare speculative next videos. Retain loaded source for reverse reuse. Return to the approved still on pause, preference fallback, offscreen stop and error. Disconnect both observers on pagehide and reobserve after pageshow. Expose per-scene state for verification; no perpetual animation loop.
- `functions.php` plus controller readiness flag: an event-only inline watchdog checks the exact external script load/error event and initialization readiness. If the controller is blocked or initialization throws, it restores approved poster URLs/srcsets. There is no timer and no fallback video request. `wp_add_inline_script` preserves WordPress nonce-hook compatibility; the current local CSP permits inline scripts via its existing unsafe-inline directive (it does not currently emit a nonce).
- `scene-handoff.js/css`: existing 280ms commerce resolution with canonical 40ms separation between heading and grid. Base content stays visible; reduced motion/Save-Data remain static.
- `scroll-world.php`: collection typography vocabulary on the below-fold Scroll World heading. `hero-composed-scene.php`: rare statement vocabulary on each existing chapter-three title. No critical title animation or invented copy.

## Strict-delivery evidence

Fresh mobile 390x900 contexts, load plus 800ms observation, baseline versus current; values below are encoded scene-poster bytes only, not total page transfers or Lighthouse results. Every current route requested zero scene videos before arrival and at most one near poster. All three original posters remain available when approached; no picture was removed.

| Route | Before posters / bytes | Current posters / bytes | Deferred bytes |
|---|---:|---:|---:|
| Home | 3 / 503,372 | 1 / 255,858 | 247,514 |
| Signature | 3 / 830,156 | 1 / 255,858 | 574,298 |
| Black Rose | 3 / 493,290 | 1 / 133,642 | 359,648 |
| Love Hurts | 3 / 541,486 | 1 / 113,872 | 427,614 |

Proof: `.artifacts/v2-cinematic-finalization-20260906/scenes/scene-delivery-before-current.json`. These timings establish discovery scheduling; they do not independently attribute Lighthouse LCP improvement. Root's trace finding that the initial collection video belonged to its hero, not the nine scene films, remains consistent with zero scene-video requests in these initial checks.

## Individual scene matrix and art direction

All rows below passed independently for: asset identity; 390/1440 containment; title/CTA presence; normal playback; pause/play; transition in/out; native commerce handoff; keyboard pause; reduced motion; Save-Data; no-JS; loading still; video-error still; reverse and rapid scroll; 390→900 landscape→390 resizing; and persisted page lifecycle handler simulation. No overflow or JS errors were recorded. Supplemental touch taps and document-hidden handler simulation are recorded separately in `touch-document-hidden.json` (must contain nine PASS entries to claim complete).

| Scene | Title | Engineering | Composition acceptance | Specific visual finding |
|---|---|---|---|---|
| SIG-COMMERCE-1 | The Golden Gate Overlook | PASS | EQUIVALENT | Full Sherpa/beanie silhouette, bridge and gold mark remain within the image; mobile retains the complete image above concise native destinations. |
| SIG-COMMERCE-2 | The Lateral Terrace | PASS | EQUIVALENT | Both silhouettes and their lateral spacing survive the narrow full-frame treatment; garment identity and three independent links remain legible. |
| SIG-COMMERCE-3 | The Departure Terrace | PASS | EQUIVALENT | Three-person composition and monogram remain intact. Native pre-order product identities extend below the mobile viewport in ordinary scroll; no combined-set promise was introduced. |
| BR-COMMERCE-1 | The Type Foundry | PASS | EQUIVALENT | Portrait remains full height, including the overhead circle and shoes. Narrow desktop portrait beside the next chapter is existing rail grammar. Initial blank QA capture was an async decode race, fixed by awaiting decode in capture tooling; final matched frames show the unchanged approved portrait. |
| BR-COMMERCE-2 | The Moonlit Waterfront | PASS | EQUIVALENT | Both models, silver lettering, moon and waterfront survive desktop and mobile containment. Title remains distinct from merchandise links. |
| BR-COMMERCE-3 | The Town Line | PASS | EQUIVALENT | Existing five-jersey room and its source are preserved. Mobile preserves the full room and individually sold links; this is preservation of the approved chapter, not new Town Line authoring. |
| LH-COMMERCE-1 | The Vow Aisle | PASS | EQUIVALENT | Foreground models and centered rose maintain the original dramatic aisle balance; three independent garment destinations remain explicit. |
| LH-COMMERCE-2 | The Rose Side Chapel | PASS | EQUIVALENT | Complete shorts silhouette stays at the original right-side scale within architectural negative space; the single native product CTA remains obvious below. |
| LH-COMMERCE-3 | The Rose Vitrine | PASS | EQUIVALENT | Portrait framing keeps the bag, hand and illuminated rose visible together; narrow composition remains deliberate rather than a landscape crop. |

“EQUIVALENT” is an integration judgment, not new approval of generated product fidelity. All mobile poster frames and all desktop motion frames were inspected eyes-on. The original nine compositions and media hashes were preserved. The choreography and return-to-still behavior are engineering improvements; founder judgment on the overall visual experience remains authoritative.

## Evidence and reproduction

Artifact directory: `.artifacts/v2-cinematic-finalization-20260906/scenes/`.

- `baseline-scene-matrix.json`, `current-scene-matrix.json`: per-scene cases and source URLs.
- `baseline-{scene-id}-{390|1440}-poster.png`, `current-{scene-id}-{390|1440}-poster.png`: matched, decoded still captures.
- `current-{scene-id}-{390|1440}-motion.png`: active scene frames.
- `{scene-id}-{390|1440}-arrival-progression-cta-handoff.webm`: all eighteen current recordings.
- `{scene-id}-{save-data|no-js|media-error|loading|lifecycle}.png`: exceptional-state images.
- `touch-document-hidden.json`: actual emulated touch pause/resume and synthetic document-hidden-handler proof per scene.

Use Node22 `/Users/theceo/.hermes/node/bin/node` and `V2_QA_PACKAGE=/Users/theceo/.codex/worktrees/19db/DevSkyy/.artifacts/v2-phase3-20260905/qa/package.json`.

Run `tools/v2-runtime/phase3b-browser/test-nine-scenes-finalization.cjs` with `V2_SCENE_PHASE=baseline V2_SCENE_BASE=http://127.0.0.1:18417` for reference captures and default current18416 for the complete matrix. Run `test-scene-delivery.cjs` for matched request timing and `test-scene-touch.cjs` for touch/document-hidden cases. These are explicit browser runners, outside unit wildcards.

## Boundaries requiring final review

- A successful `image.complete`/`naturalWidth` check alone did not guarantee painted output for the portrait poster. Capture now waits for `decode()`; the earlier blank images were replaced in both baseline/current sets.
- Back/forward lifecycle tests dispatch persisted page events. They validate cleanup/reconnection, not eligibility for actual BFcache in every browser.
- Orientation was emulated by changing viewport dimensions; no real device or screen reader was used. Touch proof is actual emulated taps, not physical hardware.
- GPU decode cost, per-scene long-frame budgets, full film loop seams, and all-engine motion need independent corroboration before an unrestricted cinematic certification claim.
- CLOSED: with JavaScript disabled, native noscript pictures work; with scripting enabled but the external controller blocked or initialization throwing, the event-only watchdog restores all three approved stills and starts no video. `controller-failure.json` records 12 PASS cases: Home plus core three collections, each in normal, network-failed, and initialization-failed mode. Normal cases retain at most one prepared near poster; the existing CSP permits the inline watchdog. No policy was loosened by this fix.

## Controller failure closure

`tools/v2-runtime/phase3b-browser/test-scene-controller-failure.cjs` passed all 12 cases after the shared build. Failure restoration waits for the external script load/error and, when needed, DOMContentLoaded so a deferred successful controller initializes first. This avoids arbitrary timeouts and preserves normal strict scheduling. The normal-delivery comparison is rerun after this fix; use the latest `scene-delivery-before-current.json`.
