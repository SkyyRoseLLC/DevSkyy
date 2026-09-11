# V2 readiness performance report

**Canonical status: NEEDS_MORE_WORK. Platform transport status: ENVIRONMENT-BOUND.** All six current simulated mobile LCP values exceed the unchanged 2500 ms budget. All six CLS values pass 0.1. Field INP and GPU execution certification are unavailable. This report does not authorize deployment or imply production performance.

## Source and method

Eighteen Lighthouse 13.4.1 runs completed successfully: six immutable-baseline, six current and six current-source gzip diagnostic routes. Profile: Lantern simulated mobile390×844, DPR1, CPU×4, RTT150ms, 1638.4Kbps; cold contexts and seeded synthetic commerce state. Single samples are not medians or statistical confidence. Baseline source is `baseline-theme/skyyrose-flagship-2`. All runs in this comparison follow the router source-isolation correction documented in `fixture-isolation.md`; earlier contaminated readiness ablations are not used as final performance comparisons.

Gzip changes text transport on the local fixture; it is not a theme-only gain or verification of WordPress.com/CDN behavior. Unthrottled observers through load+10seconds provide transfer/request/long-task gates separately. Lighthouse observed LCP breakdown subparts are from its recorded run and **must not be added to or subtracted from the Lantern simulated LCP**. Initial transfer ends at load; total at load+10seconds without interaction. LH total-byte-weight is a different window.

## Same-profile before / current / gzip

LCP milliseconds; KiB uses1024bytes. “Near” is descriptive only: PDP gzip2528ms still fails2500ms.

| Route | Baseline LCP | Current LCP | Delta | Gzip LCP | Gzip gate | Baseline KiB | Current KiB | Gzip KiB | Current CLS |
|---|---:|---:|---:|---:|---|---:|---:|---:|---:|
| home | 5878 | 5484 | -393 | 3697 | FAIL | 1591.6 | 1544.2 | 1222.1 | 0.05522 |
| shop | 7143 | 6025 | -1118 | 3329 | FAIL | 1093.9 | 1017.7 | 597.1 | 0.00054 |
| collection | 5338 | 4513 | -825 | 3692 | FAIL | 1638.6 | 1593.2 | 1251.7 | 0.00429 |
| pdp | 5506 | 5503 | -2 | 2528 | FAIL (near) | 1191.2 | 1155.3 | 648.4 | 0.00128 |
| cart | 5745 | 5730 | -15 | 2454 | PASS | 853.2 | 810.2 | 345.1 | 0.00061 |
| checkout | 4820 | 4518 | -302 | 2130 | PASS | 812.0 | 769.0 | 310.6 | 0.00000 |

The largest sampled LCP improvements are Shop and Collection; Home and Checkout also improve, while PDP and Cart are effectively unchanged at this sample precision. All six transfer totals decrease. The controlled source diff demonstrates payload savings, but these single LCP deltas do not isolate the causal contribution of each edit. Gzip meets the target only on Cart/Checkout and remains a separate diagnostic.

## Current Lighthouse observed subparts

These milliseconds describe observed insight subparts, **not addends of simulated LCP**. A dash is not applicable for text LCP rather than a measured zero image phase.

| Route / candidate | Document TTFB | Resource load delay (discovery/scheduling proxy) | Resource transfer duration | Element render delay |
|---|---:|---:|---:|---:|
| home / hero VIDEO | 485.6 | 189.1 | 2.8 | 71.7 |
| shop / portal frame IMG | 364.5 | 8.1 | 13.7 | 58.0 |
| collection / arrival IMG | 466.1 | 7.4 | 11.2 | 63.1 |
| pdp / native gallery IMG | 387.9 | 8.8 | 15.3 | 72.6 |
| cart / H1 | 343.9 | — | — | 66.2 |
| checkout / H1 | 254.4 | — | — | 70.1 |

Document TTFB is an end-to-end local fixture measurement, not a database/template/server CPU attribution. There is no Server-Timing decomposition. Resource-load delay combines discovery and scheduling; exact parser discovery is not separately proven. Render delay includes readiness/layout/paint/scheduling and is not equivalent to image decode time. Text H1 candidates do not have an independent image request or decode phase.

## Selected source changes and rejected experiments

The original full Inter secondary face is48,432bytes; the selected11-codepoint variable subset is1,872bytes, saving46,560body bytes where that fallback is fetched. The declared unicode-range keeps missing Hanken glyphs, including arrows, without substituting unrelated glyph shapes. Independent review reproduced outlines and advance widths at weights100/400/700/900. Current CSS retains primary Archivo/Hanken/Anton/Cinzel and an Inter fallback face; this is not a primary typography redesign. Actual downloads remain content-dependent. The first experiment simply removing Inter lost arrows and is quarantined/superseded, not selected.

Cart/Checkout omit unrelated legacy-world-components/content-page CSS (about44KB combined) while retaining theme/controls/shell and native Woo styling. Shop removes **only** native Woo layout and smallscreen CSS (about29KB), retaining general Woo CSS because removal of all three regressed pagination/Quick View. QV remains on PDP because native related/upsell cards can require it; variable PDP keeps its native variation authority. No global script-order/defer change was selected. Earlier native-defer and VP9 experiments were not repeated or adopted in this pass.

All22 reviewed visual comparisons are EQUIVALENT under the bounded review in `visual-freeze.md`:16 original pixel matches, four tiny arrow raster differences, and two desktop readiness/state mismatches closed with byte-identical matched-state captures. There is no claim that all original screenshots are pixel-identical. Protected visual/media hashes remain a separate preservation receipt.

## Route-specific critical path and remaining ownership

**Home — theme scheduling and host transport:** Hero VIDEO remains the Lighthouse LCP candidate; approved video bytes and composition were preserved. `visual-recovery.js` gates film activation on poster decode, visibility, motion/data preference and playback state. Initial image preloads target the responsive poster; films remain preload-none. Font reduction removes competition but does not erase film transfer. Shared render-blocking CSS/native script dependencies remain. A late video candidate is possible, but this report does not invent a video eligibility cause from historical IMG-versus-VIDEO differences. Do not transplant prior contaminated timings into this pass.

**Shop — theme discovery and native/style transport:** The founder portal frame is LCP, not just the product photo. Its eager/high-priority request remains. Native general Woo CSS stays required; only the tested layout/responsive sheets were excluded. Card frame payload, competition with critical CSS and scheduling remain candidates for further isolated tests. Do not replace or remove paid frame art.

**Collection — theme media/lifecycle and host transport:** The arrival IMG is LCP. Responsive preload and image dimensions remain; approved hero motion and scene films retain visibility/proximity-managed activation and reduced-motion/Save-Data compositions. Lower font competition helps wire cost but does not establish a complete causal explanation of the sampled LCP delta. Nine scene assets and composition are frozen.

**PDP — native Woo/gallery and host transport:** The authoritative responsive gallery IMG is LCP. Native gallery/lightbox and variation dependencies remain. Shared/native CSS, font delivery and native gallery visibility/layout are the remaining critical-path surface; QV is not assumed redundant due to related cards. Gzip2528ms is28ms over the target and remains FAIL. Product bytes are not fabricated or replaced to lower LCP.

**Cart — theme style scope, native Woo and host transport:** The H1 is LCP. Removing unrelated legacy/content CSS reduces transfer without requiring card/hero changes. Native cart forms/notices/coupons/totals and general styles remain. The uncompressed sample still fails badly, while gzip2454ms passes narrowly. This sensitivity supports host-encoding verification and further critical CSS work, not globally stripping native styles.

**Checkout — host transport and native commerce:** The H1 is LCP. Quiet-route CSS trimming retains billing/payment/native plugin dependencies. No scenes/video/character walk-in are introduced. Gzip2130ms passes; canonical uncompressed4518ms still fails. Keep personalized session HTML uncached by public edge rules, and do not defer native checkout scripts without a dedicated transaction regression.

## Fonts, CSS/JS, media, third parties and runtime limits

`artifactbundle-inventory.json` records every resource from all18 LH reports with exact report hashes, theme/native-Woo/Core ownership, wire/decoded sizes, priority and status. Source file hashes bind the selected subset and policy code. Disk body sizes, observed wire transfer and LH total-byte-weight are intentionally separate. Font-ready timing alone does not prove fonts block an image LCP. Do not preload all fonts; preserve existing matching responsive image preloads. CSS blocking and native dependency chains are verified resource classes, but Lighthouse estimated wasted durations overlap and must not be summed as certain savings.

A low sampled TBT is not field INP. Layout/paint/animation/GPU work is not inferred from transferred bytes. No fieldINP, GPU timing, physical-device thermal profile or full temporal scene certification is supplied by these traces. Browser memory and decoded texture estimates require the separate Ask Skyy evidence; initial zero model requests do not mean zero later model cost. Third-party claims must be limited to the local resource summaries; real payment/analytics/consent delivery can differ.

## Platform transport: ENVIRONMENT-BOUND

WordPress.com documents automatic Brotli for HTML/CSS/JS with gzip fallback. The hosting layer owns negotiated compression; the theme owns payload, dependency order, preload and route scope. A theme-level output compressor is not required by that platform promise. The promise is not evidence that these particular staging/production responses were measured. [Official compression documentation](https://developer.wordpress.com/docs/platform-features/storage/).

Site Accelerator documents static delivery for Core/Jetpack/Woo assets and public image optimization; it does not provide audio/video delivery. Verify actual custom-theme, font, film and model response URLs/headers rather than assuming the image CDN covers them. [Official Site Accelerator documentation](https://wordpress.com/support/site-accelerator-cdn/).

Edge/object caches are separate. Public edge eligibility and session state affect comparisons; no settings or cache were changed. Object cache is platform-managed; purge is troubleshooting, not a recurring optimization. [Official cache documentation](https://wordpress.com/support/clear-your-sites-cache/).

Read-only next platform proof should bind route/source/hash and record Content-Encoding under negotiated br/gzip/identity, Vary, Cache-Control, cache status/Age where available, response bytes and protocol for HTML/CSS/JS/font/image/film. Keep private/noindex staging policy and personalized Cart/Checkout authority. A local gzip pass neither authorizes hosting writes nor upgrades the canonical release status.

## Observer-gate closeout

Final `current-complete` observations contain all six routes. Checkout required a documented supplementary retry: the first seed setup timed out30seconds waiting for SG005 sizeM variation readiness, with no captured evidence establishing the cause. The separate same-helper/profile retry succeeded (observed H1 LCP228ms); no theme edit occurred between attempts. The first five rows remain in `current-observations.json`, the retry remains separately labeled, and `current-complete-observations.json` explicitly records each captureSource. This is a trace setup retry, not a performance improvement.


### Canonical observer payload and failures

All payload values are KiB; fonts are informational because no route font ceiling was specified. The canonical budget JSON is unchanged; the result file binds its SHA256.

| Route | Initial | Total | JS | CSS | Images | Video | Fonts | Requests | Remaining FAIL metrics |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| home | 1476.6 | 1538.9 | 211.9 | 147.1 | 261.5 | 661.1 | 159.7 | 45 | lcpMs, initialTransferKiB |
| shop | 1010.2 | 1012.4 | 193.7 | 216.4 | 297.7 | 0.0 | 159.7 | 40 | lcpMs |
| collection | 1587.8 | 1587.8 | 204.2 | 134.6 | 492.4 | 462.0 | 161.7 | 47 | lcpMs |
| pdp | 1121.2 | 1123.4 | 287.9 | 261.0 | 299.9 | 0.0 | 159.7 | 52 | lcpMs |
| cart | 806.1 | 806.1 | 259.4 | 246.9 | 39.9 | 0.0 | 134.2 | 37 | lcpMs, cssKiB |
| checkout | 763.7 | 768.9 | 258.9 | 227.0 | 5.7 | 0.0 | 140.6 | 32 | lcpMs, initialTransferKiB, jsKiB, cssKiB |

Shop CSS now passes230KiB. Cart CSS remains above240KiB; Checkout CSS above180KiB and JS above240KiB. Home/Checkout initial transfer still exceeds1400/700KiB. Every route passes total transfer, image/video ceilings, sampled request counts and sampled long-task limits. All six observed navigation windows contain zero >50ms long tasks, zero model requests and zero recognized external texture requests; these are measured navigation-window facts, not GPU or later Ask Skyy runtime results. Field INP and animationRenderWorkP95Ms remain UNVERIFIED on every route.

### Unthrottled baseline/current trace comparison

These observed values must remain separate from the simulated table. In particular Home, Collection, PDP and Cart observed LCP did not improve uniformly; single local traces do not establish a general latency guarantee.

| Route | Baseline observed LCP ms | Current observed LCP ms | Current TTFB ms | LCP tag | Font-ready ms |
|---|---:|---:|---:|---|---:|
| home | 664 | 724 | 542.1 | VIDEO | 683.4 |
| shop | 440 | 316 | 256.9 | IMG | 336.1 |
| collection | 544 | 604 | 511.9 | IMG | 646.8 |
| pdp | 488 | 500 | 415.9 | IMG | 530.4 |
| cart | 408 | 460 | 420.1 | H1 | 470.7 |
| checkout | 380 | 228 | 200.5 | H1 | 246.3 |

Current Home video-playing is702.2ms and VIDEO LCP724ms; Collection video-playing is654.2ms after arrival IMG LCP604ms. This records actual event/candidate order in this pass without inventing why a later frame does or does not become a candidate. Shop/PDP/Cart/Checkout font-ready occurs after their observed LCP; this disproves a blanket claim that waiting for all fonts defines every route LCP. It does not make font transfers cost-free.

### Resource composition, failures and remaining evidence

- home: CSS+JS wire split theme 198.8KiB, WP Core 119.3KiB, native Woo 40.8KiB; recorded HTTP>=400 responses 0; observer errors 0.
- shop: CSS+JS wire split native Woo 130.9KiB, theme 159.8KiB, WP Core 119.3KiB; recorded HTTP>=400 responses 0; observer errors 0.
- collection: CSS+JS wire split theme 178.7KiB, WP Core 119.3KiB, native Woo 40.8KiB; recorded HTTP>=400 responses 0; observer errors 0.
- pdp: CSS+JS wire split native Woo 264.9KiB, theme 164.7KiB, WP Core 119.3KiB; recorded HTTP>=400 responses 0; observer errors 0.
- cart: CSS+JS wire split native Woo 266.3KiB, theme 140.9KiB, WP Core 99.1KiB; recorded HTTP>=400 responses 0; observer errors 0.
- checkout: CSS+JS wire split native Woo 276.9KiB, theme 109.8KiB, WP Core 99.1KiB; recorded HTTP>=400 responses 0; observer errors 0.

All current Lighthouse route summaries report zero third-party requests. This is local-fixture scope only; platform payment/consent/analytics remains environment-bound. Native Woo sourcebuster/order-attribution are same-origin resources and must not be silently removed as presumed third-party overhead. No new layout/decode/GPU critical-path attribution is asserted from this resource inventory.

**Release recommendation:** retain the selected source savings and visual equivalence, preserve canonical FAIL status, and isolate next work between theme critical-path/route payload and real host encoding/cache/protocol proof. Do not relax2500ms or classify near-PDP gzip as passing. Field INP, physical-device/manual accessibility and GPU/runtime certification remain separate requirements.

## Explicit delivery responsibility map

| Layer | Required responsibility | Candidate action / verification boundary |
|---|---|---|
| THEME | Reduce payload by actual route ownership; preserve native dependency order; correct poster/image priority and dimensions; content-version asset URLs; keep personalized purchase state authoritative | Selected font subset and Shop/Cart/Checkout CSS trims are implemented. Keep the film, cards, scenes and native Woo systems. A PHP output compressor or public cache for cart/checkout is not added. |
| WEB SERVER (self-managed hosting) | Negotiate gzip/Brotli for eligible text and HTML; return correct MIME, Vary, encoding and transfer lengths; serve fonts/video/model files correctly; support video byte ranges | This is server configuration, outside theme PHP. Local gzip is a reproducible diagnostic only. WOFF2 and encoded image/video payloads are already compressed formats; text-compression savings must not be projected onto those files. |
| CDN | Use correct cache keys, respect version changes, negotiate encoding, preserve media/range behavior and bypass personalized/session-sensitive HTML | Verify actual custom-theme/font/film/model responses. Recommend long-lived immutable caching only when the complete cache key changes with asset bytes; do not assign immutable to mutable URLs or HTML. Existing content-version query strings require confirmation that the edge cache includes them. |
| WORDPRESS.COM PLATFORM | Own edge/object-cache behavior and managed Brotli/gzip delivery; determine private/public and session eligibility | Official expectations are described above. Measure actual response headers and protocol later against the authorized candidate. No platform cache setting, privacy state, staging file or deployment was changed in this pass. |

For HTML, use platform-managed eligibility and personalized-route bypass. For versioned static assets, inspect Cache-Control, Vary: Accept-Encoding, encoding, Age/cache status and version key behavior before assigning an immutable policy. For anonymous cached documents, compare cold/warm responses separately; do not combine them with authenticated Cart/Checkout measurements. No production header values or Brotli measurements are invented here.

The full font family/weight/style/body-size and declaration inventory is in [performance-audit.md](performance-audit.md); every actual route request is in [artifactbundle-inventory.json](artifactbundle-inventory.json). There are no new font preloads. Primary variable font families and weights remain intact. Archivo's source CSS says optional while Core's theme.json face says swap; this inherited duplicate declaration was not silently reclassified as an effective optional-face guarantee. Current Inter uses the bounded unicode-range subset; no whole-family deletion was accepted.

For the requested route labels, the gzip PDP is **NEAR PASS** within the explicitly declared2500–2750ms proximity band, while its hard2500ms metric gate remains **FAIL**. These labels do not alter the working budget.
