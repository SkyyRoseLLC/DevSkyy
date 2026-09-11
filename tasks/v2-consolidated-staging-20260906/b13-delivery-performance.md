# B13 — consolidated staging cinematic delivery

**PERFORMANCE: NEEDS_MORE_WORK.** Filesystem and feature parity passed independently; these delivery findings do not constitute production approval.

Candidate SHA-256: `a4ec431146f31d036d5547137d3274a693ccd3d072be6089126593f642d14f4b` (536 files). Fifteen completed browser sessions: six routes on each of two mobile profiles, one desktop Home control, and two Home re-entry/reload cases. B01–B12 network probes finished before these runs; all timed sessions were sequential.

| B13 result | Status |
|---|---|
| HOME HERO DELIVERY | FAIL |
| COLLECTION SCENE DELIVERY | PASS |
| PRODUCT CARD DELIVERY | PASS |
| QUICK VIEW DELIVERY | PASS |
| ASK SKYY DEFERRED DELIVERY | PASS |
| MOBILE CINEMATIC DELIVERY | FAIL |

PASS labels are bounded to the observed Chromium sessions, source identity and visual checks. Home fails because late LCP replacement occurred and a desktop startup CLS failure was reproduced. Mobile delivery is not certified: variable LCP, excluded early layout shifts and model latency remain; no physical-device or field-INP claim.

## Repeatable profiles and first-route observations

Mobile: 390×844 CSS pixels, DPR 2, touch/mobile emulation. Normal:CPU 1, no imposed network throttle. Moderate:CPU 2,100 ms latency,4 Mbps download/1 Mbps upload. Desktop control:1440×900,DPR 1,normal. Fresh browser contexts, normal cache behavior,1.5 s pre-navigation quiet period,12 s untouched observation after load (18 s desktop). One primary sample per route/profile, with separate Home controls; these are not field percentiles or statistical speed rankings. Normal being slower in a particular row reflects observed delivery variability, not an assertion that throttling improves performance.

| Route | Normal LCP ms | Moderate LCP ms | Normal hero first visible-frame callback ms | Moderate hero callback ms |
|---|---:|---:|---:|---:|
| home | 4220 | 824 | 4209 | 2429 |
| signature | 248 | 780 | 440 | 1938 |
| black-rose | 236 | 792 | 328 | 2015 |
| love-hurts | 284 | 840 | 507 | 2012 |
| shop | 3588 | 1256 | N/A | N/A |
| pdp | 340 | 3012 | N/A | N/A |

Desktop Home: LCP452ms, CLS0.84501953125 at391.9ms, with hadRecentInput=false. Mobile filtered CLS scores cannot clear startup stability: raw initial shifts as large as1.0 were marked hadRecentInput=true by the browser. Preserve both raw and filtered observations. The machine summary now distinguishes allEventSum (every recorded shift) from rawSum (sum after recent-input exclusion) and maxSessionWindow (the CLS calculation). Maximum initial long tasks were57ms on normal Home and up to105ms among moderate primary routes.

## Hero network timelines

All times below are milliseconds from navigation. Poster completion is network completion, not exact paint. Frame callback is compositor-submission evidence with viewport intersection, not a physical display measurement. Raw reports include loaded-visible poster observations, video canplay/playing events, all LCP candidates and every request.

| Profile/hero | Poster start/end | Film start | Film header delay | First frame | LCP |
|---|---:|---:|---:|---:|---:|
| moderate/black-rose | 174.5/622.1 | 793.9 | 974.3 | 2015 | 792 |
| moderate/home-reentry | 267.9/576.5 | 821.7 | 34.6 | 2475 | 832 |
| moderate/home | 240.1/560.3 | 824.7 | 26.2 | 2429 | 824 |
| moderate/love-hurts | 178.6/775.2 | 794 | 34.4 | 2012 | 840 |
| moderate/signature | 186.4/669.7 | 802.2 | 24.1 | 1938 | 780 |
| normal/black-rose | 81.1/151.8 | 224.6 | 29 | 328 | 236 |
| normal/home-desktop | 133.7/210 | 427.4 | 25.4 | 574 | 452 |
| normal/home-reentry | 140.3/208.5 | 419 | 42.4 | 595 | 456 |
| normal/home | 207.9/245.3 | 1765 | 2257.7 | 4209 | 4220 |
| normal/love-hurts | 130.7/204 | 319.7 | 87 | 507 | 284 |
| normal/signature | 84.2/202.1 | 348.5 | 47.6 | 440 | 248 |

## Collection film scheduling and range/playback

All six primary collection sessions recorded zero scene-film requests before the untouched observation cut. Each loaded its three approved MP4s only during scene approach; reverse navigation and re-entry used the already loaded films. The browser action sequence was1→2→3→2→1, with pause/resume on Scene1. No arbitrary seek control was invented; seeking is not exposed by these scene controls. B07 separately verified exact206byte slices and416out-of-range responses for sampled MP4/WebM assets, including the rotating mark.

| Profile/collection | Untouched cutoff ms | Scene film start ms, in order | Unique scene films / requests |
|---|---:|---|---:|
| moderate/black-rose | 13991 | 14259, 19673.9, 23405.8 | 3/3 |
| moderate/love-hurts | 14172 | 14467.1, 19770.9, 23486.9 | 3/3 |
| moderate/signature | 14721 | 15031.7, 20382.1, 24162.2 | 3/3 |
| normal/black-rose | 12325 | 12564.9, 16640.8, 19047.5 | 3/3 |
| normal/love-hurts | 12539 | 12794.3, 16779.2, 19210.4 | 3/3 |
| normal/signature | 12444 | 12700.3, 16703.3, 19159.9 | 3/3 |

## Product cards and native Quick View

The observed BR-003 product derivative is480×720 for approximately236CSS pixels atDPR 2. The decorative card frame uses its640w derivative at approximately358CSS pixels. Native lazy-loading governs later cards; two initial card slots remain eager by the accepted implementation. Raw snapshots distinguish visible and below-fold elements and include decoded bitmap dimensions. These are theme derivatives; no new CDN image transformation was inferred.

Quick View issues no catalogue-wide PDP requests on untouched Shop. After intent, document speculation produces a Prefetch and the native controller produces the real PDP Fetch for the selected product. Native WooCommerce POST form, variation behavior, loaded media and close/reopen behavior passed the separate browser suite. The two request records are not proof of double wire transfer: the Prefetch reports fromPrefetchCache and zero encoded data events. A future bounded integration test can assess excluding Quick View triggers from speculative prefetch while preserving the native form architecture.

## Ask Skyy lifecycle and cache layers

All untouched Home sessions had zero GLB/Three/Draco requests. The static character remained available before intent. The four intent sessions then loaded the existing model and renderer, followed by first draw, walk-in, idle and chat UI. The desktop Home control remained untouched. No chat message was submitted.

| Profile/case | Intent→first draw ms | Model request encoded bytes / browser cache |
|---|---:|---|
| moderate/home-reentry | 15795.6 | 6065496 / False; 0 / True |
| moderate/home | 15224.1 | 6065496 / False |
| normal/home-reentry | 2309.2 | 6065510 / False; 0 / True |
| normal/home | 3733.8 | 6065511 / False |

The canonical GLB remains6,058,568bytes; CDP encoded totals include transfer accounting beyond that decoded file size. Same-document reopen made no second model request. Reload produced a model request explicitly marked browser-cache reuse with zero encoded transfer in both re-entry profiles. Exposed CDN HIT/REVALIDATED headers are recorded separately and do not establish global CDN cache correctness. Physical GPU memory, display timing, physical-device performance and field INP remain environment-bound.

## Causal bottlenecks and ownership

- **THEME / WORDPRESS.COM integration:** desktop header/hero geometry changes from y0/h900 to y76/h824 after initial paint. Final theme rules match this exact change; deferred full CSS and missing critical geometry are the strongest explanation. The full CSS arrived promptly in this sample. Test critical coverage and application order next; preserve all approved animation.
- **THEME / WORDPRESS / WORDPRESS.COM / UNKNOWN:** normal Home recovery code downloads early but is emitted after classic blocking jQuery Migrate/wp-util, whose delivery finishes around 1661–1730ms. Hero request begins1765ms. The controller observes visibility and poster decode; it does not wait for fonts. Resolve dependency/optimizer ordering ownership with a controlled test before changing theme code.
- **CDN / NETWORK / UNKNOWN:** that hero request then waits2258 ms for response headers despite a HIT label. This is measured delivery delay, not proof of PHP execution. PosterLCP468ms is replaced by videoLCP4220ms. Other samples differ, so retain the exact request evidence.
- **NETWORK:** normal Shop spends about 2993 ms in the secure-connection interval. Its card frame is discovered promptly after HTML. Do not treat that sample as evidence of late card image discovery.
- **MEDIA ASSET / NETWORK / CPU:** the unchanged large Ask Skyy model requires roughly15–16 s from intent to first draw on the moderate profile. Website deferral works; the separate fidelity-preserving Blender phase remains necessary.
- **WORDPRESS/PHP / WORDPRESS.COM / UNKNOWN:** anonymous Cart median TTFB1846.6ms across five uncontended final requests; origin sub-timings were unavailable. Public-route medians were about75–90ms.
- **WORDPRESS.COM / configuration:** strict font/GLB MIME targets fail, and platform Likes/sharing controls introduce an About footer strip. PDP records two payment Permissions-Policy console denials per profile; no thrown page error or payment submission occurred. Preserve these certification boundaries.

The rotating header WebM begins after hero loadeddata/window load in the delayed Home sample and after the desktop layout shift. It cannot explain the earlier delays. Independent component tests verified its geometry and deferred footer behavior; do not remove the approved animated identity.

## Evidence and limits

- Full machine timelines: `.artifacts/v2-consolidated-staging-20260906/b13/delivery-summary.json` and 15 raw session JSON files.
- Source/causal review: `b13-causal-notes.md`. Platform B01–B12: `platform-delivery.md`. Functional/visual checks: `feature-parity.md` and `about-logo-verification.md`.
- No theme optimization, platform configuration change, manual cache purge, model edit, product/media authority edit, order/payment submission or production deployment followed these measurements.
- Populated Checkout, authenticated/cart-personalization isolation, physical devices, field INP and payment sandbox certification remain unverified.

## Runtime model-stage measurements

These runtime-reported values supplement the independently observed first draw; their time origin is runtime initialization, not navigation.

| Primary Home profile | Runtime first stable frame ms | Model fetch ms | Model decode ms |
|---|---:|---:|---:|
| normal | 3590.3 | 1557.7 | 1408.2 |
| moderate | 15041.4 | 12253.2 | 1678.5 |
