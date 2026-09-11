# B13 — actual staging cinematic delivery

**DIAGNOSTIC COMPLETE — NOT CERTIFIED.** The actual deployed candidate is **STAGING-OBSERVED-f2893d998356**. It differs from the accepted local build. This read-only analysis applied no source, configuration, cache, order or payment change.

| Required outcome | Result | Observed reason |
| --- | --- | --- |
| HOME HERO DELIVERY | FAIL | The deployed Home has a static Black Rose bridge image and no requested animated Home hero. Town Line is a separate preserved previsualization and was never played. Fast static-image LCP cannot certify the absent film. |
| COLLECTION SCENE DELIVERY | PASS | Scoped browser delivery pass: all three collections initially request zero scene films; scenes 1–3 load and submit visible frames on approach, pause/resume works, and reverse reentry makes no additional scene-film request. Arrival-hero speed is reported separately and fails the mobile performance target. No accepted-local art parity, arbitrary-seek, or physical-display certificate is implied. |
| PRODUCT CARD DELIVERY | FAIL | Shop serves full 1024×1536 on-model card images without srcset/sizes for approximately 168×224 CSS-pixel cards at DPR 2. BR003 alone is about 222 kB encoded. Initial visible and below-fold images are inventoried; no responsive mobile derivative is selected for the tested card. |
| QUICK VIEW DELIVERY | FAIL | Actual QV is a legacy card-data preview. It opens and closes correctly, but does not request a PDP or contain a native cart/variation form. The required intent→PDP→native-form path and in-flight abort/stale-response behavior cannot be exercised because that path is absent. |
| ASK SKYY DEFERRED DELIVERY | FAIL | Mobile untouched navigation defers GLB/Three/Draco, but desktop untouched Home auto-starts them after the staged entrance timer. Model transfer is 6,058,568 payload bytes. Moderate mobile intent→first WebGL draw is 14.8 s. Exact first stable frame and seamless poster/walk continuity are environment-bound for this deployed runtime. |
| MOBILE CINEMATIC DELIVERY | FAIL | Moderate mobile arrival-hero LCP is 4.388 s Signature,5.552 s Black Rose,3.708 s Love Hurts; corrected PDP LCP 6.264 s. Home CLS 0.1362 exceeds0.1. Normal results are faster, but do not clear the moderate profile or absent-feature gates. |

## Evidence and measurement scope

The harness ran twelve mobile route/profile observations, one untouched desktop Home probe, four corrected Shop/PDP observations, and one bounded Home reuse follow-up. The earlier pilot is preserved separately. Normal mobile uses 390×844/DPR 2 with no CPU/network slowdown. Moderate uses the same viewport,2× CPU slowdown, configured 100 ms latency,4 Mbps download and 1 Mbps upload. Chromium 145.0.7632.6 was launched with ANGLE Metal. These are single lab observations, not physical-device, field or canonical Lighthouse measurements.

Contexts were anonymous and fresh per route. No HTTP cache disabling or request interception was used. The untouched mobile interval lasts 12 s after window load; desktop lasts 18 s. The initial measurement cutoff precedes interaction. Passive analytics/Stripe traffic is included. Each collection run also makes one automatic WooCommerce POST to `?wc-ajax=get_refreshed_fragments` (200); the diagnostic is not described as GET-only traffic. No add-to-cart, purchase or payment control was activated. Only the actual verified local-only Skyy shipping path was submitted.

The original observer rejected text-node shift sources on Shop/PDP. Those four original JSONs remain intact, but their metric rows are superseded by `observer-corrected/`. The observer correction does not change the page. A similar normal Home observer error was after user intent, outside its initial metric window; the corrected reuse follow-up covers the later lifecycle. Deterministic CLS tests cover session gaps, recent input exclusion and five-second splitting.

## Mobile observations

| Profile | Route | HTML TTFB ms | LCP ms | CLS max window | Initial max long task ms | Hero viewport frame ms |
| --- | --- | --- | --- | --- | --- | --- |
| moderate | home | 67.4 | 808 | 0.136197 | 57 | Absent / not applicable |
| moderate | signature | 190.3 | 4388 | 0.000661 | 62 | 4395.4 |
| moderate | black-rose | 90.5 | 5552 | 0.000661 | 81 | 5542.2 |
| moderate | love-hurts | 73.5 | 3708 | 0.000661 | 67 | 3715.2 |
| moderate | shop | 68.9 | 1320 | 0.002722 | 0 | Absent / not applicable |
| moderate | pdp | 138.5 | 6264 | 0.031939 | 58 | Absent / not applicable |
| normal | home | 128 | 400 | 0 | 51 | Absent / not applicable |
| normal | signature | 75.1 | 784 | 0 | 0 | 742.6 |
| normal | black-rose | 61.7 | 436 | 0 | 0 | 426.1 |
| normal | love-hurts | 83.7 | 436 | 0 | 0 | 441.6 |
| normal | shop | 80 | 300 | 0 | 0 | Absent / not applicable |
| normal | pdp | 464.2 | 1724 | 0.000113 | 0 | Absent / not applicable |

CLS is the maximum session-window score. Home moderate raw no-input shift sum is 0.25198, but correct CLS is 0.13620. Normal/corrected Shop and PDP values are single later observations, not chosen best-of-run values. Desktop Home is separate: H1 LCP 268 ms, CLS 0.01766, and an unsolicited3D startup long task 144 ms during its untouched18-second interval.

## Home and arrival-hero delivery

Home’s static Black Rose bridge IMG is the mobile LCP candidate (400 ms normal;808 ms moderate). No requested animated Home hero exists in the deployed DOM. The only Home video is preserved Town Line below the page, left unplayed and excluded from the hero result. `homeTimelines` in the JSON contains every referenced stylesheet/script/font/hero/mascot request start, response-header time, completion, transferred-byte counter and decoded JS/CSS SHA. These are delivered resources; an external stylesheet is not assumed to be wholly render-critical merely because it is referenced. Inline critical CSS has no independent network request and no separately isolated execution cost here.

| Profile | Collection | Poster start→end ms | Film request start ms | Can play ms | First viewport frame ms | LCP ms |
| --- | --- | --- | --- | --- | --- | --- |
| moderate | signature | 286.8→1768.7 | 1628.8; 7596.2; 8596.5 | 4355.5 | 4395.4 | 4388 |
| moderate | black-rose | 184.5→2647.9 | 1783.1 | 5518.8 | 5542.2 | 5552 |
| moderate | love-hurts | 167.3→3457.9 | 1498.7; 7431.9 | 3680.1 | 3715.2 | 3708 |
| normal | signature | 84.7→197.7 | 381.8; 1226.4 | 666.6 | 742.6 | 784 |
| normal | black-rose | 69.1→239.3 | 336; 443.1 | 382.6 | 426.1 | 436 |
| normal | love-hurts | 93.2→228.8 | 243.3; 458.4; 1292.7; 1959.5; 3959.5 | 407.3 | 441.6 | 436 |

The collection hero’s viewport compositor callback and VIDEO LCP occur near each other. Moderate staging therefore reproduces a late animated-hero LCP behavior, but not the same source/fixture as the accepted local candidate. Request start→headers→canplay separates delivery from decode/playability; it does not isolate every cause. Poster resource completion and loaded/visible observations are available, but no exact physical poster-paint timestamp is claimed.

Home moderate hero-copy layout shifts at 1.5465 s,2.4384 s and 4.3843 s closely follow Anton1.538 s, Hanken2.420 s and Archivo4.376 s font completion. Font swaps/layout changes are a supported inference, not a fully isolated mutation trace. The first two shifts form CLS 0.1362; the later shift is a separate window.

## Scene demand loading, ranges and reentry

All six collection runs initially request zero of the three scene films. Each film submits a viewport frame after approach; only current/near scenes activate. Real pause and resume controls were exercised for scene 1, followed by 3→2→1 reverse reentry. No additional scene-film request occurs during reverse reentry. One request per scene/route/profile is recorded, with 206 responses. Approaching the pause control can scroll the horizontal carousel enough to briefly activate adjacent scene 2; that is recorded as near-scene demand loading, not an initial eager fetch.

| Profile | Collection | Scene1 loadstart→frame ms | Scene2 loadstart→frame ms | Scene3 loadstart→frame ms | Scene requests total |
| --- | --- | --- | --- | --- | --- |
| moderate | signature | 624.2 | 1459.3 | 613.4 | 3 |
| moderate | black-rose | 341.6 | 1490.9 | 611.3 | 3 |
| moderate | love-hurts | 1744.9 | 1476.4 | 329 | 3 |
| normal | signature | 80.7 | 161.1 | 181.2 | 3 |
| normal | black-rose | 64 | 328.1 | 64 | 3 |
| normal | love-hurts | 297.7 | 78.1 | 113.8 | 3 |

B07 independently verified exact start/nonzero206 slices and 416 behavior on Signature hero WebM/MP4 and scene 1 MP4. Browser playback selects WebM arrivals and 720p MP4 scenes. The browser records bytes=0- requests and, for some hero continuations, nonzero ranges/browser-cache responses. ERR_ABORTED can reflect browser media range cancellation and is not alone a playback failure. Incomplete CDP counters are retained as unknown/lower-bound, never counted as zero bytes. No arbitrary seek was manufactured because the tested experience exposes pause/resume and scene navigation, not a scrubber.

## Cards, PDP and Quick View

Shop’s BR003 card decodes 1024×1536, with no srcset/sizes or image-CDN transformation, while rendered about 168×224 CSS pixels at DPR 2. Its encoded delivery is about 222 kB. Other initial and below-fold card requests are inventoried; a one-viewport scroll requests the next lazy card. The required mobile responsive-candidate gate fails even though images load.

The actual browser PDP contains `.wp-post-image` and `.woocommerce-product-gallery__image .zoomImg`, contrary to the earlier prerequisite response’s gallery-markup absence. The main gallery currentSrc is the i0.wp.com SG005 `?w=1024&ssl=1` transformation. Browser naturalWidth 390 is density-corrected; createImageBitmap confirms 1024×1536 decoded pixels. The separate zoom image is also1024×1536 and opacity 0 until used. Geometry-based inventories also include hidden menu images; those are not claimed as painted. The cause of managed HTML/optimizer response variation is unknown.

PDP moderate image load is 4.673 s and LCP 6.264 s:1.591 s of additional render delay is observed. Concurrent fonts, related-card images and Stripe delivery are present; this trace does not prove which exclusively caused the paint delay. Payment permissions-policy console warnings occur without any payment action.

QV opens the BR003 card-data image/text preview and Escape closes it. It has no native cart/variation form, and no PDP GET before or after QV intent. The unchanged hidden text after close is not a stale network update: no native network path exists. Required abort/stale-response handling is therefore **NOT EXERCISED / FEATURE ABSENT**, not passed. Screenshots show the actual responsive preview, with lower content extending below the viewport.

## Ask Skyy lifecycle and cache

Mobile untouched Home makes no GLB/Three/Draco request during the12 s-after-load observation. The original portrait resource loads early, but neither portrait is observed in the initial viewport; the rendered recall control is reached by Playwright’s normal locator scrolling near the footer. Activated character/chat screenshots therefore do not establish initial Home placement.

Normal mobile recall→model-ready is 1466.4 ms and recall→first WebGL draw1571.2 ms. Moderate is 14628.9 ms and 14800.6 ms. The moderate GLB transfer alone lasts 12221.6 ms and reports6,065,498 CDP encoded bytes for6,058,568 payload bytes, despite x-ac HIT. Three, GLTF and Draco loader modules are fetched directly from jsDelivr before GLB loading; wrapper/WASM requests follow. The model has no separate network texture request in the observed chain; embedded texture costs are within the model payload and runtime decode.

Desktop Home proves the overall intent gate fails: without user interaction, walking-in fires12.938 s, modules start12.955 s, GLB starts13.209 s and first draw occurs14.383 s. The staged theme timer causes this initiation; cache/CDN does not. In the two primary mobile observations, the CSS idle event occurs before model-ready, so it cannot prove a rendered skeletal walk-in. The runtime exposes model-ready but no authoritative first-stable-frame signal. First draw and post-draw RAF remain distinct from stable silhouette/pose continuity. Actual rendered white-outfit Skyy identity is visible; exact local-poster parity is not certified.

Chat uses the deployed mascot bundle SHA 8a157dc8076b1738e943ff6bd579d503f48615b1ed95998be483056725e3d70e and its absent-configuration llmEnabled=false default. The native question dialog opens; local shipping submission closes it and shows the fallback guide response. No external concierge request is submitted; this is not proof of useful shipping-policy answers.

First-navigation minimize→recall returns ready idle with one model request. The same-context reload then times out waiting for model-ready, with no second GLB request; **cross-navigation model cache reuse is ENVIRONMENT-BOUND**. The runtime API exists but stays not ready. Source suggests the persisted dismissal guard misses the walking-in event fired before lazy runtime injection completes. This remains a source-supported lifecycle hypothesis, not a CDN/cache failure. `home-reuse/epoch-summary.json` keeps the two navigation epochs separate, with their own performance clocks and request offsets. No second-navigation snapshot is used to replace the first navigation’s events.

## Representation identity and causal ownership

Browser Signature CSS `/_jb_static/??a6acf50b00` is Brotli452031 decoded bytes, SHA 80875d248f2653177fc53977ea8bca01a6407044373b5752f54440dc6de51bf5. Browser PDP JS `??5a8acea822` is Brotli30679 decoded bytes, SHA 3b271f738f88447bbdf6571fe788cc0070afc3fb64ae0b9044fbce2b448a6102. Both exactly equal B01 identity/gzip bodies. They are different URLs from the old B01 Brotli failures (`??c9711b8a62`, `??955c29654e`). B01/B03 remain valid exact-URL defects; those stale payloads did not demonstrably cause these browser visuals or delays. See `browser-prerequisite-representation-comparison.json`.

| Owner | Evidence and limit |
| --- | --- |
| THEME | Missing Home animation/native QV, desktop automatic Skyy schedule, no responsive tested card candidates; deployed version gap. |
| MEDIA ASSET | 6,058,568-byte GLB and 1024×1536 card decodes; hero/scene payload sizes preserved in ranges and requests. |
| WORDPRESS/PHP | Network-inclusive document timing only; PHP execution cannot be isolated by these browser traces. |
| WORDPRESS.COM | Managed combined asset path and headers observed; exact stale-representation root cause not isolated. |
| CDN | x-ac HIT/MISS and i0.wp.com image transformations recorded separately from browser cache. A HIT does not eliminate client transfer/decode time. |
| NETWORK | Moderate profile causally imposes4 Mbps/100 ms; model transfer12.22 s aligns with 6.06MB payload. This is configured lab behavior, not a claim about all visitors. |
| THIRD PARTY | jsDelivr Three/GLTF/Draco imports are on the activated Skyy dependency chain. Passive WordPress analytics and PDP Stripe resources are present; no request blocking was used. |
| UNKNOWN | Exact PHP/edge contribution, precise PDP image-load→paint delay cause, physical first paint and first stable3D frame remain unresolved. |

Browser cache indicators and x-ac/age/server-timing headers are separate observations. A cached response’s copied edge header does not prove a fresh edge trip. No network-inclusive document TTFB is relabeled origin PHP time. No PHP/platform/cache fix was applied.

## Artifacts and remaining boundaries

`b13-report.json` contains all six decisions, mobile rows, desktop lifecycle, hero/scene timelines, exact representations and causal owners. `.artifacts/v2-a2-b13-20260906/b13/delivery-summary.json` contains selected request waterfalls, raw-stage summaries and card data; per-run JSONs retain detailed CDP events and snapshots. `review.html` links arrival/scene/card/QV/Skyy screenshots from the actual deployed candidate. The original pilot and superseded observer runs remain explicitly separate.

Remaining environment-bound gates: physical poster/video display paint, authoritative first stable Skyy frame, actual skeletal walk continuity, physical-device/GPU cost, field INP, origin-only PHP timing, protected-session privacy and platform cache root cause. No broad visual certification or accepted-local parity follows from this diagnostic pass.
