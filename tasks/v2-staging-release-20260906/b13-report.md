# B13 — approved staging cinematic delivery

**Completed: 17 sessions, including 12 primary mobile observations. Direct desktop Home CLS is 0.846951; cold moderate Ask Skyy activation is 14.2 seconds. Both remain performance failures.** All owned browsers are closed. No theme source, configuration, cache setting or customer state was changed.

The measured release is `e47588205c55e6303588a207d7ee766ad175b82f606d167198f4ec2c25b1d80d`, delivered from the authorized staging origin after ordinary managed-cache revalidation and root's filesystem/browser parity checks. The earlier old-deployment B13 evidence and stale-page pilot remain preserved separately. They are not used as the current candidate's results.

| Required outcome | Status | Scope and evidence |
|---|---|---|
| HOME HERO DELIVERY | **FAIL** | Approved animation and deferred optional 3D verified, but direct desktop B13 CLS 0.846951 fails stability; further mobile evidence: primary and quiescent B13 entries of1.0 were recent-input-excluded, while independent Lighthouse measured actual CLS1.0. No unqualified Home delivery PASS. |
| COLLECTION SCENE DELIVERY | **PASS** | Three collections × two profiles; allnine scene sources activate on approach, offscreen pause and reverse/re-entry; native pause controls verified separately in current browser-parity evidence. |
| PRODUCT CARD DELIVERY | **PASS** | Representative Shop initial/below-fold responsive delivery; first card480×720 decoded at236.3CSSpx/DPR 2. Does not certify all 33SKU product fidelity. |
| QUICK VIEW DELIVERY | **PASS** | Intent-only native form GET, correct image, Escape close; moderate settled-media supplement passes. Current browser-parity evidence covers variation and rapid reopen. No purchase submitted. |
| ASK SKYY DEFERRED DELIVERY | **PASS** | Deferred-delivery criterion passes: untouched mobile/desktop/reload load no optional3D; explicit intent activates; same-document reuse and actual disk-cache reload verified. Separate asset and activation gates fail:6,058,568B model exceeds2.5MiB, moderate stableframe14.216s exceeds2s. |
| MOBILE CINEMATIC DELIVERY | **FAIL** | Original sampled LCP0.224–2.372s, but Home large recent-input-excluded shifts prevent stability clearance; independent Lighthouse actual CLS1.0 and simulated LCP5.434s. Moderate cold Skyy14.2s remains another performance failure. |

## Home layout-shift qualification

**Direct B13 failure: untouched desktop Home has CLS 0.846951.** Its largest shift is 0.840972 at 306.7ms, followed by 0.003020 and 0.002959; all three have `hadRecentInput=false`. This is independent B13 evidence of unacceptable Home instability, not merely a Lighthouse disagreement or mobile metric-filter issue.

Both original Home samples contain a layout-shift entry of **1.0** marked `hadRecentInput=true`: normal at316.8ms and moderate at645.5ms. Standard CLS excludes those entries. Therefore their reported CLS values of0 and0.0310 are correctly calculated for the recorded flags, but **do not establish that Home layout stability is resolved**. No other primary route has a recent-input-excluded shift≥0.1.

Two bounded controls waited1500ms on about:blank before navigation, with no interaction until the12-second post-load snapshot. The large shift persisted and remained marked as recent-input-related: normal1.0 at409.2ms; moderate1.0 at622.3ms. Their computed CLS remained0 and0.0310. This does not support the simple hypothesis that insufficient idle time after new-page setup explains the exclusion. The cause of the flag remains **UNKNOWN**.

Root's independent Lighthouse Home run at21:31:02 UTC records **actual observed CLS1.0**, simulated LCP5434ms and server response time1937ms. That is a separate loading/throttling setup, not a replacement value for the B13 rows below. The full excluded-shift nodes/rectangles, both quiet controls and the exact Lighthouse report are bound in the JSON companion. Home and mobile delivery receive FAIL rather than an unqualified PASS based on discounted B13 CLS.

## Initial route measurements

Chromium 145.0.7632.6; mobile 390×844 CSSpx, DPR 2, touch emulation. Normal is CPU 1× and unthrottled. Moderate is CPU 2×, configured100ms latency,4Mbps download/1Mbps upload through CDP. The configuration is recorded; it does not guarantee every NavigationTiming responseStart will exceed100ms. Browser caching stays enabled, with a fresh context per primary run. There is no interception or artificial request blocking.

The table freezes each route before interaction,12seconds after load. Times are milliseconds from navigation. CLS uses the maximum session-window sum, excluding recent-input shifts; raw sums are separate diagnostics. Zero long task means none recorded above the observer threshold. Film callbacks are distinct from LCP and physical-display paint.

| Profile | Route | Document TTFB | LCP | Recorded CLS† | Max long task | First viewport film callback |
|---|---|---:|---:|---:|---:|---:|
| normal | home | 104.9 | 384 | 0.0000 | 62 | 542 |
| normal | signature | 71.7 | 296 | 0.0000 | 0 | 389 |
| normal | black-rose | 285.2 | 568 | 0.0000 | 0 | 676 |
| normal | love-hurts | 169.6 | 432 | 0.0000 | 0 | 556 |
| normal | shop | 64.5 | 224 | 0.0000 | 0 | N/A |
| normal | pdp | 71.2 | 252 | 0.0000 | 0 | N/A |
| moderate | home | 100.7 | 672 | 0.0310 | 0 | 2328 |
| moderate | signature | 63.3 | 684 | 0.0003 | 0 | 1991 |
| moderate | black-rose | 64.5 | 680 | 0.0000 | 0 | 1605 |
| moderate | love-hurts | 109.4 | 992 | 0.0003 | 0 | 2047 |
| moderate | shop | 107.7 | 1556 | 0.0000 | 0 | N/A |
| moderate | pdp | 152.7 | 2372 | 0.0013 | 0 | N/A |

† Read Home values with the large excluded-shift qualification above. These are single diagnostic samples, not field percentiles or Lighthouse results. All original sampled LCP values are≤2.372s, but this does not clear Home's layout instability or the14.2-second cold character activation. The quiet Home controls report LCP464/660ms and viewport film callbacks554/2316ms for normal/moderate respectively.

Desktop Home is separate:1440×900 CSSpx, DPR 1,18seconds untouched. It reports LCP 380ms, CLS **0.846951**, a viewport film callback at 473ms and zero optional 3D requests until intent. It is not mixed into the mobile table.

## Hero and scene delivery

`b13/delivery-summary.json` retains request start, first byte, completion, encoded bytes, coding, cache headers, decoded code hashes, all LCP candidate changes, poster URLs, video milestones, scene states and card inventories. Raw session JSON retains full observations. Seven unique directly served theme code bodies match the approved package, with zero mismatches. Optimized combined bundles are preserved by exact URL and decoded hash; they are not equated to individual source files or historical URLs.

Normal Home's recorded LCP candidate is a VIDEO at384ms; its film request starts348ms and finishes488ms, followed by a viewport callback at542ms. Moderate Home reports LCP672ms but its first viewport callback is2328ms. A VIDEO LCP entry may reflect a poster or early rendering; it must not be relabeled the first animated-frame paint. Desktop's LCP candidate is an IMG. Loaded-image observations, canplay, playing and requestVideoFrameCallback remain separate milestones. TownLine was not activated or substituted for the hero.

All three collection routes begin with the arrival hero playing and their three scene sources unset, readyState0. All six collection/profile observations have zero initial scene-film requests. The1→2→3→2→1 traversal activates films on approach, pauses offscreen films and resumes previous media. Each scene URL has one request per session; reverse re-entry causes no second full film download. Browser requests use `Range: bytes=0-` and206 responses, often consuming the complete selected file. This does not imply partial-transfer savings. Current platform B07 separately verifies exact nonzero ranges and416 behavior. No seek control was present, so no arbitrary currentTime manipulation was introduced.

The main runner's pause-button locator searched inside the figure, while the controls are siblings. Its missing-control notes are a harness gap, not a theme failure. Current `browser-parity.md/json` independently records working controls in all 18 scene/viewport observations on this candidate. The runner is corrected; its original executed source is retained as `main-runner-executed.cjs`. No full rerun was needed.

## Product cards and Quick View

The first Shop card selects a 480w WebP derivative, decoded as480×720 for236.3CSSpx atDPR 2. Natural width236 is browser density correction, not a236-pixel source. The first two cards are explicitly eager, including one below the viewport. Other cards use native lazy loading: normal initially loaded two additional near images, while moderate did not. Scrolling initiates later requests. This is proximity loading, not a claim that all offscreen images stay unrequested. The SG005 PDP uses its native WooCommerce image through i0.wp.com. Representative delivery does not certify all 33 SKU images or product-fidelity authority.

Quick View obtains the native form after intent and closes on Escape in both profiles. It does not eagerly fetch a PDP for every visible card. At intent, the document speculation policy also generates a Prefetch GET alongside the native Fetch GET toBR003. Normal transfers about58.7KB for each. These are distinct browser-prefetch and native-fetch requests, not evidence of duplicate QV fetch code. `platform/shop-speculation-rules.json` binds the emitted policy. Its plugin/platform generator remains UNKNOWN.

The original moderate screenshot was early: at16.423s the form existed but image.complete was false; the image transfer completed16.685s. Natural dimensions alone were insufficient readiness evidence. The original is retained. The bounded `qv-settled/moderate-shop.json` supplement waits for complete, image.decode and two animation frames. Its screenshot was inspected and the image is correct. Form and image readiness remain separate. Current browser-parity evidence also covers variation selection and rapid reopen/stale-content handling. No purchase was submitted.

## Ask Skyy deferred delivery and performance

Untouched mobile, desktop and reload states retain the approved same-model static guide with zero optional GLB/Three/Draco/runtime requests. Intent starts local modules, model fetch/decode, the first stable frame, walking-in and idle. Pause/resume and chat UI minimize/reopen are observed. **The deferred-delivery criterion passes; asset and activation performance do not.**

Runtime-reported stable-frame time is1204.6ms for normal mobile,1508.3ms for desktop and14216.3ms for moderate mobile. Moderate intent-to-first-WebGL-draw is14374.1ms. Canvas draws and RAF opportunities are CPU/compositor evidence, not physical-display or GPU timing.

The model is6,058,568bytes (5.78MiB), above the2.5MiB gate. Moderate model fetch takes 12325.9ms; model decode 1033.8ms; cooperative bounds 123.3ms elapsed. Those stage timings include network, worker and yield waits, not continuous main-thread stalls. The renderer is ANGLE Metal on Apple M5. Geometry remains1,930,256 triangles with 18 bones and one draw call/material; typed geometry plus estimated textures total 89.2MiB, excluding driver, decoder and JavaScript memory. Payload size plus constrained bandwidth is the dominant measured activation cost. A CDN HIT does not remove transfer time. No model or identity changes were made.

The separate Home probe preserves two navigation epochs. First navigation loads one GLB; minimize/reopen reuses it in memory. After a real same-context reload and12seconds untouched, intent creates a second GLB request explicitly marked `fromDiskCache=true`, with `finishedEncodedBytes=0`, followed by a stable frame in871.9ms. The first network transfer carried6,065,497 encoded bytes and a CDN HIT. This distinguishes browser-cache reuse from edge caching. Full flags and timeOrigin snapshots are in `cache-reentry-evidence.json`.

## Ownership and limits

MEDIA ASSET owns the oversized model and decoded footprint; NETWORK contributes the measured transfer delay. THEME owns the deferred loader, responsive-card policy, scene visibility and native QV fetch. Home's large shift remains a theme/layout investigation; the browser recent-input flag's cause is unknown. WORDPRESS/PHP returns the native form, but browser timings do not isolate PHP execution. WORDPRESS.COM/CDN expose optimized representations and cache labels; a label alone does not identify the responsible generating layer. THIRD PARTY background resources remain visible. UNKNOWN remains the speculation-policy generator. No platform observation was converted into a theme source fix.

All 17 sessions return200. The only non-GET requests are18 passive WooCommerce `get_refreshed_fragments` POSTs, one per navigation. No purchase, cart mutation, payment, question, suggestion-chip or chat submission was initiated. Each primary PDP run records two payment-permissions-policy console warnings, without a captured page exception or failed request. Cookie, authorization and Set-Cookie values are excluded; URL session/token identifiers are redacted. Customer data and request bodies are not persisted.

Representative Home, Signature hero, Black Rose/Love Hurts scenes, Skyy chat, PDP and normal/settled-moderate QV screenshots were inspected. Scene art, the approved character and native commerce layouts remain visible. This supplements root's complete 18 route/viewport and 10 fallback visual checks; settled screenshots do not prove layout stability during loading.

Field INP, physical-device GPU behavior, authenticated-session isolation, real checkout/payment, full catalog authority and production certification remain outside these tests. Current platform strict font/GLB MIME findings remain separate: successful decoding does not turn those header gates into PASS. Root's three Lighthouse samples are independent evidence; the Home CLS discrepancy is explicitly incorporated rather than averaged away.

All browser and CPU work is complete. Exact evidence hashes, metrics, the excluded-shift audit, quiet controls, safety checks and six outcomes are in `b13-report.json`.
