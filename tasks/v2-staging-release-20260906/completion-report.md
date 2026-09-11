# V2 staging deployment and verification

> **CORRECTION — BROWSER / FEATURE PARITY: FAIL.** Founder rotating-mark correction (2026-09-06): staging header and footer both select the static fallback. Browser/feature parity is FAIL under the clarified mandatory contract. Local restoration passes bounded checks but is not deployed. Prior filesystem/deployment receipts remain historical and valid for their captured candidate.

**The exact authorized artifact is installed on staging. Verification is complete with unresolved performance and platform findings. Overall engineering readiness remains NEEDS_MORE_WORK. No production approval or deployment is implied.**

Target: https://staging-7e48-skyyrose.wpcomstaging.com

| Independent result | Status | Evidence and boundary |
|---|---|---|
| Filesystem parity | PASS | The 447-file pre-install baseline matched exactly. All 527 installed runtime files matched after installation and again at final verification. |
| Browser parity | FAIL — mandatory rotating mark missing | Accepted candidate rendered in Chromium at desktop/mobile widths; 18 final route/viewport cases and 10 fallback cases passed. Physical devices and other browser engines are not certified. |
| Feature parity | FAIL — mandatory rotating mark missing | Five animated heroes, nine approved scenes, responsive cards, native Quick View, Ask Skyy, Shop, representative PDP, Bag, Cart and Search verified. Empty-session Checkout redirects to Cart; checkout form, personalization and transactions remain ENVIRONMENT-BOUND. |
| Platform delivery | NEEDS_MORE_WORK | Compression, sampled representation/key separation, conditional requests, direct asset hashes and video ranges pass. Strict font/GLB MIME and response-latency targets fail; authenticated/session isolation and global immutable policy remain environment-bound. |
| Performance | NEEDS_MORE_WORK | Home startup layout instability is confirmed; Lighthouse Home/Shop simulated LCP remains above target; the large Skyy model causes slow constrained-network activation. No optimization was applied. |

## Exact deployment and recovery

- Archive: `skyyrose-flagship-2.zip`, 527 runtime files, 170,023,102 bytes.
- ZIP SHA-256: `e47588205c55e6303588a207d7ee766ad175b82f606d167198f4ec2c25b1d80d`.
- Authorized procedure SHA-256: `ad4926a97cfea2a195de6366a4aaa61aa8b65b12f0a7dc685a69ef2c81231911`.
- Preserved 447-file rollback TAR SHA-256: `1b8164b36374aadea85aa4ec7c4d3f8a18e9f8e9f34042f706bd21c090a45ac3`.
- Installation completed at 2026-09-06 21:10:49 UTC using the reviewed whole-theme procedure. Both archives and the extracted recovery tree were verified before installation.
- Final verification at 21:35:13 UTC confirmed all 527 runtime hashes and all 12 recorded options unchanged. All 608 local theme source files are unchanged.

The standard installer reported an automatic cache purge. No manual purge command, optimizer/CDN setting change or configuration workaround was issued. Rollback was not triggered: filesystem parity and approved feature presence were established, with no demonstrated installation-created severe commerce/accessibility break. The unresolved CLS finding remains serious and does not become acceptable because rollback was not required. The exact backup and prepared recovery tree remain retained.

## Managed delivery incident

The first browser responses after installation still contained historical static Home, old Skyy references and nonresponsive card markup. The initial Home response exposed STALE/Batcache headers and a Last-Modified timestamp preceding authorization. Ordinary URLs naturally recovered without bypass parameters or manual purges; accepted Home markup was confirmed at 21:16:52 UTC. Initial failures and later recovered results are retained separately.

A second, more specific delivery issue remains: the inline `jetpack-boost-critical-css` block is identical in old and recovered Home HTML — 31,177 bytes, SHA-256 `9ec38548b78344db9d72aeddf1f0f6234348d14f5e33412f5d6ba048c0f6a07b`. It contains no current `.sr2-archive` or `.sr2-house-header` rules. The full stylesheet is deferred with `media="not all"` until onload. Captured layout-shift geometry moves the hero from y=0 to y=64 on mobile and y=76 on desktop when header/archive geometry applies.

This supports an inadequate or retained critical-CSS artifact as a likely contributor. The geometry rules belong to the theme; the observed critical-CSS extraction/defer behavior is Jetpack-marked delivery. Exact generator/cache configuration and controlled causality have not been established. The next investigation should resolve that ownership boundary before any theme workaround or platform change.

## Actual browser measurements and Lighthouse are separate

B13 mobile profile: Chromium 145.0.7632.6, 390×844 CSS pixels, DPR 2, touch emulation. Normal uses unthrottled network/CPU 1×. Moderate uses configured 100 ms latency, 4 Mbps down/1 Mbps up, CPU 2×. These are diagnostic samples on the test computer, not physical-phone or field measurements.

| Route | Normal observed LCP | Moderate observed LCP |
|---|---:|---:|
| Home | 384 ms | 672 ms |
| Signature | 296 ms | 684 ms |
| Black Rose | 568 ms | 680 ms |
| Love Hurts | 432 ms | 992 ms |
| Shop | 224 ms | 1,556 ms |
| Representative PDP | 252 ms | 2,372 ms |

The observed early LCP entries do not equal the first animated frame. Moderate hero viewport-frame callbacks occurred approximately 1.61–2.33 seconds after navigation. The earlier late-LCP behavior did not recur in these B13 samples; this does not establish universal LCP compliance.

**Home layout stability fails.** Desktop B13 records actual CLS 0.846951 with non-recent-input shifts. Both primary mobile Home samples contain an additional shift of 1.0 that Chromium marked recent-input-related and excluded from the standard CLS score. Two quiescent pre-navigation controls retain that large excluded shift; the simple setup-timing hypothesis was not supported. Its flag cause remains unknown. Reported low mobile CLS must not be used to clear Home stability.

Supplemental Lighthouse 13.4.1 uses a different mobile profile and simulated throttling: 412×823, DPR 1.75, 150 ms RTT, approximately 1.64 Mbps, CPU 4×. One sample per route:

| Route | Performance score | Simulated LCP | Observed CLS |
|---|---:|---:|---:|
| Home | 49 | 5,434 ms | 1.000 |
| Shop | 84 | 3,932 ms | 0.000214 |
| Signature | 94 | 2,222 ms | 0.002177 |

Lighthouse Home's trace-derived LCP breakdown is approximately 2,751 ms; its headline 5,434 ms is the simulated result. Document response times in those Lighthouse samples were approximately 1.8–2.0 seconds, unlike the faster cached B13/curl samples. Do not attribute that difference to PHP, CDN, a cache policy or the theme without additional evidence.

## Remaining bottlenecks and ownership

1. **Home startup geometry / critical-CSS coverage:** confirmed large shifts, likely contribution from missing current layout rules in Jetpack-marked critical CSS while full CSS is deferred. Ownership spans required THEME geometry and WORDPRESS.COM/Jetpack delivery; the precise generation/configuration cause remains UNKNOWN. No broad CSS inlining or theme workaround was applied.
2. **Ask Skyy asset weight:** MEDIA ASSET plus the documented NETWORK profile. The unchanged 6,058,568-byte GLB takes about 12.326 seconds to fetch under the moderate profile; decode adds approximately 1.034 seconds. First stable frame is 14.216 seconds from runtime start, about 14.374 seconds from intent to first draw. Model size exceeds the 2.5 MiB target. This is independent of successful deferred loading.
3. **Quick View duplicate document transfer at intent:** browser document-rule Prefetch and native Quick View Fetch both request BR-003, approximately 58.7 KB each in the normal sample. The native form architecture works, and there is no eager fetch for every card. The exact generator of the emitted speculation policy remains UNKNOWN; no duplicate native-fetch implementation is inferred.
4. **Platform MIME and uncached response latency:** four sampled WOFF2 URLs return a legacy MIME and the GLB returns `application/octet-stream`, despite correct bytes and successful rendering. Anonymous Checkout's 302 response median is 1,268 ms; Cart has two slow samples near 1.6 seconds. Exact server-side ownership remains unisolated. No platform problem was patched inside theme code.

Skyy reuse works: one network GLB transfer on first activation, none for same-page chat reopening, and after a real reload an intent-triggered disk-cache response transfers zero bytes. The first transfer was approximately 6.07 MB even with a CDN HIT. Browser reuse and CDN status are separate evidence.

## B13 required outcomes

| Outcome | Result |
|---|---|
| HOME HERO DELIVERY | FAIL — approved animation works; startup layout stability remains unresolved |
| COLLECTION SCENE DELIVERY | PASS — nine scenes, approach loading, pause/reverse/re-entry, no repeated full film downloads in sampled sessions |
| PRODUCT CARD DELIVERY | PASS — representative responsive delivery; 480×720 decoded for approximately 236 CSS pixels at DPR 2 |
| QUICK VIEW DELIVERY | PASS — native intent-loaded form/media; duplicate speculation transfer remains a measured cost |
| ASK SKYY DEFERRED DELIVERY | PASS — separate asset/activation performance FAIL |
| MOBILE CINEMATIC DELIVERY | FAIL — Home instability and constrained Skyy activation remain |

## Platform evidence boundaries

706 read-only probe requests were retained. Static representation equality covers 61 code URLs across nine encoding requests each, plus observed Skyy assets. All 37 original current-package-comparable bodies match across 18 URLs. Six additional old/current key probes retain the correct distinct historical/current bodies; they are not mixed into current-source comparisons. Both observed keys have long max-age without an immutable token; global policy remains unverified.

MP4 and WebM valid ranges returned 206 with exact source slices and Content-Range; out-of-range requests returned 416 with the correct total. Three sampled validators returned 304. Anonymous Cart/Checkout expose private/no-cache and BYPASS, but no populated or authenticated session was created to certify isolation.

Final uncontended five-sample TTFB medians: Home 111 ms, Shop 73 ms, Signature 73 ms, PDP 76 ms, Cart 138 ms, Checkout redirect 1,268 ms. These are network-inclusive responses from one test location; Checkout is not a document measurement and origin execution is not isolated.

## Stop point and evidence

Stopped after authorized staging verification. No production deployment, manual cache purge, platform/CDN configuration change, WordPress content edit, product/media authority change, model edit, Town Line work, order or payment submission occurred. Passive WooCommerce fragment-refresh requests generated by the site are separately recorded. No chat question was submitted.

Retain NEEDS_MORE_WORK and return these findings for founder review before subsequent optimization or creative phases. Physical-device, full accessibility, populated checkout, payment sandbox and production certification remain separate.

Companion evidence: `browser-parity.md/json`, `platform-report.md/json`, `b13-report.md/json`, `lighthouse-diagnosis.md/json`, `deployment-and-rollback.md`, and `.artifacts/v2-staging-release-20260906/execution/final-verification.json`. The browser review page links the reports, raw metric summaries and representative screenshots.
