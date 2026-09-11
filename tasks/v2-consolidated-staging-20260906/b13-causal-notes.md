# B13 causal notes — current candidate, normal mobile Home, Shop, and PDP

**Status: evidence-backed diagnosis, not an optimization or a controlled causal experiment.** Read-only analysis of `b13/normal-home.json` and existing source/decoded-response evidence. No additional network requests.

## What happened in this sample

| Event | Navigation-relative time / measured duration |
|---|---:|
| HTML response first byte / completion (navigation timing) | 186 / 204.8 ms |
| Responsive hero poster request start / completion (CDP) | 207.9 / 245.3 ms |
| First poster LCP candidate | 468 ms |
| Recovery JavaScript bundle downloaded | 208.9–340.8 ms |
| jQuery Migrate finished | 1661.2 ms |
| wp-util finished | 1730.1 ms |
| Archivo font completed (CDP) | 1722.8 ms |
| Hero film request started | 1765.0 ms |
| Hero film receiveHeadersStart (relative to request) | 2257.7 ms |
| Hero film response received (navigation-relative CDP) | 4023.7 ms |
| Hero loadeddata / playing | 4181.0 / 4187.4 ms |
| First viewport frame callback | 4209.1 ms |
| VIDEO replaces IMG as LCP candidate | **4220 ms** |
| Rotating header WebM request started | 4185.9 ms |
| Rotating header first viewport frame callback | 4442.4 ms |

Frame callbacks indicate compositor submission, not measurement of a physical display. Different CDP and Resource Timing clocks/dispatch points can differ slightly; do not silently combine them into exact arithmetic partitions. The final LCP is an observed browser result, not a Lighthouse simulation.

## The hero scheduler does not wait for fonts

`wordpress-theme/skyyrose-flagship-2/assets/js/visual-recovery.js` is an immediate IIFE:

- Lines 11–17: allowed motion, intersection visibility, pause/document state, and `posterReady` gate activation.
- Lines 23–27: copy `source[data-src]` into `src`, call `video.load()` once.
- Lines 42–47: IntersectionObserver with threshold 0.12.
- Lines 58–65: decode the responsive poster, then set `posterReady` and synchronize.
- Lines 66–75: preserve the poster until playing plus a video-frame callback.

There is **no `document.fonts.ready`, DOMContentLoaded, or window-load wait** in this hero controller. The actual decoded recovery-containing bundle also has no fonts-ready or DOMContentLoaded wrapper. Archivo’s completion near the film start is correlation, not proof of a font gate.

## Earlier blocking scripts explain a material scheduling risk

The recovery controller was downloaded by 340.8 ms, but the emitted HTML puts its classic script after earlier scripts that do not have async/defer attributes:

1. jQuery Migrate: requested 208.8 ms, finished 1661.2 ms; exposed `cache;desc=HIT;dur=1415.0`.
2. wp-util: requested 209.0 ms, finished 1730.1 ms; exposed `cache;desc=HIT;dur=1450.0`.
3. Recovery-containing bundle `/_jb_static/??e6f6452ffc`: downloaded 208.9–340.8 ms but appears after those blocking scripts.

The decoded recovery bundle SHA is `318027d9ec9ef57030be3416e4742d8ab57688852c65dc7ada1caf64347eddf5` (5,663 bytes). The film starts 1765 ms, shortly after the earlier wp-util response completes. Classic document-order execution plus the observed response ordering is a strong explanation for the late activation despite early bundle download. A JavaScript execution trace or bounded scheduling experiment is still needed to quantify the exact improvement from changing this chain.

Ownership is shared, not exclusively theme or platform:

- **THEME:** `functions.php:393–395` enqueues the recovery controller in the footer with an empty dependency array and no explicit strategy. `inc/performance.php:393–405` adds defer to selected handles, but the recovery handle is absent from that list.
- **THEME + WOOCOMMERCE:** `functions.php:423–426` enqueues native Quick View with the `wc-add-to-cart-variation` dependency on Home and other eligible routes. The pinned WooCommerce `includes/class-wc-frontend-scripts.php:238–240` registers variation handling with `jquery`, `wp-util`, and `wc-jquery-blockui`. This explains why wp-util is part of the native commerce graph; it does not authorize removing Quick View.
- **WORDPRESS.COM / OPTIMIZER:** actual deployed resources are combined into `/_jb_static/` bundles and emitted in the observed blocking order. The source enqueue order alone does not prove which optimizer/WordPress decision determined final ordering. No configuration change is proposed as proven necessary without isolating that decision.
- **CDN / NETWORK / UNKNOWN:** large first-byte delays affect earlier static dependencies despite HIT labels. Server-Timing reports sizable platform-side intervals, but those labels do not identify exact origin execution, edge queueing, cross-tier retrieval, or network causes.

## The film response itself adds a second major delay

The hero WebM response is 206 with `Content-Range: bytes 0-676789/676790`, `video/webm`, one-year max-age, and `x-ac: 3.sjc _atomic_bur HIT`. It exposes `cache;desc=HIT;dur=2144.0`; CDP records 2257.7 ms to response headers after the request begins.

This is substantial delivery wait for a 676,790-byte static object. It is not proof of an origin/PHP problem. A HIT is not proof of low latency. Attribution: **delivery path (WORDPRESS.COM/CDN/NETWORK), exact split UNKNOWN**. Theme scheduling accounts for the initial request-start delay separately; do not blame all 4.22 seconds on either layer.

Archivo also exposes a 1213 ms HIT duration and 1261.8 ms request-to-headers timing for a 90,096-byte object. That is another static-delivery outlier. Its tiny post-load text shift is observed, but it is not an explicit hero start prerequisite.

## The rotating header WebM is not the cause of the early hero delay

`assets/js/theme.js:25–33` requires pageReady before animation; lines 126–127 set it after the window load event. In this sample load starts 4181.8 ms and the mark request begins 4185.9 ms—after hero loadeddata and after the hero’s 2257.7 ms response wait has already ended.

The header request therefore cannot explain the preceding 1.765-second hero start delay or 2.258-second film first-byte wait. It begins before the final 4220 ms video LCP by 34 ms, so an absolute zero-overlap claim would be too strong; there is no evidence that this short overlap caused the main LCP bottleneck. Its own first frame arrives 4442.4 ms. Do not remove mandatory rotating identity to address an earlier unrelated critical path.

## CLS is not cleared by the small session value

At 419.6 ms the raw layout-shift observer records **value 1.0** with `hadRecentInput=true`. The hero container moves from y 0 / height 844 to y 64 / height 780; header/Ask Skyy geometry appears or moves simultaneously. Other early shifts also carry the recent-input flag. The later non-recent shift is only 0.0002019.

The harness includes 1500 ms quiet time before navigation, yet the early large event is still excluded by the recent-input flag. Its cause remains unresolved. Do not describe the small filtered session score as proof that startup layout stability is fixed. Root’s desktop control provides a separate measurement.

Current accepted Home HTML still contains the same Jetpack critical CSS block: 31,177 bytes, SHA `9ec38548b78344db9d72aeddf1f0f6234348d14f5e33412f5d6ba048c0f6a07b`. It lacks `.sr2-archive` and `.sr2-house-header`; the full combined stylesheet remains `media="not all"` until onload. This supports the earlier hypothesis that critical coverage and deferred full geometry contribute to startup movement. It does not establish the complete cause or the reason for `hadRecentInput`.

## Desktop control confirms a genuine page-level CLS failure

The separate `normal-home-desktop.json` control uses a 1440 × 900 viewport at DPR 1. It records **LCP 452 ms** and **CLS 0.84501953125**, with the layout shift at **391.9 ms** and `hadRecentInput=false`. This is an actual counted startup layout failure. It resolves the earlier uncertainty about whether the mobile raw movement represents a real stability concern; it does not explain why the mobile observer marked its early events as recent-input exclusions.

The shift’s recorded sources include:

- Hero image container: previous rectangle y 0, height 900 → current rectangle y 76, height 824, width 1440 unchanged.
- Header start and end groups: changed widths and vertical positions.
- Ask Skyy recall link: from x 0 / y 0 and 239 × 82 to its final header position near x 1217 / y 15.5 and 90 × 44.
- Brand poster: approximately 187 × 58 at x 626 → 144 × 60 at x 648.

The source geometry matches the hero displacement exactly: `assets/css/design-tokens.css:53` defines `--sr2-header: 76px`; `assets/css/home-page.css:30` adds that height as archive top padding; `home-page.css:2` uses `min-height: calc(100svh - var(--sr2-header))`. Header layout and brand dimensions are defined in `assets/css/global-shell.css:5–11`. These final source rules are intentional and correct for the approved composition; their first-paint availability is the concern.

The desktop full combined stylesheet (`/_jb_static/??2b340035b6`, 263,825 decoded bytes) starts at 151.2 ms and finishes at 288.1 ms. It exposes `cache;desc=HIT;dur=11.0`. The saved accepted HTML places the full stylesheet behind `media="not all"` and an onload switch, while the previously inspected 31,177-byte generated critical block lacks `.sr2-archive` and `.sr2-house-header`. At the settled snapshot the stylesheet has `media="all"`, as expected after that switch. The geometry changes subsequently appear in the 391.9 ms observer event.

**Bounded attribution:** this strongly implicates incomplete initial CSS coverage and late application of the final theme geometry, at the **THEME / WORDPRESS.COM critical-CSS integration boundary**. The theme owns the necessary layout rules; the platform optimizer emits the generated critical subset and deferred stylesheet treatment. This evidence does not establish which side should implement the final remedy, nor the exact stylesheet-application timestamp. A later controlled test of current critical coverage is needed before changing source or platform configuration. The 11 ms cache timing and 137 ms full stylesheet download in this control do not support blaming a multi-second CDN delay for this particular shift.

The rotating header film is gated until the approximately 570 ms window load, **after the 391.9 ms shift**. Stable component geometry once loaded therefore does not clear the page’s startup CLS. The source poster participates in the shift because initial and final layout rules differ; that is not evidence that removing the mandatory animated mark would solve the underlying page-wide CSS coverage defect.

**Desktop control result: startup layout stability FAIL; overall frontend performance remains NEEDS_MORE_WORK despite the fast LCP in this sample.**

## Normal Shop: connection setup dominates this sample

The observed **3588 ms LCP** belongs to the approved product-card frame, `black-rose-portal-statue-640w.webp`, not the BR-003 product photograph. The frame renders at approximately 358 × 598 CSS pixels and its actual bitmap is 640 × 1068 pixels.

| Event | Navigation-relative time / measured duration |
|---|---:|
| DNS start / end | 1.9 / 4.1 ms |
| Connection start | 4.1 ms |
| Secure connection start / connection end | 21.8 / 3014.5 ms |
| HTML request start / first byte / completion | 3014.7 / 3385.5 / 3408.6 ms |
| LCP frame request start / completion (CDP) | 3399.4 / 3532.8 ms |
| Frame request-relative response-header wait | 68.8 ms |
| Frame LCP | 3588 ms |

The approximately **2992.7 ms secure-connection interval** precedes the HTML request. This is the dominant measured delay. Ownership is **NETWORK / connection setup, with exact client/network/platform cause UNKNOWN**; browser navigation timing alone does not identify why TLS establishment took this long. It is not evidence that a theme image or WordPress/PHP execution consumed those three seconds. The subsequent HTML first-byte interval is about 370.8 ms and its response exposes `cache;desc=STALE;dur=326.0`.

The frame begins approximately 13.9 ms after the document first byte and finishes before LCP. It transfers a 123,928-byte body, exposes `cache;desc=HIT;dur=17.0`, and receives one-year caching. This sample does not show a substantial frame-discovery delay or a large-image transfer bottleneck.

BR-003 selects `assets/derived/card-fronts/br-003-480w.webp`, independently confirmed as **480 × 720 pixels**, for approximately 236 CSS pixels of width at DPR 2. The ideal raster width for that display is approximately 472 pixels, so the selected derivative is appropriate. Browser `naturalWidth` reports approximately 236 after density correction; that number must not be mislabeled as the decoded bitmap width. The source offers 320/480/768/1024 candidates and its mobile `sizes` rule accounts for the product photograph’s 66% card width. The selected URLs are explicit theme derivatives, with no CDN transformation visible in those URLs.

The first two product photographs are eager (BR-003 high priority, BR-014 auto), including the second card below the initial fold. Additional nearby cards start through browser lazy-loading proximity; a farther BR-009 request starts only around 16.05 seconds after scrolling. Thus initial below-fold requests are present, but this does not mean all catalogue images are eager. Preserve card identity while judging whether the small eager/proximity window warrants a later bounded experiment.

**Evidence serialization limitation:** `run-b13.cjs` applies `safeUrl()` to every JSON string beginning with HTTP(S), including the entire captured `srcset` string. URL normalization encodes the separating spaces as `%20` in the saved JSON. Those sequences are not evidence of malformed emitted markup. Actual `currentSrc` selection and the independent derivative dimensions are the reliable sizing evidence here.

## Quick View: speculative prefetch overlaps the required native PDP fetch

Both Shop profiles record two GET request entries for the identical staging `/product/br-003/` URL after Quick View intent:

| Profile | Intent | Prefetch start | Native Fetch start | Separation |
|---|---:|---:|---:|---:|
| Normal | 18277.8 ms | 18313.8 ms | 18323.8 ms | 9.9 ms |
| Moderate | 16199.2 ms | 16236.3 ms | 16257.9 ms | 21.7 ms |

The first entry is CDP type **Prefetch**, with a document script initiator and `fromPrefetchCache=true`. The second is type **Fetch**, initiated by function `b` in the native Quick View bundle `/_jb_static/??bae87d0a03`, and `fromPrefetchCache=false`. Both return HTML; neither is a separate WooCommerce API call.

Accepted Shop HTML includes document-level speculation rules with `eagerness: conservative`, excluding links matching `.no-prefetch, .no-prefetch a`. Theme source contains no speculation-rules implementation. Its Quick View link (`template-parts/commerce/product-card.php:127`) has a real product href without that exclusion class. The native controller (`assets/js/quick-view-commerce.js:74–90`) guards an already active URL and performs the one required same-origin HTML fetch to obtain the real PDP form. The evidence therefore supports **speculative navigation prefetch overlapping native form retrieval**, not duplicate invocation of the native loader.

Ownership is the **THEME / WORDPRESS or platform speculation integration boundary**: the theme controls the trigger markup; a non-theme layer emits the speculation policy. The exact emitting plugin/platform component has not been identified from the captured markup. A later bounded test may evaluate exclusion of Quick View triggers from speculative navigation while preserving the native PDP fetch and ordinary product navigation.

Each Prefetch record reports approximately 58.7 KB `finishedEncodedBytes`, but also zero `dataEncodedBytes` and `fromPrefetchCache=true`. The native Fetch reports a similar encoded completion size. These are two request lifecycles; do not claim a verified doubled network transfer simply by summing those fields. Browser prefetch reuse and CDN cache behavior remain separate observations.

**Deferred-delivery gate passes:** both entries follow explicit intent, native purchase forms appear, and closing removes the form. Record the overlapping speculative request as an optimization opportunity with the byte-accounting limitation above. No source or configuration change was made.

## Normal PDP: payment policy console boundary

`normal-pdp.json` records two console messages: “Permissions policy violation: payment is not allowed in this document.” The captured staging response has `Permissions-Policy: payment=()`. No JavaScript exception was recorded and no order or payment was submitted. These messages are consistent with the existing **platform/policy boundary**, not proof of a newly introduced PDP or Quick View feature regression. They also do not certify payment behavior; authorized sandbox payment certification remains a separate phase.

## Evidence boundaries

Primary evidence: `.artifacts/v2-consolidated-staging-20260906/b13/normal-home.json`, `normal-shop.json`, and `normal-pdp.json`; decoded bundle file under `b13/decoded-responses`; current HTML captured by `platform/timing-final-home-0.json` points to restricted raw storage. Current CSS bytes and emitted script order were read from that accepted HTML. The normal Home sample’s captured request URLs and decoded bundle hashes match those references.

No source, staging configuration, media asset, or database change was made for this analysis. Findings are diagnostic ownership boundaries for the next authorized optimization phase, not permission to change staging.
