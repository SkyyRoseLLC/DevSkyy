# Home delivery deep dive — consolidated staging candidate

Status: **NEEDS_MORE_WORK**. Diagnosis completed; no remediation deployed.

## Scope and identity

Target: https://staging-7e48-skyyrose.wpcomstaging.com/

Release SHA-256: `a4ec431146f31d036d5547137d3274a693ccd3d072be6089126593f642d14f4b`.
Fresh remote inventory matched all **536 runtime files** against the consolidated manifest. This is the current approved consolidated candidate, not the historical static Home implementation.

This investigation used read-only staging navigation, read-only SSH inspection of deployed plugin code, source inspection, and 42 isolated offline browser replays. No theme, WordPress content, platform configuration, cache purge, deployment, model, product authority, order, or payment changes were made. Passive WooCommerce fragment refresh occurred during the normal live page load; no commerce action was submitted.

## Findings

### 1. Confirmed: critical CSS does not cover the current initial Home layout

Current HTML includes 31,177 bytes of Jetpack Boost critical CSS, SHA-256 `9ec38548b78344db9d72aeddf1f0f6234348d14f5e33412f5d6ba048c0f6a07b`. This matches the earlier delivery capture. It contains older `.sr2-header` rules but no `.sr2-archive` or `.sr2-house-header` selectors required by the current Home and header.

Meanwhile, the full combined stylesheet `/_jb_static/??2b340035b6` is emitted with `media="not all"` and an onload media switch. The browser can paint the page before the current layout rules apply. The result is a substantial initial-to-final change in hero, copy, header controls, and brand geometry. In controlled desktop replays the hero image moved from y=0, height=900 to y=76, height=824; subsequent font/copy settling can make the enclosing section taller.

The actual deployed Jetpack Boost 4.7.0 `Display_Critical_CSS` class directly implements this transformation and prints the `jetpack-boost-critical-css` style. This identifies the responsible optimization mechanism. Merely having LiteSpeed Cache active does not establish that it caused this defect.

Controlled intervention: change only that stylesheet link to normal render-blocking delivery, retaining its exact CSS bytes and all media. Median CLS fell from 1.0001 to 0.0050 at 1440px and from 1.0002 to 0.0348 at 390px. This confirms that the missing initial layout styling is responsible for the dominant shift in these replays.

The intervention is a diagnostic control, not a shipping recommendation. In the test's deliberately delayed CSS scenario, desktop first contentful paint moved from 132ms to 824ms. Blocking the entire 263,825-byte decoded stylesheet trades an unstable early paint for a later stable paint. Prefer correct current critical CSS with the remainder cacheable and asynchronous.

### 2. Confirmed: hero startup is exposed to unrelated blocking scripts

The hero's immediate controller has no explicit dependency on jQuery, fonts readiness, DOMContentLoaded, or window load. It checks motion preference, intersection, document visibility, pause state, and responsive poster decoding, then assigns the film URL and starts playback.

However, the emitted HTML places the controller after classic blocking scripts including jQuery Migrate and wp-util. Downloading the controller early does not permit it to execute ahead of those scripts. The theme enqueues recovery as a footer script with no dependencies; its performance defer list does not include recovery. Native Quick View legitimately depends on the WooCommerce variation form stack, but the hero should not inherit the delivery delay of that stack through document order.

Controlled intervention: with jQuery Migrate/wp-util responses held for 1,600ms, the ordinary desktop hero film request started at median 1,680ms. Executing the same existing controller after the hero markup but before those blockers moved film discovery to 101ms. All commerce scripts remained present. The controller's existing initialization guard prevented a second initialization when its original tag executed later.

This proves the scheduling vulnerability. It does not authorize globally making WooCommerce scripts async, removing native forms, or shipping the experimental inline injection. A production change should isolate the small hero boot sequence while preserving dependency ordering and all reduced-motion, poster, pause, viewport, and visibility behavior.

Early hero startup alone did not repair CLS: it remained approximately 1.000. Fix the initial layout before accelerating playback. With both controls, desktop CLS was 0.0067 and first animated frame 905ms versus 1,790ms with CSS fixed but the hero still behind delayed scripts.

### 3. Confirmed observation: media response latency varies independently

The prior real normal-mobile capture recorded film discovery at 1,765ms, approximately 2,258ms waiting for response headers, and a visible-frame callback at 4,209ms. VIDEO replaced the poster as LCP at 4,220ms. This was real browser behavior on staging, not merely a Lighthouse model.

The new desktop control recorded film discovery at 802ms, approximately 21ms to first byte, and a visible-frame callback at 959ms. The same 676,790-byte approved WebM was served as HTTP 206 with a CDN HIT indication. This sample proves the delivery wait is variable; it does not establish that mobile delivery is now fixed. Device profiles and run conditions differ, so these are observations, not an A/B performance comparison.

In the offline control, adding 1,700ms of film-response delay while keeping early discovery changed the first visible desktop frame from 156ms to 1,840ms. Early discovery cannot eliminate time spent waiting for the film response.

A CDN HIT label does not establish low latency, explain upstream work, or prove that PHP executed for a static file. Ownership of the slow response within the WordPress.com/CDN/network path remains unresolved. Preserve request timing and server headers for platform investigation rather than guessing or purging the evidence away.

### 4. Likely mechanism: same-theme replacement did not invalidate critical CSS

Deployed plugin source provides a plausible lifecycle explanation:

- `Environment_Change_Detector` treats `after_switch_theme` as a major environment change.
- Published post saves and plugin activation/deactivation are marked minor in the inspected class.
- `Critical_CSS_Invalidator::handle_environment_change()` clears stored critical CSS only for a major change.
- Replacing files in the same active theme does not itself constitute a theme switch. Saving the About page is a minor event in this handler.

This explains how an exact filesystem deployment can coexist with unsuitable generated critical CSS. It is a supported mechanism, not proof of the complete generation/job history. The timestamp, source viewport, regeneration attempts, and any other invalidation mechanisms have not been established. Do not claim that regeneration alone is guaranteed to fix the output until the generated rules and first-paint behavior are verified.

### 5. The rotating mark and untouched Ask Skyy are not the measured upstream blockers

In the prior slow mobile capture, the rotating header film request began at 4,186ms, after the hero film had already spent more than two seconds awaiting response headers. In the new desktop capture, it began at 951ms, after the primary CSS/layout transition and hero discovery. Its animation transfer cannot explain those earlier delays. Brand geometry participates in the CSS shift, which calls for correct initial sizing, not deletion of the approved rotating artwork.

Untouched Ask Skyy requested no expensive 3D runtime in the fresh control. Every offline case retained the hero, reached its ready state, and left optional 3D inactive. This is not a new full Ask Skyy interaction certification.

## Fresh live control

Chromium desktop, 1440×900, DPR1, CPU1, no imposed bandwidth/latency limit; fresh browser context. Observe untouched page through approximately 19 seconds.

| Measurement | Observed |
|---|---:|
| HTTP | 200 |
| HTML TTFB | 189ms |
| FCP / final recorded LCP | 416 / 416ms |
| LCP element | Hero poster IMG |
| CLS | **0.845351** |
| Full stylesheet request / completion | 202 / 732ms |
| Hero film request / completion | 802 / 865ms |
| First visible hero frame callback | 959ms |
| Rotating mark film request | 951ms |
| Untouched expensive 3D requests | 0 |
| Recorded page errors / long tasks | 0 / 0 |

The fast LCP number does not clear Home: the large shift persists and this desktop run did not reproduce the prior mobile video LCP replacement. Frame callbacks measure browser compositor behavior, not a physical display.

## Controlled experiment matrix

Seven conditions × two viewport widths × three repetitions = 42 completed cases. Chromium 145.0.7632.6; viewports 1440×900 and 390×900, DPR1. The narrow case is a responsive-width replay, not mobile-device emulation. No physical-device inference is warranted.

All requests were fulfilled from preserved decoded responses or frozen release assets; unknown requests were aborted. Zero live network during replay. Baseline synthetic response waits: 20ms for most assets, 700ms for the main CSS; selected conditions add 1,600ms script wait or 1,700ms film wait. These are causal controls, not expected customer timings. Browser processing and deferred stylesheet application can extend beyond the injected response wait.

| Width | Condition | Median CLS | Median FCP ms | Film request ms | Visible frame ms |
|---|---|---:|---:|---:|---:|
| 1440 | Baseline | 1.0001 | 132 | 122 | 172 |
| 1440 | Full CSS blocking | 0.0050 | 824 | 824 | 947 |
| 1440 | Delayed blocking scripts | 0.8453 | 124 | 1680 | 1741 |
| 1440 | Early controller, delayed scripts | 1.0001 | 132 | 101 | 156 |
| 1440 | Early controller, delayed scripts and film | 1.0001 | 132 | 103 | 1840 |
| 1440 | CSS blocking, delayed scripts | 0.0065 | 824 | 1696 | 1790 |
| 1440 | CSS blocking, early controller, delayed scripts | 0.0067 | 820 | 795 | 905 |
| 390 | Baseline | 1.0002 | 128 | 114 | 166 |
| 390 | Full CSS blocking | 0.0348 | 832 | 814 | 935 |
| 390 | Delayed blocking scripts | 1.0002 | 132 | 1677 | 1724 |
| 390 | Early controller, delayed scripts | 1.0002 | 136 | 96 | 144 |
| 390 | Early controller, delayed scripts and film | 1.0002 | 128 | 96 | 1838 |
| 390 | CSS blocking, delayed scripts | 0.0319 | 820 | 1703 | 1778 |
| 390 | CSS blocking, early controller, delayed scripts | 0.0348 | 816 | 791 | 894 |

All 42 cases: hero ready, no horizontal overflow, optional 3D inactive. Settled desktop baseline and CSS-control screenshots were visually inspected: same hero artwork, crop/layout, header mark, navigation, and Ask Skyy poster. Animation phase differed naturally. This is bounded visual evidence, not full responsive/founder acceptance.

Each replay has a shared `Failed to fetch` page error from the deliberately blocked CommerceKit nonce endpoint. Analytics and passive WooCommerce fragment refresh were also blocked. This common test limitation is not a newly discovered production regression. It means these replays do not certify commerce behavior. The fresh live control had no recorded page errors.

## Ownership and next actions

| Bottleneck | Ownership | Next bounded action |
|---|---|---|
| Inadequate generated critical CSS plus asynchronous full CSS | Jetpack Boost/platform optimization; theme supplies required layout contract | Inspect generation status, regenerate for exact current candidate under separately authorized platform work, verify current selectors and first paint. If generation remains inadequate, evaluate a small route-specific critical fallback or targeted stylesheet exclusion. |
| Hero executes behind unrelated classic scripts | Theme scheduling; plugin-generated output ordering must also be verified | Isolate hero startup without changing native WooCommerce dependencies, motion controls, or media identity. |
| Multi-second static-file response wait in prior capture | WordPress.com/CDN/network, exact internal cause UNKNOWN | Repeat bounded cold/warm request observations with response headers and correlate with browser playback; do not infer PHP or purge caches. |
| Residual small layout shifts after CSS control | Theme/fonts/content layout | Inspect remaining font/copy/control shifts after dominant defect is corrected; no broad cleanup. |

Implementation order: correct and verify critical layout delivery first; isolate hero discovery second; reassess actual delivery latency third. Keep poster discoverable in HTML and present until a rendered animated frame. Do not hide animation to manipulate LCP. Do not remove the rotating mark, Quick View, Ask Skyy, heroes, or scenes.

Acceptance for a future fix: visual EQUIVALENT or BETTER at 320/390/768/1024/1440, stable header/hero geometry from first paint, CLS ≤0.1, Home LCP target ≤2.5s on documented normal and moderately constrained mobile profiles, separately reported hero first-frame readiness, unchanged native Quick View and Ask Skyy intent behavior, and exact runtime identity. Field/physical-device validation remains separate.

## Evidence paths

All paths below are relative to the repository root:

- `.artifacts/v2-home-deep-dive-20260907/remote-before.sha256`
- `.artifacts/v2-home-deep-dive-20260907/home.headers`
- `.artifacts/v2-home-deep-dive-20260907/live/normal-home-desktop.json`
- `.artifacts/v2-home-deep-dive-20260907/live/summary.json`
- `.artifacts/v2-home-deep-dive-20260907/replay.cjs`
- `.artifacts/v2-home-deep-dive-20260907/replay-results.json`
- `.artifacts/v2-home-deep-dive-20260907/combined/replay-results.json`
- `.artifacts/v2-home-deep-dive-20260907/experiment-summary.json`
- `.artifacts/v2-home-deep-dive-20260907/boost-display-critical-css.php`
- `.artifacts/v2-home-deep-dive-20260907/boost-environment-detector.txt`
- `.artifacts/v2-home-deep-dive-20260907/boost-critical-css-invalidator.php`
- `tasks/v2-consolidated-staging-20260906/b13-causal-notes.md`

Theme source anchors: `wordpress-theme/skyyrose-flagship-2/functions.php:395`, `functions.php:426`, `assets/js/visual-recovery.js`, `inc/performance.php`, `assets/css/home-page.css`, `assets/css/global-shell.css`, and `assets/css/design-tokens.css`. The frozen release export supplied runtime assets for replay.
