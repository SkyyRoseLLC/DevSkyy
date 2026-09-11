# V2 cinematic finalization performance evidence — 2026-09-06

**Status: NEEDS_MORE_WORK. Canonical performance budgets are not passing.** This report describes a local candidate, with no staging, deployment, promotion or field-performance claim.

## Measurement and comparison boundary

The valid before/current comparison is `corrected-baseline` versus `current` in `.artifacts/v2-cinematic-finalization-20260906/performance-summary.json`: both use the corrected WebM-capable fixture and the same Lighthouse 13.4.1 Lantern simulated mobile profile (390×844, DPR 1, CPU ×4, RTT 150 ms, 1638.4 Kbps). Cold contexts and a synthetic seeded cart are used. These are single samples, not repeated medians or statistical confidence. Historical first-cycle and earlier uncorrected fixture numbers are not comparable and are excluded from improvement claims.

`compressed-current` changes text transport to gzip on the same PHP fixture/source. It is a transport diagnostic, not a source-only gain, CDN test or proof that production sends the same encoding. `current-390` observers record unthrottled navigation through load + 10 seconds; their subparts cannot be added to Lantern simulated LCP. `current-devtools` provides separate actually throttled browser observations for Home, Shop and PDP; its timings and LCP candidate choices must remain separate from Lighthouse simulation.

## Six-route before / current / gzip mobile comparison

LCP in milliseconds; transfer is the Lighthouse report total in KiB (1024 bytes). It differs from the observer window used for canonical transfer budgets.

| Route | Before LCP | Current LCP | Change | Gzip LCP | Before KiB | Current KiB | Gzip KiB | Current CLS | Current TBT ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| home | 6029 | 5874 | -155 | 3997 | 2011.8 | 1591.1 | 1269.4 | 0.05522 | 37 |
| shop | 6252 | 5874 | -378 | 4761 | 1070.9 | 1093.4 | 649.7 | 0.00054 | 1.5 |
| collection | 4213 | 5268 | +1055 | 3542 | 2174.4 | 1638.0 | 1296.9 | 0.00032 | 39 |
| pdp | 5429 | 5508 | +78 | 2598 | 1180.3 | 1184.8 | 698.2 | 0.00128 | 0 |
| cart | 6017 | 5743 | -274 | 2443 | 841.9 | 852.6 | 354.1 | 0.00061 | 0 |
| checkout | 4822 | 5118 | +296 | 2420 | 810.2 | 811.4 | 319.6 | 0.00000 | 0 |

Home and Collection transfer decreases are real in these fixture samples, but lower bytes did not deliver a uniform LCP improvement: Collection worsened by about 1055 ms; PDP and Checkout also worsened. Home, Shop and Cart improved numerically, but single-run noise prevents a reliable causal claim. Gzip removes a substantial text-transfer burden; only Cart and Checkout meet 2500 ms in that diagnostic. It does not change the canonical failing results.

## Current route payload and budget result

Observer window: initial ends at `window.load`, total at load + 10 seconds without interaction. All sizes below are KiB. JS/CSS/image/video columns are actual transferred resource categories, not minified file-size estimates.

| Route | Initial | Total | JS | CSS | Image | Video | Fonts | Requests | Canonical failures |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| home | 1523.5 | 1585.7 | 211.9 | 147.0 | 261.5 | 661.1 | 207.2 | 46 | lcpMs, initialTransferKiB |
| shop | 1038.4 | 1088.1 | 193.7 | 244.8 | 297.7 | 0.0 | 207.2 | 43 | lcpMs, cssKiB |
| collection | 1632.7 | 1632.7 | 204.2 | 134.6 | 492.4 | 462.0 | 207.2 | 47 | lcpMs |
| pdp | 1196.6 | 1196.6 | 287.9 | 260.9 | 325.7 | 0.0 | 207.2 | 53 | lcpMs |
| cart | 848.6 | 848.6 | 259.4 | 289.4 | 39.9 | 0.0 | 134.2 | 39 | lcpMs, cssKiB |
| checkout | 806.1 | 811.4 | 258.9 | 269.5 | 5.7 | 0.0 | 140.6 | 34 | lcpMs, initialTransferKiB, jsKiB, cssKiB |

Every route fails the canonical 2500 ms LCP gate. Home and Checkout additionally fail initial transfer; Shop, Cart and Checkout fail CSS; Checkout fails JS. All six sampled CLS values pass 0.1. Field INP and route animation render-work p95 remain UNVERIFIED, not zero or PASS. The zero initial model/texture request gate passes in this observer window and has separate real-Metal character-intent evidence; it is not an estimate of later decoded GPU memory. Request/long-task gates refer to this unthrottled window, not the separate throttled trace.

## LCP critical path: observed timings and causal limits

The actually throttled traces identify these image candidates. Times below are milliseconds from navigation start. “Start” is Resource Timing fetch initiation, a practical discovery proxy, not exact parser discovery. Render gap is LCP minus responseEnd; it contains multiple possible causes and is not isolated decode time.

| Route | Candidate | Document TTFB | Fetch start | Request start | Response start | Response end | LCP | Post-response render gap |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| home | black-rose-bay-bridge-monuments-v4-640w.webp | 592.4 | 655.2 | 669.2 | 669.9 | 1657.5 | 3860.0 | 2202.5 |
| shop | black-rose-portal-statue-640w.webp | 349.6 | 816.9 | 3402.6 | 3403.3 | 5792.6 | 5804.0 | 11.4 |
| pdp | sg-005-320w.webp | 388.8 | 589.4 | 596.8 | 597.3 | 1477.9 | 4280.0 | 2802.1 |

**Home:** The image is explicitly high-priority, eager, asynchronously decoded and discovered via a matching responsive image preload. Its 28,800 transferred bytes complete at 1657.5 ms; the recorded IMG LCP is 3860 ms. There are 19 render-blocking resources before LCP and 171 ms of pre-LCP long tasks. Video begins playing at approximately 7959.7 ms, after image LCP; the observer records no subsequent VIDEO candidate. Lighthouse instead selects VIDEO and reports 5874 ms simulated LCP. This is a method-dependent observation. The trace does not prove why Chromium retained the image candidate; do not invent a video eligibility, decode or timing cause, and do not equate late video start with the observed image LCP.

**Shop:** The approved portal frame is the LCP image, at high priority, with no preload. Fetch begins 816.9 ms, but request starts 3402.6 ms: approximately 2585.7 ms of pre-request delay. Transfer then runs approximately 2389.3 ms from response start to end for 124,228 bytes, with only 11.4 ms remaining to LCP. This supports investigating discovery/scheduling and competition on the local HTTP/1.1 path, plus the frame payload; it does not isolate a single cause of request queuing. Do not optimize away the founder-approved frame.

**PDP:** The approved responsive 320w product derivative is preloaded and high-priority. Its 29,266 transferred bytes complete at 1477.9 ms, while LCP is 4280 ms: a 2802.1 ms post-transfer gap. There are 21 blocking resources before LCP. More image compression alone does not explain this gap; CSS readiness, native gallery visibility/layout and scheduling require an isolated intervention before causality can be claimed.

**Collection, Cart and Checkout:** No matching actually throttled observer run is included here. Current unthrottled TTFB/LCP are 468.7/556 ms, 344.4/388 ms and 260/304 ms respectively. Collection selects the hero IMG; Cart and Checkout select their H1. A text LCP has no standalone image request/transfer/decode subparts. Use their Lighthouse reports and saved traces for simulated route comparison; do not transplant Home/PDP timings onto them.

## Server, CSS, JS, fonts, decode and third-party boundaries

Document TTFB is measured end to end on the local PHP fixture. It includes request/server scheduling; there is no Server-Timing breakdown proving DB, template or cache cost, and no production latency evidence. Image response-header intervals around 0.5–0.7 ms in the actually throttled observations do not show an image origin processing bottleneck. HTTP/1.1 local scheduling differs from a production HTTP/2 or HTTP/3 host.

Render-blocking text is a concrete shared burden. Lighthouse identifies native jQuery/jQuery Migrate and theme CSS among blockers; Home also includes native utility dependencies. Checkout additionally loads native Woo styles plus legacy/content/theme/controls/shell styles. Current CSS totals are approximately 147 KiB Home, 245 Shop, 135 Collection, 261 PDP, 289 Cart and 270 Checkout. Gzip diagnostic improvements support text transport as a consequential variable. Lighthouse estimated wasted milliseconds are model outputs and overlap; never sum them as measured savings. Native Woo ordering and inline dependencies must be preserved when testing defer or route scoping.

Current fonts transfer about 207 KiB on Home/Shop/Collection/PDP, 134 KiB Cart and 141 KiB Checkout. In the actually throttled runs font-ready occurs around 7067 ms Home, 6675 ms Shop and 7398 ms PDP, after each observed image LCP. This does not establish fonts as those images’ LCP blocker, nor establish that font payload is free: shared bandwidth/layout impacts require controlled tests. Do not preload every font.

Raw `traces/current-devtools-{home,shop,pdp}.trace.json` duration events were inspected. Across the entire capture, inclusive `Layout` totals are about 344/206/148 ms, `EvaluateScript` 171/80/83 ms and `ImageDecodeTask` 13.6/22.5/7.3 ms for Home/Shop/PDP. These are trace-category observations, may overlap across threads/nesting, and are neither a pre-LCP critical-path sum nor GPU execution measurements. They do not justify assigning the 2.2–2.8 second Home/PDP render gaps solely to image decode. `decoding=async` describes requested behavior, not measured decode completion.

All current Lighthouse resource summaries report zero third-party requests/bytes. That means this local fixture provides no evidence about real analytics, consent, payment provider or production third-party cost. Native Woo sourcebuster/order attribution still counts as same-origin JS; zero third-party is not permission to remove analytics.

## Bounded experiments and disposition

The native-only defer experiment is **REVERTED / REJECTED_FROM_CANDIDATE**. Its isolated Home 6029→5998, Shop 6252→5947 and PDP 5429→5495 ms observations did not establish reliable benefit. Preserve the accepted Core/native delivery policy and CSP template cache. A changed native delivery policy would require complete native-commerce regression proof before selection. See `native-only/decision.json`.

The local VP9 codec experiment is **REVIEW_ONLY_NOT_SELECTED**, not wired or promoted. The corrected candidate retained 1440×810, 24 fps and 8 seconds without crop: 290,311 bytes versus 676,790 bytes (about 57% less). SSIM all 0.986823 and luma 0.980522, with independent sampled 1/4/7-second visual equivalence and subtle dark-grain smoothing, are not temporal/lossless certification. The mistaken first 1920 encode is excluded. Isolated gzip Home LCP was 3997.625 ms versus approximately 3997 ms before: **no LCP gain demonstrated**. Original approved delivery remains. See `codec-experiment/receipt.json` for hashes and disposition.

## Remaining work and certification boundary

1. Preserve the canonical gates and repeat controlled, same-profile runs before claiming speed gains; do not replace failures with the gzip diagnostic.
2. Isolate critical CSS/native dependency scheduling and transport changes, beginning with Checkout and Cart’s quiet surfaces, with native purchase regression coverage.
3. Investigate Shop frame discovery/queuing without changing approved card identity; investigate Home/PDP post-transfer rendering with one controlled variable per experiment.
4. Retain nine scenes, approved hero motion and Ask Skyy. Codec candidates require temporal review and a measurable performance justification before selection.
5. Complete field INP and route animation render-work evidence; report GPU timing and production/CDN behavior as unknown until actually measured.

Evidence authority: `performance-summary.json`, matching `corrected-baseline/`, `current/`, `compressed-current/` Lighthouse reports and trace files; `traces/current-390-*` and `traces/current-devtools-*`; `tasks/v2-cinematic-finalization-20260906/performance-budgets.json` and `performance-budget-results.json`; the two experiment receipts above. This documentation does not alter source, budgets or experimental dispositions.

## Global and route asset discipline

The external files actually requested on all six measured routes total **152.9KiB JavaScript and93.7KiB CSS**. Additional files are route-specific or shared by a subset of routes; these groups must not be mistaken for six independent unique bundles. Inline code remains part of Document bytes. The per-file owner/path/request map is in [bundle-inventory.json](/Users/theceo/.codex/worktrees/7116/DevSkyy/.artifacts/v2-cinematic-finalization-20260906/bundle-inventory.json).

| Route | Additional JS KiB | Additional CSS KiB |
|---|---:|---:|
| home | 59.0 | 53.3 |
| shop | 40.8 | 151.1 |
| collection | 51.3 | 40.8 |
| pdp | 135.0 | 167.2 |
| cart | 106.5 | 195.7 |
| checkout | 106.0 | 175.8 |

The final build contains19CSS and13JS generated outputs. This is output count, not a claim that all are requested on each route. The new premium behavior has one initializer and ownership for native card feedback, selected navigation preview and opt-in typography. Superseded native-deferral experiment files were removed from shipping source. Existing native Woo styles/scripts were not blindly unloaded from initial-viewport coverage because later variation/cart/dialog states and extensions use them. Remaining CSS budget failures are explicit.
