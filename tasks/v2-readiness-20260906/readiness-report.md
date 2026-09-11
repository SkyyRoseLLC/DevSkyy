# V2 engineering readiness — NEEDS_MORE_WORK

Candidate: **V2-READINESS-20260906**, continued from accepted **V2-CINEMATIC-FINALIZATION-20260906**. Creative direction is frozen. This report closes the authorized engineering/readiness pass; it does not certify release readiness.

Branch `codex/v2-cinematic-ooda-20260906`; HEAD `aabd2bffdc1e322862acd05a5640c14cf00f3acf`. Source inventory: 86 files, digest `4a23629d6b17c1d0393000cfbbe982eedf47712837dd772a66c993ad7e46bbe7`. Parent digest `ef1dcdb414d57374e6b9ee1bd84641eec0e59c102041f705fdaa88c2b1f5a8a8`. All 397 protected binary files remain unchanged. Approved scene, card, opening-media and hero manifests match; Town Line source matches. Existing unrelated dirty work is retained.

## Engineering changes

Eleven theme files differ from the frozen baseline, including generated outputs and the new font manifest/subset. Cart and Checkout omit unused legacy-world/content CSS. Shop and product taxonomy routes omit only Woo layout/small-screen CSS, retaining native general styling and an explicit opt-back-in filter. A scoped positioning rule preserves the existing Shop grid. Build outputs were regenerated.

Approved Hanken/Inter typography is preserved. An independently reproducible, licensed 1,872-byte Inter fallback subset replaces the 48,432-byte full fallback request for eleven glyphs absent from Hanken. Source font is untouched. Outline and advance-width parity passed at four weights. Tiny arrow raster differences were independently judged EQUIVALENT; primary-Hanken failure falls back to system Latin, not the original full Inter face. Broad font removal and removal of all Woo CSS were rejected after comparison.

Skyy computes exact bounds in cooperative chunks, deduplicates texture preparation, releases temporary model-buffer references and owns cancellation across shader preparation. An independently reproduced vendor compileAsync teardown problem was fixed using synchronous compile followed by owned cancellable work. No approved model, rig, poster, animation, composition or identity changed. Native Woo scripts were not speculatively deferred.

The local fixture's theme-root isolation was repaired in four request-local routers after WordPress core was found reading another worktree's theme.json. Final profiles and comparisons were rerun afterward. The shared fixture symlink and catalog were not modified. See [fixture isolation](fixture-isolation.md).

## Same-profile mobile performance

Lighthouse 13.4.1; Lantern simulated mobile, 390×844, DPR 1, CPU 4×, 150ms RTT, 1,638.4Kbps. Each cell is one cold-context sample, not a median or a field measurement. Baseline and candidate use the same repaired fixture and profile. Gzip is a separate local transport experiment.

| Route | Baseline plain LCP | Current plain LCP | Plain class | Current gzip LCP | Gzip class |
|---|---:|---:|---|---:|---|
| Home | 5,878ms | 5,484ms | FAIL | 3,697ms | FAIL |
| Shop | 7,143ms | 6,025ms | FAIL | 3,329ms | FAIL |
| Collection | 5,338ms | 4,513ms | FAIL | 3,692ms | FAIL |
| PDP | 5,506ms | 5,503ms | FAIL | 2,528ms | NEAR PASS — hard gate fails |
| Cart | 5,745ms | 5,730ms | FAIL | 2,454ms | PASS — LCP only |
| Checkout | 4,820ms | 4,518ms | FAIL | 2,130ms | PASS — LCP only |

PASS requires LCP ≤2,500ms. NEAR PASS is >2,500–2,750ms and does not pass the release gate. Production/staging delivery is **ENVIRONMENT-BOUND / unverified**, not inferred from local gzip. Current CLS is below 0.1 on all routes.

Remaining current plain route budgets: Home initial transfer 1,476.6KiB >1,400; Cart CSS 246.9KiB >240; Checkout initial 763.7KiB >700, JS 258.9KiB >240 and CSS 227KiB >180. Shop CSS now passes at 216.4KiB against 230. All routes still fail plain LCP. Other sampled total transfer, image/video, request and long-task limits passed; field INP, peak memory and sustained animation GPU cost remain unverified.

[Performance report](performance.md) contains route-level TTFB, discovery/transfer/render observations, LCP candidate changes, every font request, CSS/JS ownership and Woo/third-party costs. Unthrottled trace subparts are explicitly separate from simulated Lighthouse LCP. Home's observed current video candidate follows first playback; smaller bytes alone were not treated as causal proof. Scene scheduling retains zero films before arrival.

Theme owns conditional enqueueing, correct responsive assets and valid cache-busting. Web server/CDN own negotiated gzip/Brotli, Vary, content types/ranges and caching; immutable caching requires content/version keys that change with bytes. Personalized cart/checkout HTML must bypass shared caching. WordPress.com platform delivery must be verified with actual response headers after separate authorization. No hosting change was made. Full responsibility table and official references are in the performance report.

## Ask Skyy

Two matched Metal Chromium runs per condition reduced maximum startup long tasks from **511/315ms to 50/54ms**. First CPU render fell from 262.9/151.5ms to 32.2/34.5ms. First stable frame from runtime boot fell from 2,003.9/1,740.5ms to 1,769.0/1,590.4ms. This clock excludes loader activity before runtime boot and subsequent reveal/walk completion; it is not end-to-end invitation latency. First visible 220×340 RGBA frames were byte-identical in both pairs. Separate profiles recorded one model request per invitation. Walk → idle → local shipping chat and shader-time context-loss/pagehide fallback checks passed.

The asset remains **6,058,568 bytes, 1,930,256 triangles, 1,030,595 vertices, 18 bones, one material/draw call**, approximately 89.2MiB decoded geometry/texture lower bound and 7.38MB/13 requests. These exceed current model/runtime/request budgets. Idle 14.75fps with p95 interval 83.3ms and conversation 28.91fps with p95 50ms fail the intended 15/30fps cadence interval gates. CPU submission p95 was 0.4/0.3ms. Actual Apple M5 Metal backend was verified; physical-mobile GPU execution and peak-memory certification were not performed.

[Runtime evidence](skyy-runtime.md) and [exact Blender handoff](skyy-blender-handoff.md) pin paths, hashes, scale/origin, camera/light, 18-bone structure, material/textures, animations and poster state. The handoff targets under ~100k triangles where fidelity permits and GLB ≤2.5MiB, with explicit identity/rig/visual gates. Blender 5.2 LTS was available; asset authorship was deferred and the model was not altered. Fresh review recordings demonstrate motion/chat but are not cadence benchmarks. The desktop chat capture includes a partially scrolled earlier line and the mobile capture shows the latest response; these are not proof of every transcript line being visible. Their filename-based modelRequests counter is invalid and explicitly annotated in `skyy-review/measurement-boundary.md`.

## Content and media authority

All 33 SKUs were audited across short description, material, fit, care, story, gallery, size, product type and preorder state. **33 PARTIAL; zero COMPLETE.** Short descriptions: 23 complete/10 partial; material 4 complete/29 partial; fit 6 complete/27 partial; care 33 missing; story, size and product-type proof 33 partial; gallery 28 partial/5 missing; preorder 33 complete, including 15 preorders. Ten source conflicts remain editorial decisions. The synthetic native database has empty short/full descriptions for all 33 products and does not establish production content readiness.

[Content audit](content-audit.md) and [review queue](content-review-queue.json) contain every SKU and reason. Rejected PDP imagery remains rejected for **BR-001, BR-003, BR-004, BR-007, BR-011**. **CREATIVE ASSET BLOCKER — BR-003** remains. Card approval does not authorize PDP promotion. No missing content or media authority was invented.

## Verification and review material

Full build and full verify passed, including 95 JavaScript tests and PHP/source/generated/native-gallery checks. Fresh responsive checks: 204 across Chromium (136), Firefox (34), WebKit (34), with zero reported overflow, missing assets, page errors or axe violations. Fresh mobile art checks: 72 scene/dialog/keyboard/reduced-motion states. Native Shop filtering, sorting, URL/history and JS-off behavior; PDP variations/gallery; Shop → SG-005 M → Bag → Cart → Checkout review; quantity changes and invalid coupon responses passed. Four baseline/current × viewport transaction comparisons produced eight state records. The fixture Checkout displays its no-payment-methods notice. No order or payment was submitted.

Ten fresh hero frames were captured after actual playback, not seeking. Fresh Skyy desktop/mobile walk/chat recordings and preserved first-frame comparisons are included. The nine scenes are represented by 18 explicitly inherited desktop/mobile handoff recordings, individually hash-checked against their prior index; current scene-delivery and handoff checks are fresh. The review page distinguishes inherited recordings from new captures. Independent source and visual judgments are in [independent review](independent-review.md) and [visual freeze](visual-freeze.md).

Manual native Chrome checks used actual **200% browser zoom**, keyboard focus through menu/search/results, native size selection/reset, Bag focus cycling and dismissal, and Ask Skyy input/chat/dismissal. Zoom was restored and the dedicated tab closed. Screens were inspected through native UI tooling; no local screenshot files are claimed for that manual session. See [manual accessibility](manual-accessibility.md). Physical iPhone/Android and actual screen-reader testing were unavailable/unperformed; browser emulation and axe do not replace them.

Evidence exceptions are retained rather than hidden: one supplemental Checkout observation initially timed out and only that missing capture was retried; one Cart harness predicate failed during native AJAX DOM replacement and was corrected with a null-safe predicate; the original logs remain. A scene helper ignored its output override and overwrote the previous candidate's scene-delivery JSON. Its prior expected hash and current hash are recorded in `evidence-routing-incident.json`; the new provisional result is retained, and correct-baseline fresh delivery evidence is under `scene-delivery-final/`. The prior JSON is not claimed intact. No protected media was affected.

## Exact readiness matrix

| Surface | Status | Evidence / boundary |
|---|---|---|
| SOURCE | PASS — local integrity | 86 source hashes; 397 protected binaries unchanged; prior evidence JSON exception disclosed separately |
| BUILD | PASS | Full build and verify; 95 JS tests plus PHP/source/native checks |
| HEROES | EQUIVALENT | Five preserved films/compositions; ten fresh playback captures |
| SCENES | EQUIVALENT | Nine preserved scenes; delivery and commerce handoffs pass |
| PRODUCT CARDS | EQUIVALENT | Approved authority/treatment preserved; responsive checks pass |
| QUICK VIEW | PASS — native / visually preserved | Native selected/reset variation states and matched screenshots |
| PDP | PARTIAL | Native purchase behavior passes; content and five media authority gaps remain |
| SEARCH | PASS — local | Preview, empty, keyboard/focus and responsive checks |
| NAVIGATION | PASS — local | Menu, URL/history, dialogs and focus return |
| ASK SKYY SOFTWARE | PASS — scoped runtime | Exact continuity, lifecycle fallback and interactions; performance limits below remain |
| ASK SKYY ASSET PERFORMANCE | FAIL | Model size/geometry/runtime requests and frame cadence exceed gates |
| MOBILE PERFORMANCE | FAIL | Six plain LCP failures; additional route budgets; gzip only Cart/Checkout LCP pass |
| ACCESSIBILITY | PARTIAL | Local automated/manual 200%/keyboard checks pass; physical-device and screen-reader unverified |
| COMMERCE | PASS — synthetic local | Native cart/variation/coupon/checkout review only; no live payment certification |
| MEDIA AUTHORITY | BLOCKED — rejection truth preserved | Five approved-PDP-media needs; BR-003 creative asset blocker |
| TOWN LINE PRESERVATION | PASS — preserved only | Exact protected source; no final authoring |

## Remaining owners and stop state

Frontend/performance engineering owns Home/Shop/Collection critical path and remaining Cart/Checkout route budgets, with the same profile and visual gates. Hosting owner owns later real gzip/Brotli/cache/CDN/header verification; local gzip is not deployed behavior. Blender production owns faithful model reduction and a fresh cadence/peak-memory/physical-device comparison. Editorial/founder content review owns ten source conflicts, missing care/material/fit/story details and the five approved PDP media decisions. Device/accessibility QA owns physical iPhone/Android and screen-reader verification.

**NEEDS_MORE_WORK** is retained because major route and Skyy asset/cadence failures remain, despite stable creative work and passing functional tests. No deployment, staging modification, release package, promotion, paid generation or source-model alteration occurred. Stop after this report.
