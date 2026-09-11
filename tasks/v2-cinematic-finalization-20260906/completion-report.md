# SKYYROSE V2 CINEMATIC FINALIZATION REPORT

## Status

**NEEDS_MORE_WORK**. The continuation has a complete local review set and substantial verified software improvements. It does not meet the final mobile performance or asset-completeness gates. No deployment, staging update, package creation, promotion, paid generation, model redesign or Town Line authoring occurred. This is an uncommitted local candidate; no founder approval is inferred.

## Source

- Candidate: `V2-CINEMATIC-FINALIZATION-20260906`, continuing `V2-CINEMATIC-OODA-20260906`.
- Worktree: `/Users/theceo/.codex/worktrees/7116/DevSkyy`.
- Branch: `codex/v2-cinematic-ooda-20260906`.
- Starting and ending commit: `aabd2bffdc1e322862acd05a5640c14cf00f3acf`.
- Accepted parent digest: `35ea52d179e260ad3fe23084055e8aab0b3aae05506aaa434510abb0ad9d2d63`.
- Final candidate digest: `ef1dcdb414d57374e6b9ee1bd84641eec0e59c102041f705fdaa88c2b1f5a8a8`.
- Scope: 78 source/test/build files, canonical JSON hash-map digest; 394 pre-existing media/font/model assets byte-identical to the accepted snapshot. Only new runtime media is the same-model Skyy frame, locally encoded as lossless WebP.
- Source guard retained. Final Skyy markup hash and its build-input binding reconciled; translation catalog rebuilt to 598 singular messages / 1 plural.

[Source hashes and preservation proof](/Users/theceo/.codex/worktrees/7116/DevSkyy/tasks/v2-cinematic-finalization-20260906/source-integrity.json) · [Interactive local review index](/Users/theceo/.codex/worktrees/7116/DevSkyy/.artifacts/v2-cinematic-finalization-20260906/review.html) · [Independent source review](/Users/theceo/.codex/worktrees/7116/DevSkyy/tasks/v2-cinematic-finalization-20260906/independent-review.md)

## Performance Root Cause

The fast route observations and slower Lighthouse scores measure different conditions. The direct captures run against local PHP without network/CPU emulation. Lighthouse 13.4.1 models a 390×844 mobile device, DPR 1, CPU ×4, RTT 150ms and 1638.4Kbps using Lantern. Its observed LCP breakdown is not the same as its simulated final LCP. A separate real DevTools network/CPU-throttled run is also retained. These are single paired samples, not a statistical or production CWV certificate.

The original preview router omitted WebM MIME support and returned 404, making the browser fall back to the 840 KB MP4. Both baseline and current routers were corrected before the comparison below. Historical 4.84/5.78/5.35s results and early experiments are preserved but are not treated as comparable baselines.

The current critical path combines local WordPress response time, native/theme CSS and JavaScript, font/image discovery and mobile transfer modeling. Gzip-only isolation materially lowers simulated LCP but does not bring every route to target, so the residual cannot honestly be assigned entirely to the server. The initial scene films are not responsible: final scene scheduling requests zero scene videos before arrival. The Home hero is a separate mandatory film. Initial3D loading was removed from the untouched Home path by requiring character intent.

The actual unthrottled Home LCP changed from poster to VIDEO around 880 ms. With real DevTools throttling, the observed LCP was IMG at 3860 ms, while video began around 7959.7 ms and no subsequent VIDEO LCP entry appeared in the retained observation window. This method-dependent difference does not prove that animation always starts early, nor establish why the later film was not reported as another candidate.

Two isolated experiments were rejected: native-script deferral produced no reliable cross-route LCP improvement and was restored byte-for-byte; a 57% smaller local VP9 hero encode retained sampled composition but yielded essentially the same ~4.0 s gzip Home LCP. That encode remains review-only and is not wired. No cinematic feature was deleted.

[Detailed trace analysis and boundaries](/Users/theceo/.codex/worktrees/7116/DevSkyy/tasks/v2-cinematic-finalization-20260906/performance.md) · [Machine-readable measurements](/Users/theceo/.codex/worktrees/7116/DevSkyy/.artifacts/v2-cinematic-finalization-20260906/performance-summary.json) · [Budgets](/Users/theceo/.codex/worktrees/7116/DevSkyy/tasks/v2-cinematic-finalization-20260906/performance-budgets.json) · [Budget results](/Users/theceo/.codex/worktrees/7116/DevSkyy/tasks/v2-cinematic-finalization-20260906/performance-budget-results.json)

## Performance

Same corrected fixture and Lighthouse mobile profile. Gzip is an isolated local transport experiment, not a production/CDN result. LCP target ≤2.5 s, CLS ≤0.1. “Measurement completed” is not a budget PASS.

| Route | Baseline LCP | Current LCP | Current + gzip | Current CLS | Baseline → current bytes |
|---|---:|---:|---:|---:|---:|
| Home | 6.03s | 5.87s | 4.00s | 0.05522 | 2,060,131 → 1,629,263 |
| Shop | 6.25s | 5.87s | 4.76s | 0.00054 | 1,096,564 → 1,119,664 |
| Collection | 4.21s | 5.27s | 3.54s | 0.00032 | 2,226,535 → 1,677,343 |
| Pdp | 5.43s | 5.51s | 2.60s | 0.00128 | 1,208,599 → 1,213,204 |
| Cart | 6.02s | 5.74s | 2.44s | 0.00061 | 862,081 → 873,102 |
| Checkout | 4.82s | 5.12s | 2.42s | 0.00000 | 829,676 → 830,916 |

All six uncompressed LCP gates remain FAIL; all six CLS gates PASS. Collection, PDP and Checkout LCP regressed in the single comparable run; lower bytes do not erase that result. Gzip Cart/Checkout meet 2.5 s in their samples, PDP is 2.60 s, and the remaining routes still fail. Route budget failures also include Home initial bytes, Shop/Cart CSS, and Checkout initial bytes/JS/CSS. Field INP and GPU animation cost remain UNVERIFIED.

## Animated Heroes

Home, Signature, Black Rose, Love Hurts and Kids/The Heir retain their exact approved films/posters. Chromium: 25 current captures at 320/390/414/768/1440; 10 matched baseline captures at 390/1440. Captures use actual presented frames near 1.00–1.06s, then pause without seeking. Firefox: 5/5 actual-frame captures. WebKit autoplay raised NotAllowedError; all 5 retained the still and played after the native Play motion action. That is policy-fallback evidence, not autoplay or physical-Safari certification.

First-frame reveal now waits for the composited video frame, avoids duplicate initialization and respects reduced motion/Save-Data/visibility. The per-hero independent visual verdict is in [Hero review](/Users/theceo/.codex/worktrees/7116/DevSkyy/tasks/v2-cinematic-finalization-20260906/heroes-mobile.md). These frames do not certify every loop seam or GPU cadence.

| Hero | Matched normal-motion assessment | Independent observation at 390 and 1440 |
|---|---|---|
| Home | **EQUIVALENT** | Approved moon/bridge/rose-star film, crop, dark text scrim, headline, body copy and purchase/discovery actions retained. The current same-model static guide replaces the baseline capture's original-portrait motion-failure presentation. That is a changed renderer state, not proof that a 3D animation improved. No hero composition degradation seen. |
| Signature | **EQUIVALENT** | Golden Gate horizon, copper monuments, lit platforms and water remain in the same composition. Mobile keeps the center view with monument edges cropped as in baseline; separate collection title/copy and both native CTAs remain legible. Desktop retains the full two-monument setting. |
| Black Rose | **EQUIVALENT** | Silver metal lettering, bridge cables, moon, rose/star relief and dark reflections remain readable at desktop. Mobile retains the bridge/star emphasis and the same partially cropped left monument, with complete collection heading and clear commerce/story actions below. |
| Love Hurts | **EQUIVALENT** | Central aisle, back-turned figure, bell jar, crimson wordmark/star and reflective floor retain their approved positions. Desktop preserves the full environmental composition; mobile retains the same center-weighted crop with readable separate title/copy and CTAs. |
| Kids / The Heir | **EQUIVALENT** | Character face/hair/clothing, throne, warm lighting and The Heir lettering retain the same framing. Mobile and desktop lower-body crop is unchanged from baseline. Separate collection identity and commerce continuation remain readable where within the captured viewport. |

## Nine Scroll Worlds

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

Each of the nine scenes independently passes asset/cast identity, contained desktop/mobile composition, playback/pause, CTA, handoff, keyboard, Save-Data, reduced motion, no-JS, loading/error stills, reverse/rapid scroll, resize/orientation and simulated lifecycle cases. 90 main cases plus 9 touch/document-hidden cases passed. 12 controller-blocked/init-failure checks prove the approved posters return if the external controller fails.19authority tests bind nine choices and 27 media hashes.

All 18 requested current scene recordings exist: each scene at 390 and 1440, showing arrival, progression, CTA and merchandise handoff. Exact per-state evidence and limitations: [Nine-scene report](/Users/theceo/.codex/worktrees/7116/DevSkyy/tasks/v2-cinematic-finalization-20260906/scenes.md).

## Scene → Commerce

The approved cinematic frame resolves into its existing merchandise section with a 40 ms heading/grid separation and 280 ms restrained transition. First card alignment, native destinations and reduced-motion visibility remain intact. Strict nearby-poster scheduling reduces initial scene-poster bytes: Home 503,372→255,858; Signature 830,156→255,858; Black Rose 493,290→133,642; Love Hurts 541,486→113,872. Loaded scenes are retained for reverse scrolling. These are scene-only request-window savings, not whole-page or LCP causality claims.

## Product Cards

| State | Result | Evidence / scope |
|---|---|---|
| Default | PASS | Matched before/current approved card captures for Signature, Black Rose/Jersey, Love Hurts and Kids at 390 and 1440. |
| Hover | PASS | Four current desktop hover captures; retained frame/front/crop and native action. |
| Focus | PASS | Four desktop focus captures; visible Quick View ring; native dialog semantics. |
| Touch | PASS | Native links and pointer state remain progressive; touch Quick View/navigation mechanisms use standard controls. Navigation tap explicitly exercised. |
| Image loading | PASS controlled network fixture | Root held approved local card-front requests, captured loading, then aborted and captured intentional image failure. Card height remained866.46875px in both states; `image-loading.json` and matching screenshots. |
| Image failure | PASS fixture | Browser image failure reveals intentional unavailable copy without inventing another product image. |
| Available | PASS | Native Woo labels and current representative card data. |
| Out of stock | PASS rendering fixture | Canonical PHP test emits native unavailable label. Sampled live fixture products are available. |
| Pre-order | PASS | Current BR-003/Jersey card visibly displays native-authorized Pre-order edition; PHP state test also passes. |
| Sale | PASS rendering fixture; N/A sampled live data | Woo `is_on_sale()` and native price markup retained. No sale price fabricated. |
| Quick View open | PASS | Opener aria-expanded/state; five-width keyboard trap and Escape/focus restoration. |
| Quick View loading | PASS | Request loading/busy state and source-card state; canceled/stale response tests. |
| Add success | PASS native purchase / fixture inline event | Real QV POST and native success notices; inline simple-card state follows native added_to_cart only. |
| Add failure | PASS native transport fixture | Native Woo AJAX failure traced to exact initiating jqXHR/card; localized feedback; only originating busy state clears. Two in-flight requests and unrelated request isolation pass. |

The native transport error test explicitly changed a local variable-card control into a simple-card AJAX control, then aborted the native POST. This tests actual Woo transport/event behavior without creating a simple catalog product or mutating the cart. It is not evidence that production has a simple product of that SKU.

## Quick View

Architecture retained: eligible published same-origin PDP → inert native form extraction → native variation runtime → native POST → native Woo confirmation. The final actual-built native bridge test used **no response overrides**: size M in QV added successfully, followed by size L on the native PDP adding successfully. `aria-disabled` became `false` on native variation resolution. JS page errors and CSP errors were both empty. See `final-native-built.json`.

Visual verdict: **BETTER**, engineering review. Desktop no longer inherits full-PDP section padding inside the modal; mobile full garment and title do not overlap; title wraps are materially shorter at large desktop; quantity, native purchase action, secondary PDP link and persistent exit remain clear. Founder acceptance is still required. Native close transitions degrade safely; reduced motion disables optional animation.

The final 320 px panel intentionally scrolls vertically: the full garment remains visible, Size begins near the lower edge and purchase controls continue below. Persistent exit and keyboard traversal remain available; all controls are not claimed above the fold. The final built native QV M→PDP L purchase test was repeated after reverting the script experiment.

## PDP

| SKU | Collection / garment | Native type | Media | Native stock |
|---|---|---|---|---|
| SG-005 | Signature / shirt | Variable | One permitted commerce view | Available |
| BR-003 | Black Rose/Jersey / baseball jersey | Variable | Rejected, zero views | Available / preorder edition |
| BR-006 | Black Rose / bomber sherpa | Variable | One permitted commerce view | Available |
| LH-002 | Love Hurts / joggers | Variable | One permitted commerce view | Available |
| KIDS-001 | Kids / hoodie set | Variable | One permitted commerce view | Available |
| SG-013 | Signature / crewneck | Variable | One permitted commerce view | Available |

Silhouette → details/story → purchase navigation is wired to native regions and omits unavailable silhouette links. Native hook order and one-time excerpt placement pass their existing PHP regression. Simple and unavailable cases are rendering/behavior fixtures, not real representative PDPs in this synthetic database. Multi-view/full material/craft content is **not certified**: these six fixtures expose Size only and empty short descriptions. No generic craft prose was substituted.

BR-003: **CREATIVE ASSET BLOCKER — BR-003**. An approved card campaign image is not PDP-media approval. Explicit rejection remains authoritative.

The 33-SKU source census is [Recorded separately](/Users/theceo/.codex/worktrees/7116/DevSkyy/tasks/v2-cinematic-finalization-20260906/pdp-source-coverage.json). Opening-media raw statuses are 16 stale / 9 missing / 5 explicitly rejected / 3 unspecified(default approved in the existing validator). They are not equivalent to final native-gallery availability. The resolver separately permits verified editorial/native attachments and refuses explicit rejection. Existing rejected entries include BR-001, BR-003, BR-004, BR-007, BR-011; none received an invented replacement.

## Search / Navigation

Live product previews and native GET search remain intact. Actual query and no-results states were exercised at 320/390/414/768, including keyboard trap, Escape and focus restoration. Source tests cover debounce, empty input, error/abort handling and safe same-origin results. Collection directory previews now decode one approved still on hover/focus or explicit touch Preview. Links navigate immediately; no extra videos are preloaded. Reduced-motion and no-JS links remain usable.

Typography vocabulary is intentionally mapped: commerce reveal→QV identity; mask→directory heading; editorial→authoritative PDP house note; collection split→below-fold Scroll World heading; statement→rare chapter-three titles. Critical headings remain initially visible. Only transform/opacity/clip-path motion is used. Selective ink/glass/metal treatment strengthens QV, directory and card states without adding competing controllers or a page loader. Checkout stays restrained.

## Ask Skyy

PASS refers to the stated local software observation, not universal hardware certification.

| State | Result | Actual evidence |
|---|---|---|
| LOADING |PASS|`skyy:3d-loading` from real invitation/Home initialization; loading diagnostic capture|
| STATIC PLACEHOLDER |PASS|Same-model poster before renderer readiness; reduced-motion and static diagnostic captures|
| 3D PREPARING |PASS|Real module/model preparation; separately held model request in diagnostic C|
| WALKING IN |PASS|Real existing-rig `Skyy_Walk`, phase progression, desktop/mobile review recordings|
| TURNING |PASS|Observed `walking-in → turning → idle` phase sequence in the continuity test|
| IDLE |PASS|Waited for actual `Skyy_Idle` clip and idle phase; no arbitrary screenshot delay|
| GREETING |PASS|Native greeting and “hello Skyy” response invoke the existing wave clip|
| LISTENING |PASS|Actual input event marks listening; desktop/mobile browser rows|
| THINKING |PASS|Actual synchronous local lookup emits thinking before the answer; contract test and event trace. No artificial loading animation is claimed|
| TALKING |PASS|Shipping answer invokes existing talk clip; screenshots and recordings|
| GESTURE |PASS|SG-005 discovery invokes the existing joy gesture and native product link|
| CHAT OPEN |PASS|Native dialog, the same character stage, focus on the question input|
| CHAT CLOSED |PASS|Native close/Escape, stage restoration, conversation retained|
| MINIMIZED |PASS|Minimize closes the dialog, records minimized state and preserves transcript for recall|
| PAUSED |PASS|Actual frame counter stays fixed while paused; Pause/Resume controls retained|
| DISMISSED |PASS|Home dismissal hides the stage and restores header recall focus|
| OFFSCREEN |PASS|Actual scroll outside the Home stage, followed by observed stopped rendering and stable frame count|
| DOCUMENT HIDDEN |PASS, controlled event|Synthetic visibility event against a real initialized renderer proves lifecycle handling and stopped frames; no physical tab-background timing claim|
| REDUCED MOTION |PASS|Browser preference override: no model request, static guide, functional text interaction; dynamic restoration alignment also tested|
| SAVE DATA |PASS, controlled connection|Connection override: no model request, static guide, functional text interaction|
| WEBGL FAILURE |PASS, injected fault|Blocked WebGL2 context: no model request, visible fallback and usable question input|
| MODEL FAILURE |PASS, injected fault|Aborted canonical model request: explicit fallback and functional text interaction|
| CHAT FAILURE |PASS, injected fault|Missing guide-data contract: honest unavailable message and Contact destination; no fictitious remote chat endpoint|

31 focused DOM/runtime contract tests pass, including the quantized 60Hz cadence regression, retained transcript after minimize, safe product links, late-loader/fallback behavior, genuine bone movement and source-clip preservation. These tests are not substituted for browser/GPU evidence.

## Ask Skyy Performance

| Metric | Current measured value |
|---|---:|
| Bootstrap loader JS |3,040bytes|
| Guide JS |9,839bytes|
| 3D controller JS |12,440bytes|
| Three/loaders/utility/Draco JS+WASM |1,191,939bytes|
| GLB |6,058,568bytes|
| Rendered triangles |1,930,256|
| Draw calls |1|
| Materials |1|
| Bones |18|
| Source textures |3, each1024×1024|
| Embedded JPEG texture bytes |356,700|
| Source animation bufferView union |1,824bytes|
| Typed geometry arrays |76,754,012bytes|
| Decoded RGBA8 texture/mip estimate |16,777,216bytes|

Geometry plus estimated texture storage is 93,531,228bytes (89.20MiB). This excludes driver allocations, decoder buffers, the JavaScript heap and transient allocations. Renderer memory reports four GPU textures, including runtime-created resources, whereas the source model has three textures. Texture filenames say 8k; their measured dimensions are 1024×1024. Source animation bytes are embedded in the GLB and must not be counted as another network download. The six source clips are held poses; existing website code derives genuine quaternion movement on the same rig without editing those clips or the GLB.

The final scoped runtime resource list totals **7,375,138 transferred bytes across 13 requests** for loaded 3D, including bootstrap, character CSS, the header portrait and the 61,276-byte same-model stage poster. This is not a whole-route transfer figure. Resource Timing supplies network data; decoded GPU/driver memory remains an estimate. The preserved earlier profile excluded the stage poster from its filter and recorded 7,312,897bytes; it is historical evidence, not the final total.

Serial fresh Chromium contexts, 390×844 viewport, unthrottled local fixture, no other browser/build workload. Primary renderer: `ANGLE (Apple, ANGLE Metal Renderer: Apple M5, Unspecified Version)`. The preview router had already been corrected to serve the approved WebM hero. The initial LCP observation window is 3.5s before programmatic scroll; later candidates were also retained. These are single local samples, not Lighthouse or physical mobile measurements.

| Diagnostic state | Initial / later Home LCP | First stable 3D frame | Render-call CPU p95 | Mean / p95 frame interval | Startup long task |
|---|---:|---:|---:|---:|---:|
| A — character disabled, diagnostic only |880 /880ms|N/A|N/A|N/A|69ms|
| B — static placeholder |612 /612ms|N/A|N/A|N/A|None observed|
| C — 3D loading, model deliberately held |784 /784ms|Not yet rendered|N/A|N/A|None observed|
| D — 3D idle |636 /636ms|1,663.6ms|0.6ms|71.87 /83.3ms|310ms|
| E — conversation active |624 /624ms|1,622.1ms|0.5ms|36.39 /50ms|293ms|

All five initial observation windows recorded **zero GLB requests**. C/D/E then explicitly hover the character; D/E first-frame time starts with runtime initialization after that intent. The previous automatic Home preparation began before interaction; it has been repaired. The final profile began after root confirmed no other browser runners. An overlapping interim run is explicitly retained as a diagnostic and excluded from primary claims.

Every observed LCP remained the same Home VIDEO candidate. No late character-induced LCP replacement was observed in this controlled local run. The differences between single A/B samples are not interpreted as evidence that adding Skyy improves LCP. Idle delivered about 13.91fps against the 15fps target; conversation delivered about 27.48fps against the 30fps target. CPU render submission timing is distinct from GPU execution cost and cannot certify the latter.

An earlier explicit SwiftShader capture took about 3.26s to first stable frame and was visibly slower. It is retained as a software-renderer diagnostic, not the primary performance certificate. A default headless invocation without the maintained Metal flags returned no WebGL2 context once; subsequent correctly configured Chromium, Firefox and WebKit all rendered3D.

The final untouched Home check requests zero optional 3D resources for 10 seconds. Hover, touch and keyboard intent each request the model once. The visible guide remains usable before3D starts. The 2.5 MiB model budget,900 KiB dynamic runtime budget,12 request budget,100 ms startup-long-task budget and active/idle cadence budgets remain FAIL. First stable frame ≤2 s passes only on this local Metal host. Estimated 89.20 MiB is a lower bound, not a PASS for 96 MiB peak memory.

## Ask Skyy Continuity

The 61,276-byte lossless WebP poster was rendered from the existing model, camera, lighting and initial idle pose. The first stable 3D frame is painted before the 300 ms transition begins; actual clip completion controls the walk/turn/idle sequence. At 390 and 1440, poster/frame bounds match at 330×510 with alpha bounds (55,43)–(275,475); silhouette overlap 99.8974%, mean opaque RGB difference 0.989/255 across SwiftShader and Metal. No model/rig/face/outfit edit occurred. Dynamic preference restoration resets the walk offset, avoiding the prior horizontal jump.

Matched mobile/desktop walk, idle/chat, greeting/talking, pause, minimize and dismissal recordings are in the review index. Preference and injected-failure stills remain separately indexed.

## Mobile Art Direction

The 72 dedicated captures at 320/390/414/768 include 36 scene compositions, 12 scene/grid handoffs and 24 commerce/search/Bag views, in addition to 25 heroes and separate Skyy/QV sets. Tall BR 1 and LH 3 retain full portrait composition; wide scenes preserve all cast rather than enlarging one garment through cropping. At 320 the wide scene appears smaller and long product links extend into ordinary scroll. At 768 a next-chapter peek remains intentional rail grammar. QV retains a full silhouette and scrolling native form; Search keeps a clear query/results hierarchy; Bag exposes native actions with visible close/focus. Skyy 320 was corrected to keep composer/actions visible with a focusable bounded transcript. These are explicit art-direction tradeoffs, not a claim of universal visual excellence from overflow tests.

[Mobile findings](/Users/theceo/.codex/worktrees/7116/DevSkyy/tasks/v2-cinematic-finalization-20260906/mobile-art-direction.md) · [Hero-specific mobile assessment](/Users/theceo/.codex/worktrees/7116/DevSkyy/tasks/v2-cinematic-finalization-20260906/heroes-mobile.md)

## Accessibility

204 responsive cases passed: Chromium 136, Firefox 34, WebKit 34, with zero recorded Axe violations/page errors. Final narrower Skyy checks add 320/414/768 Axe and transcript keyboard proof after its last changes. Native keyboard checks include QV 16 Tab steps, Search/Bag 12 Tab steps, Skyy 14 Tab steps, Escape/opener restoration, native variation choice, visible focus and reduced motion. Firefox/WebKit additionally rendered the actual 3D guide, returned a native product answer and closed by keyboard.

Physical handsets, browser-UI zoom, screen-reader output and complete manual heading-sequence certification were not exercised. Viewport reflow and injected connection/visibility events are named as such. No screen-reader certification is claimed.

## Commerce E2E

Native QV/PDP selected-size purchases, quantity 2, native success/error notices, cart remove/empty behavior, Search/no-results, Bag focus and published same-origin destinations passed. Exceptional tests cover no size, invalid Size submitted to the native server, in-memory unavailable variation, aborted fetch, 12 second product-fetch timeout, approved image held/failed, and native Woo transport failure tied to the exact initiating card. Sale/out-of-stock/simple-product branches use rendering or controlled transport fixtures because the six representative products are variable and available. No payment/order or fabricated gateway response was submitted.

Final full verification: 93 JavaScript tests passed, zero failed, plus PHP/native Woo, SOT/hash, approved media, marketplace, token, translations, generated assets and responsive card checks. [Final verification log](/Users/theceo/.codex/worktrees/7116/DevSkyy/.artifacts/v2-cinematic-finalization-20260906/verify-final.log). The independent reviewer found no remaining blocking source defect in the reviewed changes; this does not clear performance/content gates.

## Visual Acceptance

| Paid system | Final engineering visual judgment | Scope |
|---|---|---|
| Five heroes | EQUIVALENT individually | Matched actual frames390/1440; current mobile widths |
| Nine approved Scroll Worlds | EQUIVALENT individually | Exact retained artwork/cast; improved scheduling and handoff behavior |
| Representative four-collection cards | EQUIVALENT | Retained front/frame/crop, readable identity/price/actions |
| Quick View | BETTER | Improved full-silhouette and native purchase hierarchy |
| Skyy static→walk-in | BETTER continuity | Same-model poster/first-frame match; actual recordings |
| Skyy idle | EQUIVALENT identity/composition | Matched stills and preserved model |
| Skyy chat | BETTER | Frontal stage, minimize/recall, readable narrow composer |

The earlier Love Hurts card WORSE report was withdrawn after original-detail inspection showed complete price/actions. The rejected first QV layout and interim profile runs remain historical diagnostics, excluded from final acceptance. These judgments are review evidence, not founder approval.

## Town Line

Pre-order Town Line PHP and the existing BR chapter-three room, media and approval records remain byte-identical. No new final Town Line experience was authored.

## Deferred Blender

Measured follow-up backlog: produce a fidelity-preserving lower-detail asset targeting under 100k triangles from the current 1.93 million; reduce the 6.06 MB GLB toward the 2.5 MiB budget; preserve face/outfit/18-bone identity and validate skinning; bake/review suitable clips rather than relying solely on held source poses; audit UV/texture packing and decoder costs; recapture the exact runtime poster after any approved model change; measure real mobile GPU, peak memory and sustained thermal behavior. This is a future asset phase. No deep Blender work was started.

## Remaining Content Gaps

**CREATIVE ASSET BLOCKER — BR-003** remains explicit. Other preserved opening-media rejections: BR-001, BR-004, BR-007, BR-011. Approved card imagery does not automatically authorize PDP media. Rich multi-view material/craft/story coverage is unavailable in the sampled fixture and remains unverified across the 33-SKU catalog. No new approval, alternate garment image or craft claim was invented.

## Final Recommendation

Retain this candidate and its complete review set as the next source-addressed engineering result. Final status remains **NEEDS_MORE_WORK**, with the unresolved work narrowed to same-profile mobile LCP/route budgets, the measured Skyy asset/startup/cadence limits, authoritative PDP content and the explicitly unperformed physical-device/accessibility checks. The isolated gzip experiment establishes a transport contribution but is not deployment evidence and does not clear all routes. The existing mandatory cinematic systems remain present and reviewable. No staging/production action or package promotion is authorized or represented by this report.
