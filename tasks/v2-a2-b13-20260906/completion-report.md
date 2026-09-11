# V2 A2 / B13 — bounded completion report

**Readiness remains NEEDS_MORE_WORK. A2: REJECT both critical-CSS candidates. B13: completed read-only diagnosis of the observed staging deployment.** No experiment was integrated, no theme or hosting configuration was changed, no cache was purged, and no order or payment was submitted. The cinematic system was not redesigned. Town Line's separate pre-order phase was not begun.

## Identity and preservation

Accepted local source digest: `4a23629d6b17c1d0393000cfbbe982eedf47712837dd772a66c993ad7e46bbe7`. End verification rehashed all 86 manifest source files and all 397 protected media files: zero changes. Branch remains `codex/v2-cinematic-ooda-20260906`, HEAD `aabd2bffdc1e322862acd05a5640c14cf00f3acf`.

Authorized target: `https://staging-7e48-skyyrose.wpcomstaging.com`. Observed identity: **STAGING-OBSERVED-f2893d998356**. This is a comparison-inventory identity, not a remote Git commit or complete-tree certificate. Of 451 requested paths, 322 exist: 295 match accepted local bytes, 27 differ and 129 are missing. The end SSH check reproduced this same inventory with zero changes. Managed HTML, database state and optimized bundles are not covered by that file-map equality.

The observed deployment differs from the accepted local candidate. These browser outcomes describe what staging actually delivered. Certification of the accepted local candidate on staging is **ENVIRONMENT-BOUND** until its deployment identity can be established under a separately authorized release step.

## A2 — REJECT B and C

Eighteen Lighthouse runs used two cold samples per route/variant, with reverse order in the second batch. Values below are the midpoint of two samples, not a statistically established median-of-three result.

| Mobile LCP | A accepted baseline | B small inline + normal CSS | C small inline + delayed route CSS |
|---|---:|---:|---:|
| Home | 5387ms | 5718ms | 5678ms |
| Shop | 5762ms | 6997ms | 6327ms |
| Collection | 4326ms | 4674ms | 4222ms |

Both candidates worsen Home and Shop. C's small Collection midpoint difference is inconsistent between samples. Critical blocks are approximately 22–27.5KB raw, below the ceiling but above the preferred 10–20KB target, and duplicate retained external CSS. Warm navigation in this fixture recorded no stylesheet cache hits; no production cache conclusion follows from that limitation.

Both candidates show worse first-visible sampled Home presentation before converging. Settled screenshots do not clear that failure. C also leaves its route stylesheet inactive when the activator fails with JavaScript enabled; the noscript fallback cannot recover that case. Coverage, rule/source provenance, font information, 144 comparison stills, six recordings, six timestamped startup strips and independent review are retained. No rules were deleted or candidate integrated.

## B13 outcomes — observed staging deployment

| Category | Result | Evidence and limit |
|---|---|---|
| HOME HERO DELIVERY | **FAIL** | Home delivers a static image, not the accepted animated Home hero. A fast static-image or heading LCP cannot clear animated delivery. The Home Town Line film was not substituted or played. |
| COLLECTION SCENE DELIVERY | **PASS** | Bounded pass across all three collections and both profiles: no initial three-scene film download, all three scenes produce viewport compositor frames when approached, and pause/resume/reverse/re-entry are exercised. This is not a physical-device frame-pacing certificate or a pass for arrival-hero LCP. |
| PRODUCT CARD DELIVERY | **FAIL** | Visible BR-003 card uses a 1024×1536 image without srcset/sizes at roughly 168×224 CSS pixels, DPR2. This oversizing is distinct from the PDP's image-CDN transformations. |
| QUICK VIEW DELIVERY | **FAIL** | Opens and closes, with no eager PDP requests, but remains the older card-preview dialog: no intent-triggered PDP request or native product form. Accepted native Quick View behavior is absent. Its stale async-PDP response handling is therefore not exercised. |
| ASK SKYY DEFERRED DELIVERY | **FAIL** | Untouched mobile defers 3D, but untouched desktop automatically starts walk-in at12.938s, dependencies at12.955s and GLB at13.209s. This is staged theme scheduling. |
| MOBILE CINEMATIC DELIVERY | **FAIL** | Moderate-profile animated collection LCP is3.708–5.552s, PDP LCP6.264s, Home CLS0.1362, and Skyy intent→first draw14.801s. Missing accepted Home/Quick View behavior also prevents cinematic completion. |

Chromium145.0.7632.6 with ANGLE Metal was used on this host, mobile emulation390×844/DPR2/touch. Normal network was unthrottled with1×CPU. Moderate used100ms latency,4Mbps down/1Mbps up,2×CPU slowdown. Twelve route/profile observations cover Home, Signature, Black Rose, Love Hurts, Shop and SG-005 PDP. A separate1440×900 untouched desktop check and bounded corrected-observer/reuse follow-ups are clearly separated. These are single route/profile observations, not field data or physical-phone certification.

## Measured bottlenecks and ownership

| Observation | Ownership | Meaning for next work |
|---|---|---|
| Moderate collection arrival LCP: Signature4388ms, Black Rose5552ms, Love Hurts3708ms. First viewport film callbacks closely align. | THEME / MEDIA ASSET / NETWORK | Real staging reproduces late animated LCP under the documented profile. Inspect actual poster/film discovery, transfer and presentation scheduling; do not remove films. Exact contribution of each stage is in the resource and event timelines. |
| Skyy normal intent→first WebGL draw1571ms; moderate14801ms. Moderate GLB transfer takes about12.22s with roughly6.07MB CDP encoded bytes despite an exposed edge HIT. | MEDIA ASSET / NETWORK; THIRD PARTY imports; THEME scheduling | Model size dominates this constrained transfer. Imports, decode and GPU upload remain separate costs. Existing Blender handoff is the asset track; an edge HIT is not proof of a fast client connection. |
| Desktop requests 3D automatically without intent. Reload after minimize/recall does not reach model-ready or issue a second GLB request. | THEME; reload cause remains a source-supported lifecycle hypothesis | Review actual staged loader timing and persisted dismissal/re-entry. First-navigation recall successfully reuses the ready model in memory. Reload browser-cache reuse remains ENVIRONMENT-BOUND; absent requests do not prove cache reuse. |
| Home moderate maximum-window CLS0.1362. Recorded title/copy shifts follow font completions. | THEME / MEDIA ASSET, font-causation inference | Investigate stable typography geometry and font discovery. Raw shift sum0.252 is diagnostic only and is not the reported CLS. |
| PDP image load4673ms, LCP6264ms: about1591ms remaining render delay. | UNKNOWN | The delay is observed, but a theme/platform/third-party cause is not isolated. The browser does display a real gallery; earlier HTML-only absence was response-specific. |
| Oversized Shop card with no responsive candidates. | THEME / MEDIA ASSET | Establish the intended deployed candidate before changing its card output; retain approved imagery and geometry. |
| B01's two exact optimized URLs return different decoded br versus identity/gzip code. | UNKNOWN within managed WORDPRESS.COM / CDN delivery path | Platform investigation input, not a theme patch. Browser routes consumed different optimized URLs whose bytes match the newer payloads; B01 drift is not established as the browser timing cause. |
| Cart TTFB variability in prerequisites, Checkout only an empty-session redirect. | UNKNOWN; WORDPRESS/PHP versus WORDPRESS.COM/CDN not isolated | Do not assign origin timing or personalization correctness from these anonymous samples. No cart additions or orders were performed. |

B07 verifies exact206 ranges and416 behavior for representative films; B13 records actual playback and re-entry separately. No arbitrary seek was introduced where the UI does not use seeking. Browser cache flags and exposed CDN headers are separate evidence layers. Exact first physical painted poster/frame, first stable GPU frame, physical-device smoothness, full cache-key/immutability behavior, and authenticated/cart personalization isolation remain environment-bound where not observable or not supplied.

## Evidence integrity and next decisions

Root independently inspected the normal Home, Shop, all three collection arrivals, representative scene captures, Quick View, PDP, Skyy and untouched desktop captures. The different staging feature set prevents accepted-local visual-equivalence certification. No media approval scope was promoted. Each collection also made a passive WooCommerce fragment-refresh POST during normal page loading; read-only UI testing is not misrepresented as GET-only network traffic. No purchase mutation or external concierge request was submitted.

Shop/PDP layout-shift observer failures were retained and corrected in bounded replacement runs. Normal Home's original post-intent observer error does not affect its untouched measurement; corrected follow-up evidence is separately identified. Measurement code failures are not blamed on the theme. The final same-context reload timeout is retained, not converted into cache success.

Next decisions are deliberately separate: retain accepted CSS delivery; resolve intended staging candidate identity before any future deployment; investigate exact managed-bundle representation drift through platform ownership; continue measured theme scheduling/font/card work only against the correct candidate; use the preserved Blender asset handoff for model optimization; keep founder content/media review separate. No next-step change is authorized or applied by this report.

Detailed evidence: `track-a2-performance.md`, `a2-independent-review.md`, `track-b-prerequisites.md`, `b13-report.md/json`, `root-visual-review.md`, and `.artifacts/v2-a2-b13-20260906/` resource, event, hash and screenshot records. Raw response headers/anonymous HTML discovery bodies are outside the local artifact webroot in private evidence storage.
