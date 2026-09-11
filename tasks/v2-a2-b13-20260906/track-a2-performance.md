# Track A2 — Critical CSS Isolation Report

## Status

**REJECT — both B and C. The overall performance objective remains NEEDS_MORE_WORK.** The baseline remains accepted. All 18 measurements are complete; neither candidate delivers repeatable material improvement, and C has a demonstrated activation failure plus adverse early-frame evidence. No accepted theme, native Woo stylesheet, founder-approved media, model, content, staging or hosting configuration was changed.

## Baseline

Source digest `4a23629d6b17c1d0393000cfbbe982eedf47712837dd772a66c993ad7e46bbe7`, corrected local fixture 18416. B uses isolated router 18428; C uses 18429. All three share the same accepted files and synthetic fixture. The prior rejected all-inline experiment was not integrated.

The performance helper is unchanged: Lighthouse 13.4.1, simulated mobile 390×844/DPR1, RTT 150 ms, 1638.4 Kbps, CPU×4, exact local-origin proxy, cold contexts. Two samples per route/variant are bounded deliberately instead of 27 median-of-three runs. The second batch reverses variant order. A two-sample median is their midpoint, not evidence of statistical significance. Warm navigation and visual capture have separate methods and are not mixed with Lighthouse metrics.

## Coverage

`coverage.cjs` collected actual CDP matched rules for visible elements, inherited styles and pseudo-elements, together with whole-document CSS rule usage. It covers Home, Shop and Signature Collection at 390/1440, initial reduced motion, initial motion, menu, search with results, Quick View, no-JavaScript and Collection handoff. Every route includes actual native Quick View rendering where available. `coverage.json` retains state/range/source data; baseline screenshots retain the observed states.

The first pass used the wrong CDP selector-range field and produced zero matched-rule records. It was rejected before extraction, retained as `coverage-invalid-selector-range.json`, then corrected to the selector's range with an explicit nonzero assertion. No optimization conclusion uses the invalid matched-rule data.

`extract.cjs` uses PostCSS to retain selected rules and their original conditional wrappers, ordered by the document's actual stylesheet link sequence (`stylesheet-order.json`), not asynchronous network event order. Relative asset URLs become equivalent root-relative URLs. Font faces and their existing unicode ranges are preserved. The output is a conservative matched-rule candidate: matching does not prove every declaration wins the cascade, and unobserved states remain unknown.

Every rule is classified as CRITICAL ABOVE-FOLD, EARLY INTERACTION, BELOW-FOLD, ROUTE-OPTIONAL, GLOBAL NON-CRITICAL or UNUSED / CANDIDATE FOR LATER REVIEW. The per-route `*-provenance.json` gives minified file, authored source file/hash, selector, exact source offsets/line/column, observed states and B/C inclusion. `typography-provenance.json` records the existing font-face dependencies separately. Global non-critical means observed usage without matching a sampled visible node; it does not establish universal non-criticality. Unused/optional classifications are potential cleanup only; nothing was deleted.

## Critical CSS Size

The design goal was 10–20 KB raw with a hard 30,000-byte ceiling. Both router variants reject oversized blocks. All three conservative candidates miss the preferred range but remain below the hard ceiling; this is explicitly reported rather than hidden by gzip numbers.

| Route | B raw inline bytes | B gzip-equivalent bytes | C raw inline bytes | Additional early-interaction union if all sheets were delayed |
|---|---:|---:|---:|---:|
| Home |23,262|5,689|24,833|22,780|
| Shop |27,367|6,018|27,512|17,845|
| Collection |22,307|5,269|23,649|21,946|

Gzip-equivalent is a level 9 local calculation for block size, not a claim that the plain local HTML is compressed. Sizes include duplicated declarations; B duplicates its entire inline block because original stylesheets remain intact. C duplicates its block too, adding only the early-interaction rules from the one delayed route stylesheet. No duplication-saving source split was performed because proving selector/state completeness and native cascade safety precedes that maintenance cost.

Home's block contains header/shell, hero dimensions/typography/poster container/actions and the actually visible lightweight Skyy placeholder. It excludes full 3D runtime styling and below-fold scene rule sets. Shop includes visible portal row, shell, heading and controls; native general Woo remains external. Collection includes arrival geometry/title/action and visible handoff context, not all three scene systems. Entire existing token/font foundations are conservatively retained, one reason these blocks exceed the preferred size range.

## Candidates and loading strategy

**A** is unchanged accepted delivery. **B** adds the route-specific critical block before the unchanged external stylesheets. **C** retains shared shell, panel, controls and native styles as normal blocking links; only home-page, shop-page or collection-world respectively is temporarily screen-inactive (`media=print`). A small same-origin external deferred script restores its original media. The link keeps its original location, URL and version, preserving final cascade order. A native `<noscript>` stylesheet link at that position supplies the original stylesheet with JavaScript disabled. C adds no inline event handler, removes no stylesheet and changes no native dependency.

The individual HTML mechanisms are standard: stylesheet media conditions and the noscript disabled-scripting fallback are documented by [MDN link](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/link) and [MDN noscript](https://developer.mozilla.org/en-US/docs/Web/HTML/Reference/Elements/noscript). That establishes syntax, not a blanket certification of this loading pattern. In particular, JavaScript enabled with a failed activator is different from JavaScript disabled: noscript cannot repair an activator failure. C therefore remains a provisional experiment with a failure-path concern, never an approved strategy merely because its ordinary no-JS state renders correctly.

## Performance — final 18-run results

All 18 runs completed successfully. Values below are the median of two comparable samples (their midpoint); raw samples, exact profile, report hashes, fonts and observed LCP insight remain in `metrics.json`. Every median LCP fails 2500 ms.

| Variant | Route | LCP ms | FCP ms | CLS | TBT ms | Requests |
|---|---|---:|---:|---:|---:|---:|
|A|home|5387|3690|0.05099|3.0|47|
|A|shop|5762|3916|0.00038|0.0|42|
|A|collection|4326|3310|0.00184|0.0|49|
|B|home|5718|4168|0.05099|10.5|47|
|B|shop|6997|5167|0.00021|3.5|42|
|B|collection|4674|3997|0.00343|0.0|49|
|C|home|5678|3766|0.04677|0.0|48|
|C|shop|6327|4366|0.03457|0.0|43|
|C|collection|4222|3487|0.00200|0.0|50|

| Variant | Route | HTML wire bytes | External CSS wire bytes | Total wire bytes | Individual LCP samples ms |
|---|---|---:|---:|---:|---|
|A|home|97753|150598|1581262|5502, 5272|
|A|shop|146241|221635|1042168|5720, 5805|
|A|collection|133883|137881|1631420|4211, 4440|
|B|home|121047|150598|1604556|5781, 5655|
|B|shop|173640|221635|1069567|7074, 6921|
|B|collection|156223|137881|1653760|5417, 3931|
|C|home|122892|150598|1606745|5665, 5692|
|C|shop|174059|221635|1070330|6486, 6168|
|C|collection|157853|137881|1655734|3935, 4509|

B worsens median LCP on all three routes: Home+331ms, Shop+1235ms, Collection+348ms. C worsens Home+291ms and Shop+565ms. Its Collection median improves only103ms while its two individual samples move in opposite directions relative to paired A; FCP worsens by177ms. This is not repeatable material improvement. No sample is discarded for looking slow. Byte weight is the Lighthouse capture window, distinct from load+10-second route budgets. Observed LCP subparts are retained separately and are not addends of simulated LCP. Local server/CPU/scheduling variability is not decomposed into invented causal contributions.

Typography delivery is unchanged by file/count: every Home/Shop sample requests four font files totaling 163,496 wire bytes, and Collection five totaling 165,574 bytes. Existing Hanken/Inter unicode fallback behavior is retained. All individual CLS values remain below 0.1, but C Shop median CLS rises from 0.0004 to 0.0346. These observations do not isolate font-caused shifts; font-ready/paint ordering and other layout work are not interchangeable. No field INP, GPU or real-device claim is made.

## Warm Navigation

`warm.cjs` navigated Home → Shop → Collection twice in the same browser context for each variant. It used the existing local-origin proxy and did not install Playwright route handlers, which would disable HTTP caching. The original stylesheet URLs, query versions, ordering and bytes remain external and identical. `warm-results.json` contains complete Resource Timing and navigation entries.

The fixture showed **zero zero-transfer CSS entries in both rounds for every variant**. Thus this run does not demonstrate a warm stylesheet-cache hit or cache benefit. It also does not prove that production caching is broken. Local static delivery lacks the independently verified production cache behavior required for that claim. Every B/C navigation repeats its inline block in the document while retaining the same external CSS transfer in this fixture.

## Visual and Early Interaction

The final sweep produced 144 PNGs: 48 per variant,390/1440, DOMContentLoaded, settled initial, early/late menu and search, Shop early/ready Quick View, Collection handoff, and no-JavaScript states. Six WebM recordings preserve startup and route/interaction sequences. All 36 route cases report zero page errors and no horizontal overflow. `visual-results.json` contains visible geometry/font/color/background records and link/state evidence; `visual-comparison.json` contains 96 baseline/candidate comparisons.

Eighty-two pairs are byte-identical. All compared settled initial, late-search, Quick View and handoff pairs match exactly. Fourteen earlier or no-JS captures differ; most retain identical recorded computed geometry/font properties and are not silently relabeled pixel-identical.

The material unresolved early-state comparison is Home 390 menu: baseline shows a rose Menu label/X and hides header Ask Skyy, while B/C show a white Menu label/plus and header Ask Skyy. B also differs in its first chapter tile focus border. Main directory geometry/readability remains consistent. Root independently viewed these three captures and explicitly did **not** accept this early-state gate as EQUIVALENT. These states were captured immediately after activation; the harness did not record enough nav-class/focus/animation timing to establish the exact cause. We retain the mismatch and withhold integration rather than asserting it is harmless timing.

## No-JS

All 18 no-JavaScript route/variant/viewport cases complete without overflow/errors; headings, navigation/product links and composed initial content remain present. B uses normal external CSS. C's noscript restores the single route stylesheet. The exact comparison includes some nonidentical image raster captures with matching recorded computed geometry; visual certification remains bounded, not a fabricated all-pixels-equal claim. No-JavaScript initial rendering does not exercise JavaScript-only dialogs or certify the failed-activator case.

## WooCommerce

Native general Woo remains untouched on Shop; existing archive layout decisions from readiness are preserved. Quick View early and ready screenshots use actual native forms and match baseline. No cart, checkout, order/payment, PDP variation/gallery or catalog write was performed. This is presentation regression evidence for the tested states, not a fresh complete transactional certification.

## Maintainability

The generated blocks have source/rule provenance and reproducible extraction rather than an opaque hand-edited blob. However, they bind to exact generated CSS positions and the sampled routes/viewport/states. Every source rebuild needs fresh coverage and source-hash validation; actual winning-declaration minimization and complete lifecycle coverage are not established. Retaining full external styles avoids destructive cleanup but duplicates bytes. These costs are part of the decision, not future work disguised as a finished implementation.

## Recommendation

Keep accepted delivery unchanged. Do not integrate B/C or delete unused-classified rules. Require repeatable performance improvement and independent closure of early state/flash/no-JS/activation failure gates before any future candidate is considered. The immediate deliverable is a bounded decision with raw evidence, not an automatic source promotion. Track B13 owns the separate platform evidence.


## Activation failure and startup-frame closeout

The bounded local failure injection made `/a2-activate.js` return 404 while scripting remained enabled. On Home, Shop and Collection, the intended route stylesheet was fetched but remained `media=print` instead of `all`, with zero active noscript fallback links. `activation-failure.json` and three screenshots preserve this result. C therefore introduces a styling dependency on successful script delivery and is rejected; it does not meet the intended resilient styling requirement merely because JavaScript-disabled tests pass. No repair or new candidate was attempted after this result.

Six single-thread FFmpeg contact strips sample the first 3 seconds of the recorded sequences at nominal 250 ms intervals, 12 frames per strip; `clips/sampling.json` records timestamps and limits. The final B and C 390px strips both expose an adverse initial Home presentation: each first visible sampled frame has smaller copy, vertically stacked CTAs and no visible Skyy, before converging to the final larger typography, side-by-side CTAs and mascot presentation. The baseline first rendered sampled frame already shows the latter composition. This is sampled transient WORSE evidence, not explained away by identical settled screenshots. The precise missing-rule/font/script cause is not established. Screenshots themselves wait for fonts/actionability and do not establish pre-style timing; the earlier early-menu mismatch also remains unresolved. Sampling cannot certify absence of shorter-than250ms flashes or later transitions; full videos remain available for review.

Root and the independent reviewer reopened the final 390px startup strips and confirmed adverse initial sampled presentation in BOTH B and C: smaller copy, stacked CTAs and absent Skyy before convergence. The earlier characterization of B as only a mascot scale/position difference is superseded by this final-strip review. No precise CSS-versus-lifecycle timing cause is asserted. Both candidates remain uncleared for early visual equivalence. This experiment does not self-certify a candidate for promotion. The final source-invariance check rehashed all 86 files in the accepted readiness source manifest and found zero differences. Coverage comprises 12 sessions / 38 states; the later comparison sweep comprises 36 cases / 144 stills. Browser profiling/capture and clip generation have stopped, and CPU ownership has been released to the root task.
