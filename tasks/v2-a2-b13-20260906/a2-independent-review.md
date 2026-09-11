# A2 independent review

**FINAL: REJECT both B and C — no integration.** The completed18-run performance, activation-failure and startup-strip evidence below supersedes the initial NEEDS_MORE_WORK assessment. This disposition applies to the isolated experiment, not to the unchanged accepted theme. Read the full user brief and inspected coverage/extraction, routers, provenance, size, visual and warm-navigation artifacts. No browser, Lighthouse, heavy diff, build, source modification or staging request was run for this review. Only this report was written.

## Findings and acceptance gaps

1. **HIGH integration gate — early Home menu states are not equivalent as captured.** Independently viewed `a-home-390-menu-early.png`, `b-home-390-menu-early.png`, and `c-home-390-menu-early.png` at original detail. Baseline shows an X with rose-colored Menu and no header Ask Skyy; B/C show a plus, white Menu and visible header Ask Skyy while the directory is displayed. The first collection tile also has a different outline. These are visible control/state differences, not merely differing PNG compression. The captures do not establish whether the cause is interaction timing, focus/state mismatch or a CSS race. Do not certify equivalence or attribute a root cause without synchronized expanded state, focus, timestamps and stylesheet readiness. The currently submitted early-state gate is unmet.

2. **HIGH integration gate — C depends on a successful activation script while JavaScript is enabled.** `router-c.php` changes the route stylesheet to `media="print"`; only `/a2-activate.js` restores its original media. `<noscript>` correctly covers JavaScript disabled, but cannot recover an enabled-JavaScript page where that script is blocked, fails to fetch or fails before applying the media change. Source inspection establishes that the route sheet then remains unavailable to screen rendering indefinitely. No activation-failure test is present in the inspected evidence; this is a source-derived failure case, not a fabricated browser reproduction. The user's resilient/non-JS-dependent styling requirement is not established for this branch. Preserve the normal stylesheet or demonstrate a robust activation/failure strategy before recommending integration.

3. **MEDIUM evidence gap — early visual tests do not exercise missing deferred CSS.** All six C `dom` snapshots record the target link as `media: all, sheet: true` before subsequent early-menu/search/QV captures. The activation script is deferred and runs before DOMContentLoaded. `page.screenshot()` also ordinarily waits for fonts, while locator clicks wait for actionability. These are useful post-DCL interaction checks, but not proof that controls remain styled before the delayed sheet arrives. The coverage tool additionally waits700ms before its initial matched-rule capture. A controlled delayed-route-CSS case and event/readiness timestamps are needed for the specific early-styling claim. No artificial delay should be mixed into performance results.

4. **MEDIUM evidence gap — warm navigation did not demonstrate cache reuse.** Every stylesheet resource in all18 warm-navigation records has positive transferSize. Round2 external CSS totals are identical to round1 for every route/mode. The local static routers do not emit explicit cache lifetimes or validators. Retained stable/versioned URLs preserve the possibility of production caching, but that is not observed cache reuse. Report this fixture limitation and quantify duplicated inline bytes; do not claim warm-cache improvement from these samples. The user allows reasoned warm-navigation analysis, so honest limitation reporting can complete the experiment without inventing cache success.

5. **MEDIUM maintainability limitation — coverage provenance is not a complete production ownership contract.** The generator records hashes, selectors and offsets from minified source, which makes this snapshot traceable. It does not enforce a manual component/rule allowlist or assert captured source hashes against current source on regeneration. The viewport heuristic checks vertical bounds, display and visibility but not horizontal exclusion or all occlusion/opacity cases. A matched selector is not itself proof of criticality, and only the tested content/states are covered. Dynamic native extensions, validation/error states, alternate collection copy and changed content require further dependency review. `UNUSED / CANDIDATE FOR LATER REVIEW` must remain a later-review label; it does not authorize source deletion.

## What is supported

- Coverage JSON contains12 sessions: three routes × two widths × JS/no-JS. JS sessions include reduced-motion initial, normal-motion initial, menu, search and QV; Collection includes handoff. All recorded session error arrays are empty.
- Independently compared every captured theme stylesheet text to its current local file: **15 distinct CSS files match exactly**, with no mismatch. The declared accepted baseline digest is `4a23629d6b17c1d0393000cfbbe982eedf47712837dd772a66c993ad7e46bbe7`; the comparison here verifies captured CSS inputs, not a freshly recomputed whole-theme digest. Routers read the accepted theme and inject isolated output; no source integration is present.
- Node syntax check for the extractor and PHP syntax checks for both routers pass. This is JavaScript experiment tooling, with no dedicated TypeScript or ESLint project command at this artifact location. Syntax checks do not replace runtime/cascade verification.
- B retains all normal external stylesheets. C changes only its selected route handle, preserves the URL and eventual media value, and includes the original link inside noscript. Native Woo general CSS is not dequeued. No native source styles are deleted. Once all sheets apply, their external DOM order remains unchanged; transient order before activation is not proven equivalent merely by that final order.
- The extractor preserves selected rules inside their original at-rule wrappers and follows the recorded stylesheet order within the inline subset. Relative resource URLs are rebased. Existing font faces and the Hanken/Inter subset strategy are retained; no invented fallback metrics are added. Keyframes are excluded from generated subsets, while complete original stylesheets remain external.

## Budget and duplication

Independently checked emitted file sizes against the recorded size table:

| Route | B inline raw | C inline raw | B standalone gzip estimate | Early-interaction classification raw |
|---|---:|---:|---:|---:|
| Home |23262B|24833B|5689B|22780B|
| Shop |27367B|27512B|6018B|17845B|
| Collection |22307B|23649B|5269B|21946B|

All are below the30000-byte router ceiling; all exceed the suggested10–20KB raw starting target. That is not a silent100KB inline experiment, but the excess requires an explicit explanation. The standalone gzip estimate is not the measured incremental compressed HTML cost. B duplicates the entire selected subset already present in retained external files; C does too, adding selected early rules from only the deferred route sheet. Other early-interaction rules stay in normally loaded external sheets. External bytes are not reduced.

The warm navigation records show external CSS transfer totals of151658B Home,222591B Shop and138943B Collection for **each** A/B/C visit in this fixture, in both rounds. Each B/C navigation additionally carries its inline block. These numbers include the recorded resource transfer accounting; they are not compressed production-CDN byte predictions.

## Visual and no-JavaScript boundaries

There are36 visual records and144 stills, including18 no-JavaScript route/width/mode captures. The comparison JSON has96 B/C-versus-A pairs:14 are not byte-equal; two (Home390 early menu B/C) are also not equal under the recorded computed-style fields. Every settled comparison is byte-equal. No-JavaScript comparisons match the recorded geometry/style fields, although several desktop stills differ at the pixel level. This supports bounded recorded no-JS layout behavior, not independent eyes-on approval of every still. The computed fields omit some state and style properties and cannot override a visible mismatch.

Startup videos exist but were not decoded or independently reviewed here while Lighthouse owns CPU. No FOUC/flash PASS is issued from screenshots alone. No actual native purchase/variation submission was performed by this reviewer; the inspected QV captures demonstrate recorded opening/ready states, not a complete transaction regression suite.

## Performance and recommendation

Lighthouse collection was still running during this source/evidence review. The runner counterbalances A/B/C order across two rounds and targets the existing shared profile. Its two-sample midpoint aggregation is correctly labelled as such, not median-of-three or statistical significance. No final performance win is certified here. Typography resource records and overall CLS do not by themselves isolate a font-specific CLS/LCP effect.

Initial disposition before final profiling was NEEDS_MORE_WORK. The final closeout below replaces that pending decision. Keep A accepted and B/C isolated; no automatic integration or unused-CSS deletion.

## Final closeout — REJECT

Independently read `metrics.json`, checked all18 underlying report SHA-256 values (zero mismatches), and confirmed one common screen/throttling/method profile across their recorded settings. Two samples per route/variant remain a bounded experiment, not a population confidence claim. Their midpoint results are:

| Route | A LCP / FCP | B LCP / FCP | C LCP / FCP |
|---|---:|---:|---:|
| Home |5387 /3690ms|5718 /4168ms|5678 /3766ms|
| Shop |5762 /3916ms|6997 /5167ms|6327 /4366ms|
| Collection |4326 /3310ms|4674 /3997ms|4222 /3487ms|

B worsens LCP/FCP midpoints on every route. C worsens Home and Shop; its small Collection LCP midpoint reduction is not repeated consistently across paired runs, and Collection FCP worsens. C Collection individual LCP samples3935/4509ms compare with A4211/4440ms: one faster, one slower. There is no repeatable material gain to justify the inline duplication or new lifecycle complexity. No median LCP meets2500ms. This is enough to reject these candidates without inventing a causal CPU/network decomposition.

The added actual-browser failure evidence upgrades finding2 from source-derived risk to a reproduced failure. `activation-failure.json` records JavaScript enabled, loaded route sheet still `media=print`, and zero active noscript fallback links for Home, Shop and Collection when the activator returns404. This proves the specific failed-activation behavior; no claim of unrelated browser failures is made.

Independently viewed the exact current `clips/a-390-startup.png`, `clips/b-390-startup.png` and `clips/c-390-startup.png` at original detail. **Both B and C have a sampled first-visible Home presentation that is WORSE than A's first-visible presentation**: the candidate frames show a single-line subtitle and smaller body treatment, stacked CTA buttons with a visibly different text treatment, and no visible Skyy; A's first-visible frame already has the wrapped subtitle, side-by-side condensed-label CTAs and Skyy. B and C later converge toward the accepted composition. This final observation supersedes any earlier narrower B description based on a different glance/capture. The precise CSS/font/script timing cause is unproven. The retained early-menu mismatch is an additional unmet state gate.

The contact strips sample nominal250ms intervals in the first3seconds, with the documented possible source-frame offset. They prove the visible sampled transient differences; they cannot certify the absence of shorter flashes or later-route issues. Reviewer did not decode video or launch browsers during B13's CPU window. Desktop strip review and complete temporal inspection are not claimed.

Read `source-invariance.json`: the owner's final check reports all86 accepted-manifest files unchanged at digest `4a23629d6b17c1d0393000cfbbe982eedf47712837dd772a66c993ad7e46bbe7`. Independent review separately verified captured CSS inputs earlier; it does not relabel the owner's whole-manifest check as independently rerun.

**Final recommendation: retain accepted A; reject B/C as submitted.** The experiment has reached a defensible negative decision. The broader performance objective remains open, but no new candidate, source split, deletion, media change, deployment or staging/cache modification follows automatically from this result.
