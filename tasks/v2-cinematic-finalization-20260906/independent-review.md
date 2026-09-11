# Independent finalization source and evidence review

Review baseline: `.artifacts/v2-cinematic-finalization-20260906/baseline-theme/skyyrose-flagship-2`, the immutable accepted intermediate candidate. This review compares that snapshot with current theme source, rather than treating the Git HEAD diff as the finalization scope.

## Verdict

No remaining actionable blocking source defect identified in the reviewed finalization changes after the Skyy restart correction below. This is a bounded engineering review, not a whole-site certificate or founder visual approval. Browser, recording, actual device, and exclusive performance runs remain their owners' evidence.

## Latest source-freeze delta review

This section supersedes earlier references below to the native-defer experiment being part of the candidate. Root reverted that experiment after finding no reliable causal LCP improvement. An independent byte comparison confirms current `inc/performance.php` is exactly equal to the accepted immutable baseline. Native-defer graph tests and external-adapter review are retained experimental evidence, not evidence of a shipping performance optimization. Do not attribute final performance gains to the reverted experiment.

The Home mascot loader now mounts the lightweight guide promptly but requires character pointer/tap or Ask-action focus intent before preparing the heavy renderer. Existing chat activation remains available. Reduced-motion and Save-Data checks continue to prevent the heavy script path. Source inspection found no new blocker in the readiness/intent conjunction. Final reports must describe this as character-intent activation, rather than claiming the previous load/idle auto-entrance still occurs.

The narrow conversation log is now focusable with `tabindex="0"`, and the initial greeting starts at scroll position zero. These changes address keyboard access to the bounded transcript without hiding the composer. Source correctness does not substitute for the dedicated narrow-browser keyboard evidence.

The revised hero helper requires visible video opacity/readiness and records actual presented frames at 1.00–1.06 seconds before a diagnostic pause. This is stronger than simply seeking and assuming a painted frame. One helper-level issue was reported to root: the frame-selection Promise needs a finite deadline and cancellation path, or a stalled video/repeatedly missed capture window can hang the serial capture run. This is a capture-harness concern, not a storefront defect.

Latest independent light rerun: Skyy concierge plus visual-recovery tests **39/39 PASS**, diff whitespace **PASS**. No browser/build/compile/package work was started by this reviewer.

## Findings and resolution

1. **Skyy dynamic-preference continuity — corrected.** `showFallback()` resets `firstReveal`, but the first stable pose initially reset only entry progress, leaving the previous walk's horizontal `--skyy-entry-shift` in place. Restoring motion after toggling reduced motion/Save-Data mid-walk could offset the canvas from the poster during the next handoff. The current `skyy-3d.js` now explicitly resets that shift to `0px` before the stable frame renders. The Skyy owner is adding/exercising the matching dynamic-preference regression; this reviewer did not run that browser scenario.
2. **Native graph test fixture fallback — aligned.** The new real-WordPress graph test originally required `V2_WP_FIXTURE` explicitly. The canonical suite already required an installed fixture through its existing PDP-gallery test, so this did not introduce the installed-WordPress requirement itself. The new test now shares the existing default fixture fallback. The broader dependency claim was withdrawn after checking the existing suite.

## Source reviewed

- WordPress Core-native defer policy and opt-out/async preservation; external native variation-template adapter ordered through the Woo handle dependency graph.
- Quick View source-card states, native POST busy lock, later-extension submission cancellation recovery, native variation accessibility state, and retained stale-request guards.
- Premium commerce image states, Woo-only request/card correlation, unrelated AJAX error isolation, native links, preview focus/tap behavior, decode timeout, and navigation lifecycle cleanup.
- Strict collection poster preparation and active-video arbitration, no-JS fallback markup, visibility/page lifecycle, reduced-motion/Save-Data behavior, hero stable-video-frame reveal, and restrained staggered commerce handoff.
- Skyy stable-pose presentation, existing-model poster reference, motion lifecycle, software state reporting, bounded profile instrumentation and its stated measurement boundaries.
- Product card/PDP/navigation PHP additions and their escaping/native authority boundaries.
- Trace-navigation helper, Lighthouse route/session handling, and actual-WordPress graph regression design.

The new native script strategy uses Core dependency resolution rather than overriding Core's resolved ordering. The external template adapter is a dependency of Woo's native variation handle. The existing Quick View inline-after and cross-origin eager fallbacks remain relevant compatibility protection.

## Independently executed lightweight checks

- Node `v22.23.2`.
- Quick View dependency graph regressions: **19/19 PASS**.
- PHP performance contract: **PASS**.
- Skyy concierge plus visual-recovery DOM/runtime tests: **35/35 PASS**.
- Diff whitespace check: **PASS**.

No browsers, builds, Lighthouse sessions, or remote writes were launched by this reviewer. ESLint was unavailable in the existing checkout; this review did not substitute a successful syntax/DOM test for ESLint certification. The finalization scope contains JavaScript/PHP, not new TypeScript.

## Independent eyes-on matched capture assessment

Files below are under `.artifacts/v2-cinematic-finalization-20260906/`.

| Surface inspected | Before/current files | Finding |
|---|---|---|
| Mobile Quick View | `commerce/{before,current}-quick-view-390.png` | **BETTER** in the inspected state: more compact identity/variation hierarchy, complete contained approved image, visible purchase and continuation controls. The before capture leaves more required controls below its shown portion. |
| Signature desktop card | `commerce/{before,current}-card-signature-1440.png` | **EQUIVALENT**: same approved portrait, frame, crop, identity, price, availability and actions; no visible regression. |
| Mobile Skyy idle | `skyy/{before,current}-390-idle.png` | **EQUIVALENT**: character identity, anchor, scale, readable hero and controls retained in the inspected stills. |
| Desktop Skyy chat | `skyy/{before,current}-1440-chat.png` | **BETTER** frontal character presentation and an explicit Minimize control, with transcript/input readable. Background scroll positions differ, so this is not a pixel-parity comparison. |

These sampled stills cannot establish timing, motion smoothness, all nine-scene visual acceptance, every SKU state, or every viewport. No independent recording playback occurred in this source review.

## Evidence and certification boundaries

- Raw route LCP observations and Lighthouse simulated mobile results use different measurement conditions; retain both and explain their profiles. Trace-navigation records an explicit observation window and unthrottled/devtools profile. It is not field CWV or automatically equivalent to Lighthouse simulation.
- Lighthouse cart/checkout helpers seed a disposable local cart and assert the final route. The run marker's PASS means the measurement completed, not that LCP/CLS or every budget passed. Final reporting must read the metric values separately.
- Skyy `frames` measures CPU time inside `renderer.render()`, not GPU execution, mixer work, or the full animation frame. `intervals` is render cadence. Geometry-array bytes and the RGBA8/mip estimate exclude driver, decoder, heap and transient allocations. The Skyy report correctly states these boundaries.
- Matched poster/canvas layer geometry and silhouette/colour measurements support static-to-first-frame continuity, not physical-device GPU performance. A desktop Metal or SwiftShader run must retain its actual renderer qualification.
- The commerce report explicitly separates synthetic unavailable/sale/native-error fixtures from sampled real local product states, and does not certify missing rich material/story/media content. Preserve that distinction in the combined report.
- BR-003 source-authority/media limitation, real-device/manual screen-reader gaps, final route performance budgets, and any missing required recordings remain open wherever their owners have not supplied direct evidence. They cannot be cleared by this source review.

Root follow-up: hero helper now has a20second deadline with video-frame callback cancellation; successful capture clears the deadline. This closes the reviewer’s bounded-run helper finding. The live fixture CSP permits inline scripts via unsafe-inline, and does not currently add a nonce; the scene watchdog uses WordPress’s inline-script API and remains compatible with nonce hooks. No nonce presence is claimed.
