# Ask Skyy — website-state completion

Status: **SOURCE AND BROWSER BEHAVIOR VERIFIED; independent visual/source acceptance and root's full-theme performance/package gates remain separate.** Base: `415dc4368b44efc06d0d28928362901f9624a20f`. Authority: founder completion directive `7468daf4-9f25-4287-bf79-7bc117adc639`, managed by root's `scope.json`.

## Completed experience

The canonical portrait previously disappeared immediately when the renderer became ready. Home also translated the entire shared wrapper into the gait's starting position, so the static character jumped before becoming3D. Different portrait aspect fitting compounded the perceived scale discontinuity. Static, loading, reduced-motion, Save-Data and failed states lacked deliberate feedback.

**NEW COMPLETION:** the existing portrait and canvas now form one opacity handoff. The guide responds to the preserved renderer's `skyy:3d-visible` event, keeps the same portrait in the layer stack, establishes a paint boundary and crossfades opacity over240ms. There is no duplicate image, canvas, model, rendering loop or asset request. Repeated Pause/Resume visibility events do not restart the fade. Hidden/failure states cancel pending callbacks.

Home gait translation now affects the canvas alone. The portrait stays stationary, bottom-aligned and scaled within the existing reserved character area. The actual390 diagnostic revealed one additional shift: displaying Pause widened an intrinsic control column. Reserving that column at8rem eliminated the readiness reflow. All nine final cases measured **0px horizontal wrapper displacement** and stable wrapper width/height through the handoff.

Short translated status messages distinguish loading, ordinary presence, motion off, data-saving and failed-motion states. They use existing tokens and the reserved Home control area. Failure does not block conversation or create a misleading repeated loading state. The native invitation becomes a visible Contact link if guide loading fails; both its title and label come from escaped WordPress translation attributes, and owned focus is preserved.

The native dialog retains44px controls, safe-area height clearance and a compact two-column layout in short landscape viewports. Small screens retain native vertical scrolling to reach the form. Escape and Close focus behavior remain verified. No product facts, route-specific claims or external chat functionality were introduced; assistance still uses the localized Woo-derived catalog and validated native destinations.

## Ownership and preserved systems

Changed sources: `assets/js/mascot.js`, `assets/js/mascot-loader.js`, `assets/css/mascot.css`, `template-parts/skyy-mascot.php`, and focused Skyy tests/harness. Root owns enqueues, PHP pins, compiled artifacts and full integration.

`assets/js/skyy-3d.js` and `assets/models/skyy-mascot.glb` have zero diff against HEAD. The current18-joint model, runtime-derived gait, camera, meshes, materials, textures and source clips remain unchanged. Deeper Blender/rig/texture/LOD production remains explicitly deferred. No deployment, paid call, catalog/order/payment write or new runtime dependency occurred.

The new state layer schedules at most two animation frames per handoff and uses compositor opacity. It adds no continuous loop or network dependency. The preserved renderer still owns playback, timing, visibility, pause and GPU disposal. No global performance improvement is inferred solely from this bounded state change.

## Verification results

- **29 focused tests passed,0 failed:** `node --test tools/v2-runtime/test-skyy-concierge.cjs tools/v2-runtime/test-guide-focus.cjs`. Coverage includes handoff paint boundary, repeated visibility, cancellation, failure/lightweight truth, usable conversation after renderer failure, translated native fallback labels/focus, and the existing actual Three.AnimationMixer/source-preservation tests.
- PHP lint, JavaScript syntax, PostCSS parsing and scoped diff checks passed. Root rebuilt the served assets before final captures.
- **Nine actual browser cases passed:**320,360,375,390,414,768,1440;844×390 landscape; and a separate390 case with a deliberately injected1600ms local model delay. Each observed14 intermediate opacity frames, retained the same portrait during the fade, measured0px wrapper X displacement and stable dimensions, and recorded70–84 samples of the preserved changing Walk pose. Dialog overflow,44px Close targets, conversation input and native focus checks passed.
- **Representative390 Home controls passed:** same-canvas reparenting, Escape and Close focus return, offscreen render stop/resume, dismissal/header recovery, one GLB fetch, fresh reduced-motion/Save-Data with zero heavy requests, and visible native Contact navigation without JavaScript.
- **Injected local404 cases passed:** model failure retains the approved portrait and truthful status while chat remains usable; guide failure exposes translated Contact semantics and navigates to the native reachable destination.
- **Eight settled dialog/form capture cases passed.** Initial dialog-onset screenshots are temporal animation states; use `*-dialog-settled.png` and `*-dialog-input.png` for steady-state review.

All actual3D cases used **ANGLE Metal on Apple M5**, with other task browsers closed during this window. Frame timestamps/counters are recorded observations, not a general mobile-device benchmark. The slow case explicitly includes network delay and must not be used as an unqualified initialization-performance figure. Served compiled asset SHA values are checked against local files. Exact pose/artistic continuity remains an independent visual judgment; CSS box measurements alone do not certify painted identity quality.

## Evidence and reproducibility

All artifacts are under `.artifacts/v2-completion-20260906/skyy/`:

| Evidence | Result |
| --- | --- |
| `completion.json` | PASS nine viewport/loading cases; opacity/geometry/gait/time samples and hashes |
| `cinematic-integration-skyy-home-controls.json` | PASS controls, lifecycle, lightweight and no-JS cases |
| `failures.json` | PASS model/guide failure states |
| `settled-dialogs.json` | PASS eight settled dialog/form capture cases |
| `390-handoff.webm`, `slow-390-handoff.webm` | Unedited native browser recordings |
| `{case}-loading.png`, `{case}-onset.png`, `{case}-idle.png` | Actual Home presence states |
| `{width}-dialog-settled.png`, `{width}-dialog-input.png` | Steady conversation and reachable form, including landscape |
| `model-failure-home.png`, `model-failure-dialog.png`, `guide-failure-contact.png` | Branded failure-state evidence |

Artifact helpers: `verify-skyy-completion.cjs`, `verify-skyy-home-controls.cjs`, `verify-skyy-failures.cjs`, `capture-skyy-dialogs.cjs` in `.artifacts/v2-completion-20260906/`. Each has an independent RUNNING/PASS/FAIL receipt and local-only network guard. Run them serially when root grants exclusive browser access; do not overlap Lighthouse or other browser jobs. `SKYY_CASE=390` bounds the main helper to a diagnostic case.

Two early helper failures are retained separately. `390-first-failure.json` captured an assertion made before native close/focus processing completed; the helper now waits for the actual lifecycle. `failures-first-attempt.json` captured an assumed non-null navigation response; final verification instead checks the actual destination plus a redirect-disabled local HTTP200 read. Neither early receipt is presented as successful verification.

Browser ownership was returned to root after all contexts closed. Independent visual/source acceptance, wider theme performance, packaging and founder approval are not claimed by this implementation report.
