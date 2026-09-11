# Ask Skyy software completion and runtime review

Status: **NEEDS_MORE_WORK for performance certification.** Available-browser software states and the same-model continuity checks pass; the startup long task and model/runtime payload and cadence budgets do not; peak memory remains unverified. Final 320/414/768px visual and Axe checks pass; the narrow transcript is keyboard-scrollable and the Ask composer stays visible. This report does not certify a physical mobile GPU, a screen reader, field INP, or founder visual approval.

The approved GLB, embedded textures, face, clothing, skeleton, and original portrait are preserved. The new stage poster is a deterministic local frame from that same model, runtime camera, lights and relaxed idle pose. The header still uses the original approved portrait. No paid generation, model authoring, Blender operation, or deployment occurred.

## Implementation and continuity

- Home shows the same-model static guide immediately. Heavy modules, Draco and the GLB wait for character hover, pointerdown/touch, keyboard focus on Ask Skyy, or an explicit Ask invitation. The real walk remains available after intent; no diagnostic disable flag ships. Actual Metal tests observed zero optional 3D requests for 10 seconds before interaction, then exactly one GLB for hover, touch and focus independently.
- The renderer paints the time-zero relaxed idle pose before emitting its visible event. That frame remains stable through the 300ms portrait/canvas handoff, then the existing rig takes a restrained two-step arc and turns back to idle. Removed the initial sideways rotation and the independently enlarged 1.28x portrait.
- The same stage and transcript move between Home and the native conversation dialog. Minimize closes the dialog and preserves the transcript for header recall. Removed the superseded CSS entrance fade that briefly hid the character on chat opening.
- Listening, actual synchronous deterministic lookup/thinking, greeting, talking, gesture, chat failure, chat open/closed/minimized, visibility and motion states are recorded independently. No artificial remote-chat delay or simulated remote service is introduced.
- The guide waits for actual clip completion before reporting idle. A wall-clock timer previously reported idle while a heavily loaded renderer was still walking.
- A missing site-guide data contract produces an honest contact fallback. Reduced motion, Save-Data, WebGL/model failure preserve the static guide and text interaction. The original portrait also remains an image-request fallback.
- Idle rendering targets 15fps; active clips target 30fps. A 0.5ms tolerance prevents rounded 33.3ms RAF timestamps being rejected against 33.333ms and unnecessarily dropping to 20fps. Offscreen, hidden, dismissed and paused states stop rendering.
- A preference change mid-walk now resets the horizontal arc offset to 0px before revealing the matched first frame. The independent reviewer identified this restart defect and it was fixed and browser-tested.
- The stage poster is lossless WebP: 61,276bytes, down from 122,305bytes for its raw canvas PNG. Both have identical decoded pixels. This is local encoding of a rendered frame, not an edited identity.

At 390px and 1440px, the poster and actual first stable Metal frame have identical 330×510 dimensions and alpha bounds `(55,43)-(275,475)`. Silhouette intersection/union is 99.8974%. Opaque RGB mean absolute difference is 0.989/255, with a 95th-percentile channel difference of 5/255, between the SwiftShader-generated poster and Metal-rendered first frame. The shader backends do not produce byte-identical pixels; there is no scale, anchor or pose discontinuity. The actual restart test also verifies that canvas and portrait rectangles align exactly after a reduced-motion toggle mid-walk.

## Complete software state matrix

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

## Actual browser and accessibility checks

- Chromium Metal: matched before/current 390px and 1440px walk, idle, chat, greeting/talking, gesture, pause, minimize and dismissal recordings ; 14 Tab traversals remain inside the dialog.
- Firefox: actual 3D ready, native product-link answer, 10 Tab traversals, Escape close, zero page exceptions. Exposed renderer string is `Apple M1, or similar`; this may be privacy-rounded and is not taken as exact hardware identification.
- WebKit: actual 3D ready, native product-link answer, 10 Tab traversals, Escape close, zero page exceptions. Exposed renderer string is `Apple GPU`.
- Browser-emulated narrow layouts at 320/414/768 are checked separately from the 390/1440 3D review. The initial 320 capture showed the question composer below the dialog fold. The final ≤359px correction keeps the full-body guide, uses two columns for suggestions and bounds the scrollable transcript so the question and Ask action stay visible. All three final captures pass zero Axe violations, fit the viewport, and preserve focus on the input. At 320px the transcript has a visible keyboard focus treatment and PageDown scrolls it; the first greeting starts at its beginning.
- No physical phone was available through these tools. Viewport emulation is not a real-device claim. A screen reader was not exercised. A native keyboard and focus test is not screen-reader certification.

## Model and resource profile

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

## Exclusive A/B/C/D/E profile

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

## Budgets and failures

Primary certification uses the predeclared `performance-budgets.json` Ask Skyy gates. Stricter future asset-authoring targets are separate proposals, not substituted gates.

| Canonical Ask Skyy gate | Result |
|---|---|
| Bootstrap JS ≤20KiB |PASS:12,879bytes loader+guide|
| Dynamic 3D controller/runtime/decoder ≤900KiB |FAIL:1,204,379bytes|
| Model ≤2500KiB |FAIL:6,058,568bytes|
| External textures ≤256KiB |PASS:0 separate downloads; texture payload is embedded in GLB|
| Decoded textures ≤48MiB |PASS estimate:16MiB RGBA8/mips, not peak driver allocation|
| Decoded total ≤96MiB |UNVERIFIED peak:89.20MiB geometry+texture lower bound excludes transients/driver/heap|
| First stable frame ≤2000ms |PASS in this local Metal sample:1622.1–1663.6ms after intent; physical mobile unverified|
| Initial Home extra model requests =0 |PASS:all A–E initial windows0; independent10-second no-intent observation also0|
| Scoped request count ≤12 |FAIL:13 including same-model poster|
| Initialization maximum long task ≤100ms |FAIL:293–310ms after intent|
| Long tasks over50ms count ≤4 |PASS local sample:1 in each D/E run|
| Idle frame interval p95 ≤66.8ms |FAIL:83.3ms|
| Active frame interval p95 ≤33.4ms |FAIL:50ms|
| CPU render submission p95 ≤8ms |PASS:0.6ms idle /0.5ms active; GPU execution unmeasured|
| Ask Skyy CLS ≤0.01 |UNVERIFIED by this component profile; root route traces remain authoritative|
| Interaction ≤200ms / physical mobile INP |UNVERIFIED by this component profile; no field/physical-device certificate|

The final 7.38MB scoped transfer and 293–310ms startup task remain material failures. Historical before-intent measurements were 7.31MB (poster excluded) and284–301ms; moving preparation behind character intent fixes initial-load contention but does not eliminate the asset initialization cost. State A is diagnostic-only and is not shipped. The full character activates on intent with the same approved rig and actual motion.

Proposed deferred asset-production targets: GLB≤2MiB, website LOD≤100k triangles, decoded geometry+textures≤32MiB, desktop first-frame≤1s, and stage poster≤64KiB. Only the poster currently meets this stricter proposed set. These proposals do not redefine the canonical certification gates above.

## Visual review and acceptance boundary

| Character presentation | Engineering visual comparison |
|---|---|
| Static→first stable 3D |BETTER: matched same-model pose, scale, camera and anchor with measured silhouette proof|
| Walk/turn |BETTER continuity; the same rig genuinely moves, with a restrained path instead of the initial sideways/position jump|
| Settled idle |EQUIVALENT approved model/identity, camera and lighting|
| Chat opening |BETTER: painted character remains visible; duplicate opacity entrance removed|
| Chat controls |BETTER: explicit Minimize, retained transcript, visible narrow-phone composer after final CSS check|
| Failure/reduced-motion guide |BETTER continuity: same-model poster and clear accessible state; conversation remains usable|

These are engineering comparisons for founder review, not founder approval or asset-authoring certification. Named review recordings are the valid review set; interrupted/randomly named capture files are not certified review material. Character source stayed identical across the router correction; background hero codec behavior in older captures should not be mistaken for a character change.

## Deferred Blender/asset-production backlog

1. Produce founder-reviewed website LODs from the unchanged character identity, preserving face, outfit, silhouette and rig weights. The current 1.93 million triangles and 76.75MB geometry arrays are the measured primary asset costs. Target below 100k triangles and validate at actual 96–220px display widths.
2. Validate topology and skinning after reduction, especially curls, fingers, sleeve/hood/jacket seams and shoes. Do not modify identity to reach the triangle target.
3. Bake and review actual walk/turn/idle/greeting/listen/talk/gesture/exit clips on the same approved rig. Current source clips are held poses; the website derives movement at runtime.
4. Retain current 1024×1024 normal, roughness/metallic and base-color fidelity; measure any approved texture-format/packing change against the current material appearance. Geometry optimization has the larger demonstrated payoff.
5. Re-capture posters from the approved camera and first stable pose after any authorized asset revision, and rerun the pixel/anchor and dynamic reduced-motion restoration checks.
6. Measure physical mobile GPU memory, frame times, initialization long tasks, thermal behavior and context recovery before issuing device-performance certification.

## Evidence files

All evidence is under `.artifacts/v2-cinematic-finalization-20260906/skyy/`:

- `final-evidence-index.json`: source hashes and authoritative final review evidence.
- `poster-provenance.json`, `poster-source.png`, `glb-profile.json`.
- `continuity-390.json`, `continuity-1440.json`, `continuity-pixels.json`, actual first-stable-frame PNGs.
- `state-matrix-current-all.json`: final intent-driven390/1440 journeys,12 rows,zero page errors. Earlier width-specific JSON files predate the final intent gate. `state-matrix-all-all.json` supplies the five injected failure/preference cases.
- Named `before-{390,1440}-review.webm` and `current-{390,1440}-review.webm`, plus corresponding screenshots.
- `cross-engine.json`, `engine-firefox.png`, `engine-webkit.png`.
- `runtime-profile.json`, its log and A/B/C/D/E screenshots.
- `mobile-art-direction.json`, `mobile-{320,414,768}.png`, `mobile-320-log-focus.png`.
- `intent-gate.json`: actual hover, touchscreen tap and keyboard focus activation.
- `before-intent-gate/runtime-profile.json`: preserved earlier automatic-Home-loading profile; `runtime-profile-overlap-diagnostic.json` is excluded from primary performance claims.
- `unit.log` (31 passing tests).

Reproduction tools are the scoped `capture-skyy-poster.cjs`, `test-skyy-continuity.cjs`, `test-skyy-finalization.cjs`, `test-skyy-mobile.cjs`, `test-skyy-engines.cjs`, `test-skyy-intent.cjs` and `profile-skyy.cjs` scripts in `tools/v2-runtime/phase3b-browser/`. The poster generator additionally requires Pillow and accepts `V2_PYTHON` for the pinned interpreter.
