# Ask Skyy runtime readiness

**Software startup improvement verified; asset performance remains blocked.** The final lifecycle-safe candidate reduces the measured initialization stalls and preserves the exact first visible frame. Model size, dynamic runtime size, request count, decoded peak memory and frame cadence remain open gates. Two local samples are not physical-device certification.

The approved model, rig, identity, poster, camera, lights, intent-triggered loading, conversation behavior and 15 fps idle / 30 fps active targets are unchanged. No CSS, loader, model or poster edits belong to this pass. Only `skyy-3d.js`, its deterministic minified output and focused tests/helpers changed.

## Measured cause and software repair

The model contains 1,030,595 vertices. Three r170 synchronously transforms every skinned vertex to compute its initial bounding box, then does another full scan for the bounding sphere used during first render. The final baseline traces measure the box at 104.9–168.9 ms and sphere at 101.8–187.5 ms. These scans explain a large portion of the uninterrupted startup task; texture uploads and animation setup were much smaller.

The candidate computes exactly the same box and sphere, in the same vertex order with Three's own `getVertexPosition` and `expandByPoint` math, but yields between chunks. The box is calculated before the existing normalization. The sphere is calculated at the same derived idle time-zero pose used by the first frame. Partial bounds are never published. Disposal is checked before every chunk and after asynchronous preparation.

Shader preparation begins with `renderer.compile` before the cooperative sphere scan, allowing driver work while those chunks run. Each embedded texture is initialized with a yield between uploads. This does not claim that shader compilation or GPU execution has completed at the `compile` return; the first real render remains authoritative. The fetched ArrayBuffer reference is released after parsing.

An intermediate version used `renderer.compileAsync`. Independent review correctly identified an r170 lifecycle defect: its delayed material poll can dereference disposed program properties after context loss or permanent pagehide, throwing outside its Promise and leaving it pending. The final source removes that uncancellable poll entirely. Its only asynchronous boundary belongs to the controlled scheduler and checks disposal before continuing. The regression extracts the actual bundled r170 implementation, reproduces the exact poll/disposal failure and verifies the replacement's cancellation behavior.

## Final paired startup result

Serial fresh Chromium contexts used ANGLE Metal on the same desktop host, with frozen baseline `18423` and current `18416`, a 390 × 844 viewport and actual Ask invitations. Root reserved the browser window; other browser jobs were closed. The final paired run occurred after fixture routing was corrected to resolve each theme's own configuration. Synchronous Three/WebGL diagnostic wrappers were identical in both conditions. These are CPU submission measurements, not GPU execution timers.

| Metric | Frozen baseline, two samples | Final lifecycle-safe candidate, two samples |
|---|---:|---:|
| Character startup maximum long task | 511 / 315 ms | **50 / 54 ms** |
| First-render CPU submission | 262.9 / 151.5 ms | **32.2 / 34.5 ms** |
| First stable frame from runtime start | 2003.9 / 1740.5 ms | **1769.0 / 1590.4 ms** |
| GLB requests per fresh invitation | 1 / 1 | 1 / 1 |
| First visible frame | Reference, 220 × 340 RGBA | **Byte-identical in both samples** |

The maximum-task 100 ms and first-stable 2000 ms gates pass in these final local samples. Earlier candidates exceeded the first-frame gate; they remain archived rather than being blended into the final result. The current samples do not establish universal device performance or guarantee every initialization will stay below these thresholds.

Final current phase elapsed ranges:

| Phase | Elapsed time |
|---|---:|
| Local module loading | 17.0–18.9 ms |
| Model fetch | 15.6–16.6 ms |
| Parse/decode | 1274.0–1446.5 ms |
| Cooperative initial box | 105.9–107.3 ms |
| Renderer initialization | 7.4–11.6 ms |
| Animation setup | 4.6–4.8 ms |
| Shader preparation kickoff and yield | 3.2 ms |
| Cooperative idle sphere | 103.9–104.1 ms |
| Texture upload and yields | 11.4–12.5 ms |

Box and sphere elapsed times span multiple tasks. Parse/decode includes worker and image activity; it is not main-thread CPU duration. Driver allocation, decoder transients and JavaScript heap were not directly measured. Reducing the geometry remains the principal asset-side remedy.

## Steady-state cadence

A separate serial baseline/current test waits for the real walk, then measures three seconds of settled idle and 1.8 seconds of the shipping-answer talking action. It resets frame instrumentation between states. It does not change the 15/30 fps policy.

| State | Baseline | Current |
|---|---:|---:|
| Idle delivered cadence | 14.92 fps | 14.75 fps |
| Idle frame interval p95 | 66.8 ms | **83.3 ms** |
| Active delivered cadence | 28.07 fps | 28.91 fps |
| Active frame interval p95 | 50 ms | **50 ms** |
| Idle CPU render submission p95 | 0.3 ms | 0.4 ms |
| Active CPU render submission p95 | 0.4 ms | 0.3 ms |

The current idle 66.8 ms and active 33.4 ms interval gates remain **FAIL**. CPU submission remains below 8 ms. Startup scheduling does not claim to fix frame pacing, sustained thermals or GPU performance on phones.

## Lifecycle, appearance and resource verification

- **33 focused tests pass**, including exact real-Three box/sphere equality, yielding, safe cancellation without partial bounds, and the actual r170 shader-poll failure regression.
- The final four-run browser profile observes real `Skyy_Walk`, settled `Skyy_Idle` and shipping conversation. One GLB is fetched per fresh context.
- First visible current frames are byte-identical to baseline at 220 × 340. The approved 330 × 510 poster is untouched.
- Actual `WEBGL_lose_context` during the owned shader yield produces static fallback, zero frames, `ready=false`, no page exceptions and working shipping text.
- A controlled non-persisted `pagehide` event during that yield stops initialization cleanly with no page exceptions. This is explicitly an event simulation, not a navigation timing claim.
- Intent loading retains one dependency promise for Three/GLTF/Draco imports and a started guard for the model. One Draco worker handles the sole parse and is disposed afterward. Retaining it indefinitely would retain memory without a second model to decode.
- DPR remains capped at 1.5, with no shadows or post-processing. Existing offscreen, document-hidden and pause handling stops rendering. Conversation remains independent of GPU success.

## Remaining budgets and chunk tradeoff

The GLB remains **6,058,568 bytes**, with **1,930,256 triangles**. Typed geometry plus estimated texture storage remains **89.20 MiB**, a lower bound rather than peak memory. The minified controller grows from 12,440 to **13,847 bytes** (+1407 bytes) for cooperative preparation and bounded timing diagnostics. Dynamic runtime remains above 900 KiB, and model payload remains above both the existing 2500 KiB gate and the requested eventual 2.5 MiB asset target. The established scoped request set remains 13; no extra asset is fetched by this repair.

The frozen implementation uses **8192 vertices per chunk**: 126 chunks and 125 yields per scan, or 252 chunks / 250 yields across box and sphere. It uses `scheduler.yield()` when available, otherwise `setTimeout(0)`. This is a vertex budget, not a guaranteed wall-clock deadline on every device.

The final box/sphere elapsed spans imply about 0.82–0.85 ms elapsed per chunk when divided by 126, but individual CPU chunks and isolated scheduling overhead were not directly instrumented. That quotient must not be presented as measured per-task CPU duration. Larger 16384/32768 chunks remain untested future tuning and were not applied after freeze. They require another matched maximum-task / total-readiness comparison and the same exact-bounds/cancellation tests.

## Evidence and authoring handoff

All runtime evidence is in `.artifacts/v2-readiness-20260906/skyy/`:

- `runtime-before-after.json`: final lifecycle-safe paired startup profile.
- `first-frame-continuity.json`, `before-*-first.png`, `current-*-first.png`: final exact pixel comparison.
- `shader-lifecycle.json`: actual WebGL loss and controlled permanent-pagehide tests.
- `cadence-before-after.json`: isolated idle and talking cadence samples.
- `unit.log`: 33 passing focused tests.
- `bounds-only-before-after.json`, `shader-texture-before-after.json`, `pre-lifecycle-fix-before-after.json`: superseded experiments, excluded from final certification.
- `pre-lifecycle-fix-images/`: archived images before the lifecycle correction.
- `evidence-index.json`: final source and evidence hashes.

Reproduction helpers are `profile-skyy-readiness.cjs`, `test-skyy-shader-lifecycle.cjs` and `profile-skyy-cadence.cjs` under `tools/v2-runtime/phase3b-browser/`.

`skyy-blender-handoff.json` and `.md` contain the exact immutable input, complete 18-bone transforms/hierarchy, skinning, materials, textures, root normalization, camera/lights, animation naming, poster state and ordered optimization checklist. Blender 5.2.0 LTS is installed; only its version was queried. No model import, authoring, replacement, packaging, promotion, deployment or staging modification occurred.

### Timing boundary clarification

`firstStableFrameMs` is measured from `skyy-3d.js` runtime `boot()` (`profile.startedAt`) to its first stable render. It is not an end-to-end click-to-completed-walk metric; the loader work before runtime boot and the subsequent reveal/walk choreography are separate. All paired samples use this same boundary. The captured renderer identifies ANGLE Metal on AppleM5; this establishes the local renderer backend, not physical mobile performance or GPU execution-time certification.
