# Final JavaScript review — Skyy walk-on and visual recovery

Status: **APPROVE for the final reviewed source scope, including runtime rig motion**. The focus-restoration defect, two rig-motion integration findings, and diagnostic lifecycle finding are resolved on final reread. No remaining actionable critical, high, or medium findings were identified. This is a source and synthetic-DOM review, not browser/GPU, deployment, or visual approval.

Reviewed in `/Users/theceo/.codex/worktrees/19db/DevSkyy` against local HEAD `fec4e9339cad7077bcc812876668c128eb93b6e4`. Staged diff was empty; unstaged changes and new `visual-recovery.js` established the scope. This is a local review; PR merge readiness was not assessed.

## Scope and checks

- `wordpress-theme/skyyrose-flagship-2/assets/js/mascot.js`
- `wordpress-theme/skyyrose-flagship-2/assets/js/mascot-loader.js`
- `wordpress-theme/skyyrose-flagship-2/assets/js/skyy-3d.js`
- `wordpress-theme/skyyrose-flagship-2/assets/js/visual-recovery.js`
- The changed button lookup in `wordpress-theme/skyyrose-flagship-2/assets/js/collection-scene-motion.js`, with surrounding lifecycle context.

Read the corresponding dialog, header and home markup plus enqueue/localized-data wiring and character CSS. The bounded theme package contains JavaScript, with no TypeScript check script or owning TypeScript configuration; the monorepo root TypeScript command checks a different source tree. ESLint was unavailable in PATH and the root, WordPress, and theme local executable directories, so no lint success is claimed.

Executed after the focus fix and again after the runtime rig-motion addition:

```sh
node --test tools/v2-runtime/test-skyy-concierge.cjs tools/v2-runtime/test-guide-focus.cjs tools/v2-runtime/test-visual-recovery.cjs
```

Result on the final frozen rig-motion delta: **30 passed; 0 failed**. This includes native-link gestures, late guide completion, unavailable WebGL, dependency timeout fallback, reduced-motion/Save-Data gating, real SKU links, inert answer text, home visibility and dismissal, stage reparenting, corrected opener focus, deferred video loading, pagehide/pageshow, late video-play completion, and a real Three AnimationMixer test using the canonical GLB's bone hierarchy and animation data. Added regressions verify repeated transient duration renewal, cleared rig references, and safe diagnostics after teardown. The actual-rig test also verifies that the relaxed hand is below its shoulder.

## Resolved rig-motion delta findings

1. **P2 / medium — legacy entrance animation masking Home traversal. Resolved.** The original generic walking-in animation overrode the normal Home transform driven by `--skyy-entry-progress`. Final `assets/css/mascot.css:350` scopes `skyy-concierge-enter` to `#skyy-ask-dialog`, allowing the renderer-clock transform to control Home traversal.
2. **P2 / medium — repeated transient actions ending too early. Resolved.** A second talk request at elapsed 2.3 seconds previously retained that elapsed value and ended after another 0.1 seconds. Final `skyy-3d.js:340` resets elapsed time on each transient request while preserving clip phase for the same current action. The production-function regression verifies the full renewed 2.4-second interval.
3. **Low — diagnostic exception after teardown. Resolved.** Final teardown clears `rigMotion`, and `getMotionEvidence()` independently guards both the rig and model. The regression invokes the real teardown and getter implementations and verifies a null result.

The current GLB contains six held poses with no varying values across its tracks. The new runtime derivation creates movement on the existing rig; it does not reveal previously baked animation. The actual Three mixer test verifies changed bone quaternions and unchanged source-track serialization. The arithmetic normalizes composed quaternions and uses parent-space shoulder offsets versus local-space remaining-joint offsets. Physical gait quality and camera composition require the author's actual pixel validation. Future clips detected as varying are returned by identity rather than being rewritten.

Final reread also covers the bind-pose shoulder/forearm bases and first-frame initialization. The source bind transforms are cloned before offsets are composed; source clip objects remain unchanged. On the first action, `mixer.update(0)` applies the pose without fading through the unloaded rest pose, and initializes facing/entry progress before rendering. No actionable correctness defect was identified in those final changes. Removing the unused scene-shift style write in `visual-recovery.js` leaves the navigation index and scroll logic intact.

## Resolved finding

**P2 / medium — restore the moving hero opener before focusing it.** The original close handler tried to focus the hero Ask Skyy button while its stage still lived inside the closed dialog; the dialog CSS also hides hero actions there. A read-only synthetic reproduction with closed-dialog focus rejection returned `restoredStage:true, focusReturned:false, focusIsBody:true`.

The final `mascot.js:243–258` captures whether focus needs restoration, calls `restoreHome()` first, then checks connectivity, rendered rectangles, and hero visibility/dismissal before choosing the original opener or the persistent header link. The new `hero conversation opener is restored after its stage leaves the closed dialog` regression passes. The author also strengthened the fixture's hidden-ancestor behavior. Finding resolved on reread.

## Reviewed behavior and limits

- Answer content uses `textContent`, links require HTTP(S) and the current origin, and product data comes from published, visible, non-password-protected WooCommerce products. No external chat request or commerce mutation is added by these modules.
- The 3D path validates local asset origins, rejects redirected model fetches, bounds dependency/fetch/decode waits, validates the skinned model and all six populated canonical actions, and falls back to the portrait on failure. Disposal and late decode paths were inspected.
- Hidden, paused, lightweight, and background states stop the renderer loop; home preparation waits for poster decode, page load, idle scheduling, and current home visibility. Header invitation remains available after home dismissal.
- Video initialization waits for responsive image decode and preference/visibility checks; late play completion checks current state before allowing continued playback.
- No browser, Lighthouse, live site, paid provider, build, or deployment operation was performed by this reviewer. Actual animation quality, GPU frames, production origin behavior, and final minified-asset parity belong to the main task's validation.

## Reviewed source SHA-256

| Source | SHA-256 |
| --- | --- |
| `mascot.js` | `2e79443aedda9ee1f77b813d7ae7dd0c921e4f71fe6cfd4db95b8e746eb14135` |
| `mascot-loader.js` | `c03b2549f42fa46d8fd3638d9f2d35f73b6effd0011f880135abc65ce4aeb36a` |
| `skyy-3d.js` | `19ac08881f68d7a44e06a9af8eb25d13e5ddd88227e1c16d3fd2e3a8be27b535` |
| `visual-recovery.js` | `f3180c63f938fed98725ec9f911c8049326da1a742843e362aaf0d8466a738ad` |
| `collection-scene-motion.js` | `0772f847a945c67fe8accbc156c0d0eea41429b88ef7fb494437c3d3704dfe80` |
