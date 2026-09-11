# Independent readiness review

Scope: current `assets/js/skyy-3d.js` and its generated delivery counterpart against the immutable `.artifacts/v2-readiness-20260906/baseline-theme/skyyrose-flagship-2`, not Git HEAD. Source review only; no browser, build, deployment, or asset changes by reviewer. Theme is JavaScript-only and has no theme TypeScript/ESLint command. The focused Node suite was rerun independently: **32/32 PASS**.

## Finding requiring resolution

**HIGH — disposal during asynchronous shader polling can throw an uncaught exception.** New `skyy-3d.js:571` awaits `renderer.compileAsync(scene, camera)`. The existing context-loss and permanent-pagehide handlers can call teardown during that await, disposing materials and renderer. The bundled Three r170 implementation polls `tt.get(material).currentProgram.isReady()` from a `setTimeout` callback. Material/renderer disposal removes the properties that callback expects. A subsequent poll then throws outside the Promise executor; the awaited promise does not reject or resolve. The post-await `disposed || failed` guard therefore cannot prevent or catch the exception.

Confirmed independently without WebGL by extracting the exact bundled `compileAsync` function, returning an unfinished program on its first poll, removing the material properties as disposal does, and invoking its scheduled callback: `TypeError: Cannot read properties of undefined (reading 'isReady')`. This is an extracted-library lifecycle reproduction, not a real-browser context-loss reproduction. Fix shader preparation/disposal ownership so pending library polling cannot access disposed state; cover cancellation during shader preparation explicitly. A timeout alone does not cancel Three's internal polling.

## Verified behavior and evidence boundaries

- Cooperative box and sphere accumulation uses the native vertex order and math. The real bundled Three regression verifies exact equality for both and confirms cancellation without publishing a partial box. Completed bounds are assigned only after a whole mesh; disposal checks bracket subsequent startup stages and texture-upload yields.
- The initial idle pose, scale, camera, lights and visible-render handoff remain preserved in the reviewed delta. The new texture loop deduplicates embedded material textures and yields between uploads.
- Read `skyy/runtime-before-after.json`, `first-frame-continuity.json`, `walk-idle-chat.json`, and the owner runtime report. Two paired Metal Chromium samples show startup maximum tasks **396/332 → 87/52 ms** and first stable frames **1909.0/1949.4 → 2000.3/2133.1 ms**. The candidate trades total readiness delay for shorter blocking. Both candidate samples miss the 2000 ms first-stable target; these samples do not establish universal device performance.
- Recorded first-visible RGBA equality at 220×340 is byte-identical in both pairs. This is narrow first-frame continuity evidence, not temporal animation or physical-device certification. The separate action journey supports local walk/idle/chat continuity.
- CPU submission, phase elapsed time including yields, decoded memory estimates and actual GPU execution are distinct. Existing report correctly preserves those distinctions and the model/runtime-byte blockers.

## Follow-up: shader correction and route/font scope

The shader-polling finding above is **RESOLVED** in the reviewed source. `prepareShaders` now starts `renderer.compile` synchronously and yields only through the controller's own lifecycle. It checks cancellation before compiling and immediately after that yield. The uncancellable vendor poll is no longer called, including in the minified delivery file. The new regression reproduces the old failure using the actual bundled poll, then verifies cancellation during the replacement yield and refusal to compile already-disposed state. Independently rerun focused suite: **33/33 PASS**.

The quoted timing results above describe the preceding compileAsync candidate, not the corrected synchronous-compile candidate. Updated matched startup profiling and pixel continuity evidence are required before attributing those exact numbers to this source. Synchronous compile is a potential blocking step; its final duration must be measured rather than assumed equivalent.

Additional immutable-baseline source review covered `functions.php`, `inc/performance.php`, `assets/css/shop-page.css`, `assets/css/design-tokens.css`, `theme.json`, and `scripts/test-performance.php`:

- Cart/Checkout now omit the legacy-world and editorial-content composition sheets. Shared shell and controls stay loaded; examined content selectors target editorial/contact/about/reserve compositions, not native transaction forms. This does not establish compatibility with arbitrary merchant-added editorial content in checkout.
- Only Shop/product-taxonomy requests dequeue Woo layout/smallscreen handles. Woo general forms/notices remain. The filter `skyyrose2_archive_native_woo_layout` restores the native layout for extensions. The archive stylesheet loads on the matching archive route, and the product wrapper's explicit relative positioning replaces the removed native positioning rule.
- Inter's unused face and secondary fallback are removed coherently from CSS and theme.json. Hanken Grotesk remains first-choice body typography; unusual missing-glyph/font-failure rendering now uses generic sans-serif. Settled normal-font screenshots cannot establish equivalence for those fallback conditions.
- Independently executed `scripts/test-performance.php`: **PASS**; changed PHP syntax checks pass. New test checks both Shop and taxonomy removal and restore-filter behavior. Read the CSS ablation JSON: Cart and Checkout settled records at both sampled widths have zero computed-style differences. Shop settled, Quick View and filter records have documented computed differences before the explicit positioning replacement; do not call the entire ablation a zero-computed-difference result.

Final scoped **SOURCE APPROVE**: no remaining actionable correctness defect identified in these deltas. This is not performance certification, full extension compatibility, or physical-device certification. No browser or build was run by this reviewer; root owns refreshed final-candidate profiling, native commerce journeys and generated-asset verification.

## Final evidence and corrected font delivery

The later safe-compile evidence has now been read independently. Fresh matched `runtime-before-after.json` records baseline maximum startup tasks **511/315 ms**, first stable **2003.9/1740.5 ms**; corrected candidate tasks **50/54 ms**, first stable **1769.0/1590.4 ms**. These replace the earlier candidate numbers for current-source reporting. Both current samples satisfy those two timing gates locally; the sample size and desktop Metal environment remain explicit limits. First-visible RGBA equality remains true in both 220×340 pairs.

`shader-lifecycle.json` records actual `WEBGL_lose_context` during the owned shader yield: zero rendered frames, ready false, static fallback and no page exceptions. Synthetic non-persisted pagehide at that point also records stopped rendering, ready false and no exceptions. This supports the source correction; the synthetic event is not an actual navigation test. No independent browser was launched by reviewer.

The earlier complete Inter removal is **superseded/rejected** after paired visuals exposed changed arrow glyphs. Final CSS restores the original Hanken/Inter body stack and registers a derived 1872-byte Inter face restricted to the 11 codepoints absent from Hanken. Independently checked using the supplied fontTools 4.59.2 interpreter:

- Original Inter cmap minus Hanken cmap equals the derived subset cmap exactly: U+0000, U+02BB–02BC, U+2002, U+2009, U+200B, U+2032–2033, U+2191, U+2193, U+FEFF.
- Decomposed glyph outlines and advance widths are identical for all 11 glyphs at weights 100, 400, 700 and 900. Decomposition avoids falsely treating renamed internal composite references as changed artwork.
- Source and output SHA-256 match `data/font-fallback.json`; original source remains 48432 bytes. Repeating the manifest recipe into a temporary directory yields byte-identical output (SHA-256 `9561a48d0deadad8b3b17f0f9a2b384c14baedb0c6d3e53738a91c0641b2471e`). Source guard pins both derived output and manifest.

This proves reproducible subset delivery and preserved tested glyph geometry. Root's refreshed 22 paired browser captures remain the visual integration gate. It does not prove equivalence when the primary Hanken font itself fails to load: removed shared Latin glyphs would then use generic sans-serif rather than full Inter. No font/source blocker identified for the normal loaded-primary condition under review.
