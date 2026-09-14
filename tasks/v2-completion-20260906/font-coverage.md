# Independent Inter coverage validation

Reviewer: `/root/final_visual_review`. Date: 2026-09-06. **Bounded verdict: PASS for the observed font-range change.** This does not change the site's mobile performance release verdict or authorize deployment.

The parent added `unicode-range` to the existing Inter face. The reviewer made no runtime source edits. The original Inter and Hanken files have identical before/after hashes; no font bytes, font stack, weight range or display policy changed in this task. Browser windows were coordinated with the PR agent, all contexts closed, and no Lighthouse ran concurrently.

BEFORE: 14:15:34.300–14:15:42.167 UTC. AFTER: 14:17:29.836–14:17:37.477 UTC. Chromium used isolated contexts at390×844 and1440×1000, reduced motion, no heavy3D. Read-only Home requests used the raw local fixture at127.0.0.1:18303. The glyph sample was injected only into a disposable browser DOM; the Hanken404 was a test-only request interception. No server font or response behavior was changed by the reviewer.

| Case | Before | After | Result |
|---|---|---|---|
| Home390 | Inter downloaded,48,432 encoded bytes | No Inter request | PASS |
| Home1440 | Inter downloaded,48,432 encoded bytes | No Inter request | PASS |
| Synthetic Inter `↑ ↓ ′` and Latin | Actual custom Inter glyphs; font downloaded | Same actual custom Inter glyphs; font downloaded | PASS |
| Synthetic Inter `← →` | Actual system Lucida Grande glyphs | Same system glyphs | PASS |
| Hanken404 | Intro renders with custom Inter; font downloaded | Same custom Inter; font downloaded | PASS |

Chromium `CSS.getPlatformFontsForNode` confirms the rendered Home heading uses Archivo, intro uses Hanken Grotesk, and both CTAs use Anton at both widths. Font attribution, glyph counts and all21 measured text-node records are identical before/after, including each node's x/y/width/height, computed family, size, weight, line height and letter spacing. The failure probe still uses Inter for73 intro glyphs. All eight case executions recorded zero page errors; the intentional Hanken404 is explicitly retained as the test condition.

All eight screenshots were directly inspected with `view_image`. Decoding both PNGs in each pair to RGBA shows **zero differing pixels across all four complete viewport pairs**, as well as zero differences within each measured heading/body/CTA/glyph rectangle. The comparison covers2,427,480 pixels per revision. This is exact parity for these deterministic reduced-motion captures, not a universal browser/font/rasterizer guarantee.

Evidence root: `.artifacts/v2-completion-20260906/font-coverage/`; `before/receipt.json`, `after/receipt.json`, all eight adjacent PNGs, and `comparison.json`. Reproduction runner: `check-fonts.cjs` with phase argument `before` or `after`. The isolated probes preserve the useful fallback path while demonstrating that the normal-page Inter request is omitted; no LCP gain is inferred without a new exclusive measurement.

Source bindings:

- Inter file unchanged: `c940764593d0fe5d596be327ca7558855e018039fb78509aa21921fd3644c3e4`.
- Hanken file unchanged: `1f21c6eaa0000f3329cfcfac966b43d5bebf5aa610303e33294ac31bc6f4bb59`.
- BEFORE design-tokens.css: `c0286ff8785d21c1eeeb60340ce812d6b92d22d1dd29fcfbdec8eb768616ea5b`.
- AFTER design-tokens.css: `152edd95cf807d45f868ad452f15336209b28e1ef6987b4eda62f75ea49359f1`.

Limits: the reviewer sampled Latin and the explicitly requested arrows/prime; exhaustive cmap-to-range equality belongs to the separate source audit. Other languages and browsers were not newly certified. All original font bytes remain available. Global release approval remains separate from this bounded regression PASS.
