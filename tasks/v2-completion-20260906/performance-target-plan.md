# Performance target closure — active follow-up

The founder explicitly directed continued optimization after the first candidate failed mobile targets. The earlier manager report is a checkpoint, not completion of that instruction. Preserve aabd2bffd and its evidence; implement substantive improvements and measure, never hide the failed measurements.

## Verified bottleneck

Current final simulated mobile LCP: Home5.80s, PDP5.19s, Shop5.72s (repeat5.65s). Main-thread blocking is24/1/0ms. All three have large uncompressed styles/native scripts; the fixture has no text compression. The exact original LCP image/hero/card frame remains present. Observed localhost LCP is a different measure and is not substituted for simulated mobile LCP.

## Experiments and acceptance

1. Whole-theme native CSS inlining, local-only MU experiment: Home LCP3.69s, CLS0.0083, same content/script graph. This is useful causal evidence that stylesheet delivery matters. Its160KB inline allowance adds too much repeated HTML for an automatic production default; it is removed and preserved as an experiment artifact. Possible overlapping integration CPU means it is not final performance certification.
2. Implement real HTTP text compression in an isolated local Nginx gateway with production-suitable delivery configuration. Preserve decoded bytes, MIME/status and private/non-cacheable commerce behavior. Match baseline and candidate through the same gateway and exact-origin verification proxy. Do not modify Lighthouse scoring/proxy to manufacture performance gains. Local delivery proof is not deployment proof.
3. Opt small exact-source theme styles into WordPress Core inlining within its bounded default budget; preserve original handles/cascade, relative-URL normalization, stylesheet fallback, allrules/noJS/nativepluginstyles and jQuery ordering. Measure compared to compressed baseline before deciding whether this belongs in the final candidate.
4. If still necessary, investigate source-derived route bundles, critical-image priority and glyph-faithful responsive font delivery with full visual/commerce review. Do not implement all options speculatively.

Native mechanism inspected against installed WordPress7.1 and official source: https://developer.wordpress.org/reference/functions/wp_maybe_inline_styles/ . LCP interpretation: https://web.dev/articles/optimize-lcp .

Acceptance: mobile LCP≤2500ms across representative Home/PDP/Shop repeated runs under the declared delivery profile, desktop performance preserved, CLS≤0.1, no nativecommerce/a11y/visual regressions, protected402asset hashes +exactapprovedscene manifest unchanged. Shipping and field performance remain separate gates. Main synchronization, PR CI green and founder creative acceptance never substitute for those measurements.

## Follow-up observations (target remains open)

The matched compressed baseline is Home4.011s, PDP2.483s, Shop3.190s mobile LCP. Bounded Core style inlining records3.627/2.330/3.049s; the PDP performance score falls96→88 because Speed Index varies despite its improved LCP. This result remains visible and requires final repetition. Raising whole-theme inlining to160KB under the same gateway records Home3.697s, no improvement over the bounded option; that temporary MU experiment was removed.

First native archive card frame now shares the front image's existing high-priority boundary. Its independent review and native index tests pass. Shop measures3.030s versus3.049s for the preceding bounded-style candidate, too small a difference to claim a material gain; target remains open.

One same-frame AV1 candidate was rejected:812,364B versus approved VP9676,790B, first keyframe126,924B versus71,468B. Full192frame SSIM0.996018 does not override the20% size regression. No candidate was wired or approved originals overwritten. Artifact receipt: `.artifacts/v2-completion-20260906/hero-av1-candidate/encode.json`.

The local PHP origin ignores `Range: bytes=0-1023` and returns the entire676,790B video as200. Static Nginx delivery is being implemented with a read-only public-asset boundary and verified206/416/HEAD responses; native dynamic commerce remains proxied and uncached. Historical compressed proxy-only evidence remains distinct from the new delivery configuration.

Font inspection found Inter downloaded during fallback probing for arrows absent from both primary Hanken and Inter. Its exact cmap is now declared through CSS unicode-range while retaining the original font bytes, weight range and fallback stack. Browser comparison includes actual Inter-only glyphs and primary-font failure, not just the happy path. Final outcome and timing evidence pending.


## Archive optimization follow-up

The native font-face correction and responsive frame checkpoint measured Home mobile 2.175s, PDP mobile 2.434s, and Shop mobile 3.183s under the declared delivery profile (`responsive-final-1`). Home and PDP are single-run passes; final repeated acceptance remains open.

Rejected experiments remain recorded: concatenating all seven archive styles measured 3.412s, and adding native Woo layout inlining to that experiment measured 3.259s. The seven-file bundle and temporary layout MU plugin were removed. High-quality AVIF frame candidates were 11.6–18.4% larger than the accepted 384w WebP variants and changed alpha values; none were promoted. Preloading Hanken and Anton with the archive projection measured 3.482s and was removed.

A separately generated archive-only projection removes only positively identified non-archive selector families from the delivered copy of `theme.css`, retaining original source/full minified files, shared controls, dialogs, card rules, media conditions, keyframes and rule order. Its exploratory Shop result is 2.799s. That run may have briefly overlapped the final focused test startup and is not final acceptance evidence. The next bounded experiment keeps font tokens and the projection eligible for Core's unchanged 40,000-byte aggregate inline budget, with the remaining five contiguous theme styles delivered in their original order from one external file. No native jQuery or WooCommerce stylesheet is removed.

## Final delivery design under verification

The five-style companion was superseded by native Core inlining of the exact seven-style archive chain. The final policy permits a100,000-byte aggregate allowance only in a verified head/main-archive context; previous40,000-byte statements above describe earlier candidates. All individual handles and dependencies remain native. Unknown plugin paths, media conditions, source/filter/order drift and unsupported contexts fall back. This avoids custom multi-handle remapping and removes the unused companion output.

Reviewed q80 mobile frames measured Shop2.493s then3.065s, so the isolated first PASS did not close the target. Artifact-only native100KB inline trials measured2.580/2.346/2.653s. Those experiments are retained separately from final production-code verification. A reviewed360w derivative now supplements384w and640w sources, reducing the358px slot delivery by another8.69–12.09%; this byte saving does not itself prove LCP acceptance.
