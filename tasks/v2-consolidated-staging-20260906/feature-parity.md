# Protected feature visual parity — independent review

Status: PASS within the captured desktop/mobile feature scope. Founder visual approval, physical-device certification, performance certification and transaction certification remain separate.

Evidence: `.artifacts/v2-consolidated-staging-20260906/browser/final-results.json`, reconciled 18 route/viewport rows. Desktop Home and Ask Skyy use the fresh `browser/home-revalidated/` captures; other surfaces use `browser/current/`. Root established exact 536-file runtime parity; this review relies on that manifest for original protected media identity and does not recalculate remote hashes.

No network requests or staging/source changes were performed during this review. Six labeled contact sheets were generated from existing screenshots in `browser/visual-review/` for visual inspection.

## Protected surfaces

| Surface | Result | Evidence and scope |
| --- | --- | --- |
| Home animated hero | PASS | 1440/390 captures show the accepted Bay Bridge / Black Rose monument composition, usable house CTA and mascot area; recorded frame callback confirms motion rather than the historic static implementation. |
| Signature animated hero | PASS | Sunset waterfront, Golden Gate setting and two approved golden marks preserved at both widths. |
| Black Rose animated hero | PASS | Monochrome moonlit Bay Bridge and black rose monument remain present. |
| Love Hurts animated hero | PASS | Red aisle, gothic setting and approved rose/star monument remain present. |
| Kids / The Heir animated hero | PASS | Approved child character and throne composition remain present. |
| Nine Scroll World scenes | PASS | Inspected all three Signature, three Black Rose and three Love Hurts captures at both widths; chapter artwork, scene identities and commerce panels remain present. The horizontal desktop captures include neighboring scenes, while mobile uses distinct stacked compositions. |
| Product cards | PASS | Desktop and mobile retain ornate collection-specific frames, approved model imagery, names, prices and Quick View entry. Recorded responsive image srcset/sizes and derivative selections remain present. |
| Native Quick View | PASS | Desktop side-by-side media/form and mobile stacked media/form remain recognizable and usable. Recorded native Woo POST form and variation resolution support architecture parity; this is not the historic card-only preview. |
| Ask Skyy | PASS | Desktop/mobile 3D and chat captures preserve the character face, hair, white/black outfit and proportions, with the current controls and chat composer. Recorded intent gate and events support deferred 3D/walk-in behavior. No chat submission was performed. |
| Shop / representative PDP / Bag / Search | PASS within sampled states | Final route and interaction evidence passes. No claim of every product/content decision is made. |
| Cart | PASS for empty cart | Captured native empty-cart presentation is present. |
| Checkout | ENVIRONMENT-BOUND to empty-cart behavior | Both recorded checkout navigations end at `/cart/` with `Your bag`; a populated checkout form and payment sandbox were not exercised. Do not report payment or full checkout certification. |

## Comparisons and limits

The accepted local `v2-readiness-20260906/matched-states/current-home.png` and `paired-visual-final/shop-quick-view-390-current.png` were inspected as reference captures. The prior staged `v2-staging-release-20260906/browser/home-final/skyy-390-3d.png` was also inspected as historical character evidence, explicitly distinct from local evidence. The revised Home omits the retired tagline as authorized; that copy change is not a protected-system regression. Staged Quick View includes the native product description in addition to the earlier local capture; its media/form structure remains intact.

This is a bounded visual inspection of sampled animation frames, not a synchronized frame-by-frame image-diff certification. Original protected media hashes being unchanged supports source identity but alone cannot establish perceived runtime equivalence. No missing hero, scene, card identity, character identity or native Quick View surface was found in the inspected captures.

The `Town Line` name visible in an existing Black Rose scene is preserved approved scene content, not evidence of beginning the separate Town Line / Pre-Order project.

The separate About/logo report records the WordPress.com Likes/sharing strip and limited initial About layout shifts. Those platform observations remain visible to founder review and are not erased by this protected-feature PASS.

## Full Search GET follow-up

After the B13 timed-run completion gate, independently navigated `/?s=Rose` at 1440px and 390px. Both returned HTTP 200, carried the optimized rotating-mark candidate hook, displayed `Results for “Rose”` with Products/Collections/Pages sections, and exposed 52 product links (not a claim of 52 distinct products). No horizontal overflow or page JavaScript errors were recorded. Evidence: `browser/search-final/results.json` and `search-1440.png` / `search-390.png`. No purchase, cart or chat submission occurred; browser contexts were closed afterward.
