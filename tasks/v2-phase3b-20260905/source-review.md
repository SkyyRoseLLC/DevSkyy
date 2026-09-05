# Phase 3B independent source review

Date: 2026-09-05

## Product-card review checkpoint

Scope: local staged/unstaged changes against accepted runtime `75ce80b90380d0e5915be79dab33484eda6d3fca`, on `codex/v2-phase3b-flagship-commerce-20260905`. Inspected the canonical product-card PHP, native Woo loop integration, approved-front derivative helper, the modified Quick View click handler with surrounding overlay code, rendition tooling, and relevant grid CSS. No PR was supplied; remote CI/merge readiness is not asserted.

Verdict after card and extraction recheck: **APPROVE for the reviewed scope.** Both the responsive image contract and extraction ordering findings are resolved. No critical/high security, transactional, or JavaScript correctness defect was identified in the inspected modifications.

### Resolved card finding

**MEDIUM — responsive `sizes` does not represent all existing card consumers.** `inc/approved-card-fronts.php` declares half the viewport below `47.99em`, then one third below `74.99em`, regardless of caller. Existing plain `.sr2-products` editorial grids become one column below 640px and two columns below 900px; the Jersey grid likewise becomes one column below 640px. These single-column editorial cards are described as half-column images, allowing the browser to choose an undersized rendition. The canonical card should support caller-specific image sizing and use defaults that match its current grid, with future editorial/feature consumers passing their layout intent. Verify browser `currentSrc` against measured card width and DPR after repair.

Evidence limitation: the initial source-only assessment also identified the Shop grid as one column from the theme rule. The parent measured a 172px actual Shop card at 390px because a remaining native Woo item width of 46% affects final geometry. Therefore undersampling on that specific current Shop viewport is **not established**; it must not be reported as a reproduced defect. This finding concerns the missing caller-specific contract and the existing plain editorial grid, with final viewport selection verification owned by the parent.

Resolution: the canonical partial now accepts an optional caller-owned `sizes` string, uses conservative full-width mobile defaults for editorial and feature cards, and supplies it consistently to both approved fronts and permitted Woo attachment output. Main archive slots remain separate. The new PHP regression covers editorial/feature/default/explicit sizes. Parent-produced `card-browser.json` records PASS at 390/768/1440 with selected sources, measured card widths, zero Axe violations, valid Quick View focus return, and no runtime errors. The reviewer inspected that receipt but did not run the browser. Exact high-DPR source quality remains the parent's image-selection check.

### Source observations

- The new Quick View anchor preserves native navigation with JavaScript absent and preserves modified clicks. The handler prevents default navigation only after the shared dialog coordinator opens successfully. Text payloads use `textContent`; PHP escapes attributes and URLs.
- Native Woo product price, stock markup, permalink, purchasability, and loop add-to-cart rendering are retained. A `try/finally` restores the previous global product after editorial rendering, including an extension exception.
- Approved card-front authority remains separate from opening/editorial approval. Without an approved front, the Phase 2 resolver is used instead of reading raw Woo image assignment directly, preserving the explicit rejection guard.
- Heading tags and card variants are selected from finite allowlists. No new query input or custom transactional state is introduced by this card change.
- Renditions remain derived delivery assets. Detailed Python/source-integrity findings and independent generator checks are recorded separately in `rendition-review.md`.

### Verification evidence and limits

This V2 surface is JavaScript/PHP, with no V2 TypeScript or ESLint command/configuration. TypeScript checking is inapplicable. The parent-owned canonical build/verification log `.artifacts/v2-phase3b-20260905/card-build-verify.log` was inspected: 27 Node regressions, PHP media/checkout/search checks, token checks, verified 33-front derivatives, and five Python tests pass. The reviewer did not run another build or modify runtime source. Browser, pixel, and final package verification remain independent integration gates.

Shop/PDP/collection/home changes have not yet been reviewed by this checkpoint.

## Extraction and closed-navigation follow-up

The changed global nav is `display:none` while closed and `display:grid` after `.is-open`; the existing coordinator applies that class before querying and focusing visible navigation controls. The no-JavaScript `scripting:none` rule explicitly restores grid display. No source-level focus/overlay defect was found in this change; delayed-startup, no-JavaScript, and overlay browser checks remain required.

The PostCSS extraction preserves declaration order inside the new page files and wraps split selector rules in their original conditional ancestors. The matched Home/PDP class families are consumed by the expected PHP routes. However, changing stylesheet order can change the cascade even when declaration text is intact.

**Resolved MEDIUM — page stylesheet order restored a superseded fit-guide font size.** Initially `functions.php` enqueued `legacy-product-page` dependent on `skyyrose2-global-shell`, placing it after the shared shell stylesheet. The extracted file repeats `.sr2-pdp-product__fit-guide { font-size: .64rem; }`, while Phase 3A's shared shell sets that exact selector to `font-size: var(--sr2-type-utility)`. Before extraction, the legacy declaration occurred in the earlier `theme.css`, so the utility token won. A read-only PostCSS comparison found this differing-property collision between exact selectors.

The repaired enqueue order is theme, applicable legacy page, controls, global shell; the legacy page depends on the theme. Source reinspection confirms this ordering. Parent-produced `extraction-runtime.json`, inspected by the reviewer, records the same actual stylesheet order, a 14px computed fit-guide font, initialized native gallery opacity 1, and zero closed-navigation preview requests. The parent also reports the delayed/no-JavaScript startup regression passing. This closes the identified regression; it is not an assertion of exhaustive pixel or computed-style parity across every route.
