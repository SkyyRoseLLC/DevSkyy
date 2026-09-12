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

## Shop source checkpoint

Reviewed the new `inc/shop-archive.php`, rewritten native archive template, conditional Shop CSS, bootstrap/enqueue integration, canonical-card size argument, and the actual installed WordPress/WooCommerce request and ordering implementations. Review remains read-only; plugin-inactive guards and a dedicated harness are concurrently owned by the implementation agent.

**Resolved HIGH: hierarchical native category URLs lost their query constraint.** The initial `skyyrose2_shop_request_vars()` merged a pretty route's `product_cat` value into the state validator. That validator accepted only exact term slugs, so a legitimate hierarchical value such as `parent/child` became empty and the request filter unset `product_cat`. WooCommerce registers category rewrites with `hierarchical => true`; WordPress normally reduces such a value with `wp_basename()` inside `WP_Query::parse_tax_query()`. The request filter ran earlier and removed the value before native resolution.

The repaired source preserves scalar category routing values unchanged for WordPress to resolve, including hierarchical paths and unknown terms, while clearing malformed containers. Display/control state derives the terminal category slug separately. Independent reinspection confirmed this separation. The implementation agent reports `test-shop-archive.php` PASS; the reviewer inspected its assertions for nested pretty and GET paths, unknown route retention, inactive Woo, hostile dimensions, main/admin/AJAX/REST/secondary isolation, OR-preserving stock composition, taxonomy/product inclusion preservation, URL fields, and no-JavaScript ordering. This reviewer did not rerun the harness or mutate the runtime.

The rest of the inspected query design preserves the native main query: stock filtering runs on `woocommerce_product_query`, checks main frontend product archive/taxonomy scope, and wraps existing meta conditions under AND without replacing an existing OR relationship. The template retains Woo loop, notices, pagination, empty-state, and main-content hooks present in the prior override. Native ordering output remains authoritative and receives an explicit submit button for no-JavaScript use. Filter controls use GET URLs and native hidden-field escaping. Shop-only CSS defines the 1/2/3/4-column grid and overrides native Woo float/item widths; the new image slot string follows those breakpoints.

**Shop source verdict: APPROVE for the inspected settled source.** No remaining actionable critical/high defect was identified. The parent-owned `shop-behavior.json`, inspected by the reviewer, records PASS for filter/refresh/sort/history/empty/pagination/malformed/taxonomy cases with JavaScript enabled and disabled, no runtime errors, and no filter Axe violations. `shop-responsive.json` includes measured one-column 288px cards at a 320px viewport. This approval does not certify unreviewed future changes, pixel quality, final package parity, payment behavior, or deployment.

### Editorial aside follow-up

Reviewed the subsequent `skyyrose2_shop_world_note()` helper, first-result collection lookup, once-per-loop aside insertion before the ninth result, Oakland/Living Archive heading copy, and associated Shop-scoped CSS. **APPROVE for this narrow delta; no new actionable source defect found.** The helper allowlists its slug against existing canonical collections and escapes the existing name, manifesto, and story URL. Its list item contains a labeled aside and is not given Woo's product class. The independent counter does not change the native loop counter, product ordering, pagination totals, or query conditions; the template still renders every native result in sequence. Only one call is possible in this template, so the associated heading ID is unique within this usage. The 320px image rule retains `object-fit: contain` rather than cropping garment pixels. Final visual quality and the running full build remain parent-owned verification gates.
