# Phase 3B performance diagnosis — Phase 3A evidence and source

Status: diagnosis complete; recommendations have not been implemented by this reviewer. No browser session, build, runtime mutation, media mutation, deployment, or commit was performed for this diagnosis. Root owns fresh profiling.

## Evidence boundary

Authorized runtime: `75ce80b90380d0e5915be79dab33484eda6d3fca`; checked-out evidence descendant: `06f68bd11a08aeab608b6519f2f80a5709476f9b`. Sources inspected were clean at diagnosis start. Historical reports are `.artifacts/v2-phase3-20260905/lighthouse-{home-mobile,pdp-mobile,home-desktop}.report.json`, generated 2026-09-05 17:04 UTC with Lighthouse 13.4.1. Their asset URL hashes bind the sampled resources; use fresh root measurements to certify the exact 3B starting tree rather than treating historical reports as a new run.

Mobile configuration: 390×844, DPR **1**, simulated throttling, 150ms RTT / 1638.4Kbps throughput / 4× CPU slowdown. Desktop: 1440×1000, DPR1, simulated 40ms RTT / 10240Kbps / 1×CPU. All HTTPS requests were blocked; all recorded successful network resources were local HTTP/1.1. These reports therefore do not measure a real mobile DPR2/3 device, deployed compression/CDN behavior, remote gateway performance, or third-party execution.

**Do not mix the simulated score metric with recorded localhost timings.** Home mobile simulated LCP is 12,013.78ms, while the recorded trace LCP is 510ms. PDP equivalents are 12,387.88ms and 733ms. The `lcp-breakdown-insight` gives the recorded trace subparts, not an additive decomposition of the 12-second simulated result. It would be incorrect to claim the 28.5KB home hero took 12 seconds to download or attribute the entire gap to PHP or animation.

## Comparable recorded baseline

| Surface | Performance | Simulated FCP | Simulated LCP | CLS | TBT | Recorded trace LCP | Total transfer |
|---|---:|---:|---:|---:|---:|---:|---:|
| Home mobile | 56 | 5,104.59ms | 12,013.78ms | 0.156684 | 3ms | 510ms | 3,085,618B |
| PDP mobile (SG-005 fixture) | 61 | 5,405.52ms | 12,387.88ms | 0.032301 | 3ms | 733ms | 2,458,665B |
| Home desktop | 84 | 965.75ms | 2,569.64ms | 0.007876 | 0ms | 482ms | 3,521,128B |

| Resource type | Home mobile | PDP mobile | Home desktop |
|---|---:|---:|---:|
| Images | 2,232,475B /19 requests | 1,469,127B /13 requests | 2,667,985B /23 requests |
| Stylesheets | 373,130B /8 | 384,686B /10 | 373,130B /8 |
| Scripts | 178,533B /13 | 293,382B /20 | 178,533B /13 |
| Fonts | 211,835B /5 | 211,835B /5 | 211,835B /5 |
| Document | 84,011B | 94,001B | 84,011B |

Numbers above are Lighthouse transfer sizes (headers included); the five local font files total 211,100B. The user’s 267,962B theme CSS and 62,287B theme JS are package totals, not the bytes of `theme.min.css` / `theme.min.js` individually or the full page including Woo assets. Preserve these distinctions when reporting net growth.

## Exact LCP paths

### Home mobile

- Element: `section.sr-house-hero > div.sr-house-hero__scene > picture > img`.
- Selected request: `assets/sot/images/hero/responsive/black-rose-bay-bridge-monuments-v4-640w.webp`.
- Actual local file: **640×360 WebP, 28,500B**; transfer 28,647B; priority High.
- Markup: `front-page.php:55–60` emits a picture with mobile ≤47.99em / tablet ≤74.99em / desktop fallback. Other verified files: 1024×576 /63,508B and 1440×810 /108,926B. The fallback img declares 1440×810; CSS fills an absolute hero scene with object-fit:cover. At390 the LCP bounding box is395×842 because of scene animation/scale. A landscape640×360 source is aggressively cropped/upscaled into this portrait box; it is already small in bytes, so degrading it further is a weak priority and risks image quality.
- `fetchpriority=high`, `decoding=sync`, no lazy attribute. Lighthouse confirms initial-document discovery, high priority, and eager load.
- `inc/performance.php:206–245` already emits mutually exclusive route preloads matching the picture breakpoints. Do not add a second unconditional desktop preload.
- Recorded request discovery ~351.25ms, request ~352.23ms, complete ~358.61ms after navigation. Recorded LCP breakdown: TTFB334.65ms + load delay19.24ms + load duration7.36ms + render delay148.54ms ≈509.8ms.
- The recorded asset is not waiting for JavaScript insertion or a CSS background discovery. The mobile critical-path recovery should first remove competing bytes and blocking CSS, while retaining the verified hero source.

### PDP mobile

- Element: `.woocommerce-product-gallery__wrapper > .woocommerce-product-gallery__image > a > img.wp-post-image`.
- Request: `/wp-content/uploads/2026/09/sg-005.webp`; High priority, no lazy attribute, `decoding=async`.
- Actual **local synthetic fixture** file is1024×1536 WebP /197,900B; transfer198,048B. Markup width600 height900, data-large_image_width1024/height1536; rendered300×450 at x45,y454. The archived snippet has **no srcset or sizes**. This is concrete for the fixture, not proof that live WordPress attachments lack generated sizes. The fixture seed (`.artifacts/v2-phase3-20260905/seed.php`) explicitly writes attachment metadata with only width/height/file and never generates the `sizes` entries; this establishes the fixture cause of the missing srcset.
- Native media flow: `woocommerce/single-product.php` → `template-parts/commerce/product-hero.php` verified attachment ID filters → Woo product gallery renderer. Preserve the Phase2 resolver and Woo hooks; do not select an editorial/rejected source to solve delivery.
- `inc/performance.php:297–323` preloads the Woo `woocommerce_single` source and includes `imagesrcset`/`imagesizes` when native attachment metadata provides them. Missing fixture derivatives mean no responsive selection is possible there. Current preload sizes use100vw below source width; a future gallery layout needs the same actual slot sizes in both preload and img to prevent redundant fetches.
- Recorded request discovered316.10ms and complete325.52ms. Recorded LCP breakdown: TTFB310.44ms + load delay6.90ms + download9.42ms + render delay406.02ms ≈732.8ms.
- The installed Woo11.1.0 `templates/single-product/product-image.php:54` emits `style="opacity: 0; transition: opacity .25s ease-in-out;"`; its `assets/js/frontend/single-product.js:255/279` reveals the gallery during initialization. This is a verified JavaScript-dependent visibility path despite early image download. The theme invokes that native gallery before the summary. The exact fraction of the recorded406ms render delay due to this fade versus font/layout remains unmeasured. Phase3B should make the initial verified primary image visible without waiting for that presentation enhancement, while preserving native variation-image updates and gallery semantics. Root should measure delayed-JS and normal startup before deciding the narrow compatibility override.

### Home desktop

- LCP is **text**, `h1#sr-house-title`: “Luxury grows from concrete.”, bounding463×347.
- The desktop hero image still downloads as1440×810 WebP /108,926B at High priority, but it is not the measured LCP element.
- Recorded LCP breakdown: TTFB338.38ms + element render delay143.22ms ≈481.60ms. A desktop-image-only optimization cannot directly explain all text-LCP improvement.

## Critical request chain, blocking CSS, and JavaScript

The recorded request graph starts at HTML, branches into eight blocking styles and head jQuery/jQuery Migrate, then discovers all five fonts from design-tokens.css. The longest recorded home dependency chain ends at native Woo `get_refreshed_fragments` (~653ms navigation-to-end); this is not itself the LCP chain and must not be removed merely because it is longest. It maintains the Phase3A bag count/content authority.

Blocking-resource audit estimates3,810ms mobile-home and4,050ms mobile-PDP improvement from the combined opportunity. These are modeled opportunities, **not sums of independent guaranteed savings**:

| Blocking resource | Home transfer | Reported modeled resource duration |
|---|---:|---:|
| theme.min.css | 204,507B | 4,052ms |
| Woo woocommerce.css | 92,285B | 3,002ms |
| jQuery | 87,712B | 1,952ms |
| Woo layout.css | 20,314B | 752ms |
| Global shell CSS | 15,963B | 602ms |
| jQuery Migrate | 13,736B | 452ms |
| Mascot CSS | 13,401B | 602ms |
| Controls CSS | 12,615B | 602ms |
| Woo smallscreen.css | 8,814B | 602ms |
| Tokens CSS | 5,231B | 452ms |

`theme.min.css` has204,346B resource size against204,507B transfer: this local server delivered effectively uncompressed CSS. The same relationship holds for jQuery. Compression is a real delivery opportunity, but changing test-only compression to improve a score would not establish theme-source improvement or deployed-host behavior. Record source bytes and actual wire encoding separately; assess host compression when deployment is separately authorized.

Lighthouse estimates unused `theme.min.css` bytes at164,455B (80.48%) on home mobile and182,673B (89.39%) on PDP mobile. This supports **route-aware extraction and replacement**, not blindly deleting “unused” selectors: unopened bags, forms, account endpoints, hover/focus, variation states, and alternate viewport rules were not all exercised by the audit. Woo common CSS is99.37% unused on the sampled home viewport but the native mini-cart needs it when opened. Do not globally dequeue Woo CSS until equivalent native mini-cart/checkout/account compatibility is demonstrated.

`functions.php:336–344` enqueues theme/controls/shell CSS and theme + house-of-roses-motion scripts globally. Collection scene CSS/JS and home Kids reveal are already conditional. Theme-owned scripts already receive `defer` through `inc/performance.php:360–380`; head jQuery remains blocking. It is a preserved Woo dependency, not a disposable visual library.

Mobile TBT is only3ms on both samples. Home main-thread breakdown: style/layout459.7ms; script evaluation158.18ms; script parse21.32ms; rendering82.35ms. PDP: style/layout313.50ms; script evaluation149.98ms; parse32.34ms; rendering48.66ms. Therefore “too much JS CPU causes12-second LCP” is not supported. Script **transfer/order**, large styles, layout churn, and eager media are better-supported targets.

PDP also requests native zoom2673B, FlexSlider21591B, PhotoSwipe31626B +UI9749B, and single-product13277B (resource sizes). A simpler native-compatible editorial gallery could eliminate unnecessary presentation features only after testing variation image changes, zoom/no-zoom requirements, keyboard access, and no-JS behavior. Retain add-to-cart-variation, wp-util CSP repairs, jQuery dependency order, and Woo event bridge.

## Media competition and card root cause

`template-parts/commerce/product-card.php:69–70` uses **loop position instead of page visibility**: every local loop’s first4 cards are eager and each index0 is high priority. The index defaults0 when a caller does not supply it. This incorrectly prioritizes first cards in below-fold homepage chapters and PDP related products. High priority belongs to the actual page LCP/above-fold media, not every section’s first card.

Home mobile fetches five approved card-front files totaling1,146,012B (exact per-file inventory below) plus four statue frames totaling501,228B during initial navigation, although sampled product image boxes begin around y1365/2259/3148 or farther below the viewport. The home hero itself is28,500B. First SG-005 and first Kids card both receive High priority alongside the hero. PDP similarly downloads four related product images (sg-015,sg-014,sg-007,sg-011), including two High-priority requests, before these cards become relevant.

Approved-card/fallback image branches emit a single src without srcset/sizes. Native Woo attachment branches use `wp_get_attachment_image`, which can emit responsive attributes only if attachment metadata exists. Preserve these two authority paths rather than inventing one generic image substitution rule.

All four hidden navigation preview images are requested in each report despite `loading=lazy`: Signature70,526B +Black Rose63,508B +Love Hurts108,290B +Kids66,610B =**308,934B resource bytes**. CSS visibility/opacity alone does not reliably keep native lazy images outside the preload distance. A shell compatibility improvement can keep the closed nav actually nonrendered while preserving static/no-JS navigation and set preview sources only at activation if that is the chosen enhancement contract. Do not remove the approved previews that established brand recognition. Confirm initial network absence and instant accessible opening on slow connections.

## Fonts and layout instability

All five registered fonts load on every sampled page: Archivo90,096B, Hanken34,664B, Anton12,004B, Cinzel25,904B, Inter48,432B. Existing `@font-face` uses Archivo optional and other faces swap. The font-display audit passes; this does **not** imply stable metrics.

The home mobile CLS audit identifies exactly two shifts in `.sr-house-hero__copy`:0.1192459 attributed to Archivo/Hanken/Anton/Cinzel font loads and0.0374377 attributed to Inter. Total0.15668365. PDP’s0.03230137 shift is attributed to its product container with the first four font loads. `unsized-images` passes. The supported first repair is stable fallback typography/line boxes; blaming missing image dimensions or removing decorative media would not match this evidence.

`theme.css:1076–1087` mixes display Archivo, body Hanken/Inter, utility Anton, and collection Cinzel in the first hero. It uses tightly balanced8ch headline, .86 line height, and a centered/min-height grid; font changes can alter line wrapping and move the whole copy group. Mobile already reserves headline2.58em (`theme.css:1145`), but the recorded shifts show this alone is not sufficient. `theme.js:953` also replaces plain headline text with inline-block word spans after initialization outside reduced motion. The recorded audit attributes shifts to fonts, so JS word wrapping remains a **testable secondary cause**, not established causality.

Recommended: reuse approved type roles but limit above-fold families, remove legacy role usage as obsolete page compositions are replaced, use a metric-compatible fallback face/size-adjust/ascent/descent/line-gap strategy measured against existing fonts, and keep headline markup/line geometry the same before/after JavaScript. Do not preload all five faces: that adds five high-priority competitors. Consider only a proven above-fold face preload after checking whether it improves actual rendering without harming hero transfer or optional-font behavior.

## Prioritized implementation recommendations

1. **Card loading context (high confidence, small dependency surface).** Pass explicit page/placement priority into the canonical card. Default below-fold/editorial/related instances lazy+auto; only actual initial visible card candidates receive eager/high where justified. Preserve real Woo data, CTA, status, and media resolver. Test initial requests on home/PDP and visible PLP cards across widths; do not use UA or Lighthouse detection.
2. **Responsive derived media (high confidence).** Use existing verified responsive derivatives where available. For approved card originals, add deterministic resizes through the established generated-manifest/build pipeline, with source hashes and fixed encoder/toolchain—not replacement renders or edits to originals. Native Woo media uses native metadata/size APIs; fill synthetic fixture derivatives separately and report that environment correction honestly. Do not alter16stale/9missing-front/5rejected/3approved truth states.
3. **Replace/split page styles while delivering each surface (high confidence).** Keep tokens/controls/global shell common; move home, PLP/card, PDP, shared collection styles into route-scoped source modules and remove superseded selectors. Keep registry/minified/package hash reproducibility. Validate all Phase3A overlays and commerce routes after every extraction. Avoid automated purge based on one viewport.
4. **Stabilize homepage first-viewport typography (high confidence for target, measured solution required).** Same geometry without JS, during font load, and with reduced motion; compare layout-shift attribution, not just aggregate score. Do this before layering new arrival choreography.
5. **Prevent closed-nav preview competition (observed issue, shell regression required).** Preserve imagery and shell behavior; ensure preview requests become intentional. Test desktop/mobile keyboard-open, no-JS links, Escape/focus return, fast open/close, and no layout movement.
6. **Scope house/page enhancement JS (medium confidence benefit).** `house-of-roses-motion` is currently global10,092B though many routines early-return on absent selectors. Move specific page behavior without duplicating shell code; preserve all native Woo dependencies. Confirm no-JS/delayed-JS composition. Do not expect CPU reductions alone to resolve12s simulated LCP.
7. **Profile native gallery initialization (unresolved measured delay).** With verified Woo media and realistic attachment sizes, identify what causes the recorded406ms render delay. Keep initial primary image visible and dimensioned before enhancement. Reduce optional carousel/lightbox features only as an intentional gallery design decision, not a blind plugin dequeue.
8. **Delivery encoding/TTFB (environment-aware).** Local root document response329ms/308ms/336ms is measurable but not the dominant supported source defect. Profile backend timing separately if it persists. Record actual compression headers and transport; do not present localhost HTTP/1.1 uncompressed results as production behavior or silently change only the test to hide source weight.

## Safe CSS extraction map

These are candidate ownership boundaries, not permission to delete selectors by line range. Source order and shared selectors matter.

| Destination / loading rule | Candidate selector families | Preserve / verify |
|---|---|---|
| Common base, every route | universal reset; html/body; screen-reader/skip; shared text/button primitives; header/nav/footer; dialog/search/quick-view/size-guide; Phase2 overlay rules | Existing global-shell.css extends baseline `.sr2-header*` behavior; retain underlying hide/open/focus rules. Keep accessible hidden/reduced-motion utilities globally. |
| Canonical card module, any route rendering cards | current `.sr2-c-editorial-card*`, its frame/crop/inscription rules, `.sr2-products` grid and shared product state hooks | Home, Shop, collection, related products, search/pre-order can all render cards. Do not scope solely to is_shop(). Remove older `.sr2-c-product-card*` / `.sr2-c-product-portal*` prototypes only after searching template/function consumers. |
| PDP module, is_product() | `.sr2-product-page*`, crumbs/release/shell, `.sr2-pdp-product*`, portal-specific gallery/summary/details, `.sr2-fit-help`, product-world and sticky-buy rules | Do not move broad `.woocommerce` form/notices/account/cart/checkout selectors with it. Keep canonical cards available for native related products. |
| Home module, is_front_page() | `.sr-house-hero*`, filmstrip, homepage chapter wrappers, `.sr-home*`, `.sr-kids-procession*` | `.sr-home__button` and other legacy primitives may have non-home consumers; first extract/alias shared primitives. Move corresponding keyframes and responsive/reduced-motion rules together. |
| Shared collection module, exact collection templates | `.sr2-collection*`, collection-specific story/hero/scene wrappers | Existing scene styles already conditional. Preserve collection color contexts in tokens and shared cards. Split only after Signature proves the structure. |
| Protected commerce/shared content | cart/checkout/account selectors, generic content forms, native Woo notices | Leave stable for Phase3B; do not delete based on home coverage or rename Woo hooks/classes. |

The final component-order responsive contract at `theme.css:1810` deliberately overrides earlier portal rules. Reproduce the resulting order when moving source groups; avoid splitting base rules into one file and late mobile overrides into another route absent on some card consumers. Register every new generated `.min.css` in the source-certification/build/package contracts and remove replaced outputs only when no consumer remains.

## Certification and authority guardrails

- Keep identical viewport, DPR1, Lighthouse13.4.1, simulated throttling, HTTPS blocking, clean-cache condition, runtime, DB contents, and serial measurement conditions for direct comparison. Add real DPR2/3 checks separately. Use multiple runs and report median/range; retain raw values and requests.
- A fresh Phase3B proof should include Home mobile/desktop, Shop mobile, PDP mobile/desktop plus actual visual snapshots with reduced motion and JavaScript delayed. Do not collapse static identity quality into Lighthouse score.
- For each page record initial above-fold/eager/lazy inventory, selected source dimensions/format/bytes, CSS/JS/fonts, high-priority requests, LCP element, CLS attribution, and interactions after load.
- Local fixture attached images are not staging source authority. Fixture derivative generation must not overwrite catalog, approval files, or source assets. Reconcile any report improvement due solely to fixture realism separately from theme implementation gains.
- Preserve route preload deduplication and responsive alignment, attachment metadata authority, phase2 media resolver, native variation/CSP templates, cart fragments, exact bag product/variation/quantity/subtotal, native checkout/account/search behavior, and source certification hashes.
- No evidence here authorizes production/staging promotion, payment, inventory/fulfillment changes, new imagery, or new scene technology.

## Initial-navigation image inventory

Resource bytes below are from actual recorded requests; all are WebP except the listed inline SVG. These tables include native lazy images fetched inside Chromium's threshold and hidden-nav images. “Lazy” does not mean “not downloaded.” Loading attributes are established above from source and audit element snippets; do not infer eager solely from request presence.

### home-mobile

| Image filename | Resource bytes | Priority | Start ms | End ms |
|---|---:|---|---:|---:|
| black-rose-bay-bridge-monuments-v4-640w.webp | 28,500 | High | 351.25 | 358.61 |
| skyyrose-logo-still-384w.webp | 5,320 | Medium | 352.91 | 361.86 |
| signature-sg-005-512w.webp | 37,318 | Medium | 353.14 | 362.56 |
| signature-portal-statue-640w.webp | 122,698 | Medium | 353.22 | 368.50 |
| sg-005-onmodel.webp | 197,900 | High | 353.28 | 361.00 |
| black-rose-portal-statue-640w.webp | 123,928 | Low | 353.33 | 388.67 |
| br-004-onmodel.webp | 215,006 | Low | 353.37 | 390.49 |
| love-hurts-portal-statue-640w.webp | 137,998 | Low | 353.41 | 391.40 |
| lh-004-onmodel.webp | 290,126 | Low | 353.45 | 393.38 |
| kids-capsule-portal-statue-640w.webp | 116,604 | Low | 353.49 | 392.25 |
| kids-001-onmodel.webp | 194,818 | High | 353.54 | 362.16 |
| kids-002-onmodel.webp | 248,162 | Low | 353.60 | 394.04 |
| jersey-series-town-line-train-v1.webp | 138,992 | Low | 353.64 | 395.55 |
| skyy-canonical-v2-512w.webp | 19,758 | Low | 353.68 | 394.92 |
| signature-golden-gate-monuments-v2-1024w.webp | 70,526 | Low | 486.72 | 491.04 |
| black-rose-bay-bridge-monuments-v4-1024w.webp | 63,508 | Low | 486.81 | 491.22 |
| love-hurts-rose-aisle-monuments-v3-1024w.webp | 108,290 | Low | 486.88 | 492.08 |
| kids-capsule-heir-throne-v3-1024w.webp | 66,610 | Low | 486.93 | 492.84 |
| black-rose-br-004-512w.webp | 43,610 | Low | 486.97 | 496.00 |

### pdp-mobile

| Image filename | Resource bytes | Priority | Start ms | End ms |
|---|---:|---|---:|---:|
| sg-005.webp | 197,900 | High | 316.10 | 325.52 |
| skyyrose-logo-still-384w.webp | 5,320 | Medium | 317.44 | 328.59 |
| signature-portal-statue-640w.webp | 122,698 | Medium | 317.51 | 328.70 |
| sg-015-onmodel.webp | 228,692 | High | 317.56 | 326.94 |
| sg-014-onmodel.webp | 238,928 | High | 317.63 | 329.37 |
| sg-007-onmodel.webp | 146,130 | Low | 317.69 | 345.04 |
| sg-011-onmodel.webp | 166,950 | Low | 317.75 | 345.89 |
| skyy-canonical-v2-512w.webp | 19,758 | Low | 317.79 | 345.55 |
| Inline Woo SVG (no network bytes) | 239 | Low | 362.31 | 362.36 |
| signature-golden-gate-monuments-v2-1024w.webp | 70,526 | Low | 408.22 | 412.07 |
| black-rose-bay-bridge-monuments-v4-1024w.webp | 63,508 | Low | 408.31 | 412.28 |
| love-hurts-rose-aisle-monuments-v3-1024w.webp | 108,290 | Low | 408.36 | 412.70 |
| kids-capsule-heir-throne-v3-1024w.webp | 66,610 | Low | 408.41 | 413.47 |
| sr-monogram-rose-gold.webp | 31,900 | Low | 408.46 | 417.35 |

### home-desktop

| Image filename | Resource bytes | Priority | Start ms | End ms |
|---|---:|---|---:|---:|
| black-rose-bay-bridge-monuments-v4-1440w.webp | 108,926 | High | 341.91 | 347.48 |
| skyyrose-logo-still-384w.webp | 5,320 | Medium | 342.78 | 348.95 |
| signature-sg-005-512w.webp | 37,318 | Medium | 342.86 | 349.18 |
| signature-portal-statue-640w.webp | 122,698 | Medium | 342.90 | 353.31 |
| sg-005-onmodel.webp | 197,900 | High | 342.95 | 348.78 |
| black-rose-portal-statue-640w.webp | 123,928 | Low | 342.99 | 364.34 |
| br-004-onmodel.webp | 215,006 | Low | 343.04 | 364.91 |
| love-hurts-portal-statue-640w.webp | 137,998 | Low | 343.08 | 365.79 |
| lh-004-onmodel.webp | 290,126 | Low | 343.12 | 366.37 |
| kids-capsule-portal-statue-640w.webp | 116,604 | Low | 343.16 | 366.08 |
| kids-001-onmodel.webp | 194,818 | High | 343.20 | 348.85 |
| kids-002-onmodel.webp | 248,162 | Low | 343.27 | 367.30 |
| jersey-series-town-line-train-v1.webp | 138,992 | Low | 343.31 | 367.38 |
| skyy-canonical-v2-512w.webp | 19,758 | Low | 343.36 | 367.24 |
| signature-golden-gate-monuments-v2-1024w.webp | 70,526 | Low | 434.97 | 439.45 |
| black-rose-bay-bridge-monuments-v4-1024w.webp | 63,508 | Low | 435.06 | 440.45 |
| love-hurts-rose-aisle-monuments-v3-1024w.webp | 108,290 | Low | 435.20 | 442.99 |
| kids-capsule-heir-throne-v3-1024w.webp | 66,610 | Low | 435.31 | 444.33 |
| black-rose-br-004-512w.webp | 43,610 | Low | 435.37 | 445.60 |
| love-hurts-lh-004-512w.webp | 63,418 | Low | 435.43 | 445.80 |
| scene-kids-capsule-playroom.webp | 102,036 | Low | 435.50 | 449.83 |
| kids-capsule-mascot-red-guard.webp | 93,008 | Low | 435.57 | 450.85 |
| kids-capsule-mascot-purple-guard.webp | 96,032 | Low | 435.61 | 450.62 |

## Appendix — parser-blocking scripts and dependency boundaries

Read-only follow-up on 2026-09-05 during Phase 3B Shop/PDP work. Evidence: actual HTML fetched from local port 18303 for `/`, `/product/sg-005/`, and `/my-account/`; installed WordPress/WooCommerce source; Phase 2 reports. This is not staging optimizer certification or a new benchmark. No runtime, plugin settings, or build files were changed for this investigation.

### Actual emitted scripts

| Position | Observed handles | Strategy and implication |
|---|---|---|
| Head, all three routes | `jquery-core`, then `jquery-migrate` | Both blocking; holding their requests prevents the parser reaching `main`. Versions 3.7.1 and 3.4.1 respectively. |
| Head, common Woo | `wc-jquery-blockui`, `wc-add-to-cart`, `wc-js-cookie`, `woocommerce`, `wc-cart-fragments` | Already `defer`, with `data-wp-strategy="defer"`; not parser-blocking. |
| Head, PDP additions | `wc-zoom`, `wc-flexslider`, `wc-photoswipe`, `wc-photoswipe-ui-default`, `wc-single-product` | Already native-deferred. |
| Head, account additions | `selectWoo`, `wc-account-i18n` | Already native-deferred. |
| Footer, theme | `skyyrose2-theme`, `skyyrose2-house-of-roses`, `skyyrose2-mascot-loader`; home also `skyyrose2-kids-capsule-reveal` | Already deferred; no new blanket theme deferral saving here. |
| Footer, attribution | `sourcebuster-js`, `wc-order-attribution` | Blocking after `main`; can delay document completion, but do not explain parser stall before `main`. Preserve attribution behavior. |
| Footer, PDP support | `underscore`, `wp-util`, then inline `wp-util-js-after` | Blocking, with a required synchronous callback after `wp-util`. |
| Footer, PDP consumer | `wc-add-to-cart-variation` | Already native-deferred after localization/gallery-default assignment. |

Inspected head `*-js-extra` blocks assign configuration variables; they do not invoke jQuery in this sample. The material inline callback is the theme's PDP `wp-util` after script. This three-route local sample does not establish that other states/plugins/staging optimizers have no additional inline consumers.

### Preserved dependency contracts

`inc/performance.php:362–376` deliberately defers only five theme enhancement handles. `scripts/test-performance.php:114–136` protects that scope and prohibits theme strategy changes to `jquery`, `jquery-core`, `jquery-migrate`, `jquery-blockui`, and `wc-add-to-cart`. The actual installed BlockUI handle emitted here is `wc-jquery-blockui`; future emitted-HTML assertions should cover that handle too, while preserving the current contract.

`tasks/v2-phase2-20260905/gate-2.1.md` documented a real race: a homepage-only theme request to defer core reached optimizer output as deferred jQuery followed by blocking Migrate and a blocking Jetpack `_jb_static` consumer bundle. The repair removed only the theme's homepage core strategy request. WordPress eligibility inspection had predicted blocking core, but the optimizer emitted defer anyway. Local-only WordPress strategy checks therefore cannot certify staging optimizer behavior. The gate subsequently checked fourteen canonical/reload route cases and native cart behavior after cache correction; that ordering remains a preservation requirement.

Exact installed source under `.artifacts/v2-phase3-20260905/wordpress/` corroborates the boundary:

- `wp-includes/script-loader.php:907–909`: `jquery` is an alias depending on core plus Migrate; Migrate itself has no direct core dependency in its registration. An external optimizer must not treat it as a safe blocking consumer of deferred core.
- `wp-includes/class-wp-scripts.php:1098–1136`: concrete handles without intended strategy, handles with inline **after** scripts, and incompatible dependents eliminate delayed strategy eligibility.
- Woo `includes/class-wc-frontend-scripts.php:148,163`: registration/enqueue defaults to `defer`. Around line 240, variation depends on `jquery`, `wp-util`, and `wc-jquery-blockui`; around line 235, add-to-cart depends on jQuery and BlockUI. Preserve native dependency resolution.

Observed PDP order is `underscore` → `wp-util` → theme inline cache initializer → Woo variation configuration/gallery-default assignment → deferred variation consumer. `inc/performance.php:385–415` reads `window.wp?.template?.cache` and populates two native variation templates through fixed allowed-field interpolation. A blanket tag filter that defers `wp-util` while leaving its after callback immediate makes the cache unavailable; the callback returns, and Woo can fall back to Underscore dynamic compilation, violating the current CSP and breaking variation selection. Do not weaken CSP, insert timing retries, or replace Woo variation state to hide this ordering defect. The observed Woo `wc-add-to-cart-variation-js-before` block only assigns serialized native gallery defaults; it is not a jQuery invocation.

### Supported optimization scope

There is **no demonstrated safe new broad deferral change** in this evidence. The only external head parser blockers found are the jQuery pair protected by Phase 2. Theme enhancements and most native Woo scripts already defer. Holding every script measures a stalled parser, not progressive enhancement.

Narrow supported seams: route-scope theme modules whose markup is actually retired, reduce their payload without touching native commerce dependencies, split page-specific CSS, keep the primary PDP image visible before gallery initialization, and deliver responsive derivatives from accepted source media. Prioritize the earlier measured image/CSS/font critical path. Speculative jQuery preloading can compete with LCP when the request is already head-discovered. Moving all scripts to the footer, setting `async`, or blanket `script_loader_tag` rewrites are unsupported. Attribution is native Woo behavior and is not an authorized casual dequeue target.

Any future jQuery strategy experiment needs a separate controlled change with the exact staging optimizer/dependency inventory, canonical and cached output, deliberately slow core versus fast Migrate/consumer requests, emitted order inspection, and native commerce under the existing CSP. Local success alone cannot resolve the previously demonstrated optimizer disagreement.

### Static harness interpretation

The current ignored `.artifacts/v2-phase3b-20260905/static-surface.cjs` holds only script requests under `/themes/skyyrose-flagship-2/assets/js/` in `delayed` mode. It navigates with `waitUntil: 'commit'`, waits for `main h1`, fonts and visible images, captures the initial state, then releases theme requests and observes the loaded result. Separate `nojs` mode disables all JavaScript. Report these distinctly as **theme enhancement delayed** and **JavaScript disabled**. This does not prove native Woo operates while all native scripts are withheld. The harness aborts requests outside the local origin, so third-party behavior is outside its coverage.

No-JavaScript PDP evidence must show the primary image and product information without waiting for native gallery initialization. Holding every native script while waiting for `main` deadlocks behind blocking jQuery and produces no useful static-content verdict. Screenshot creation alone does not establish usable static content or stable interaction after enhancement release.

### Evidence required for any subsequent script-loading change

1. Keep existing performance strategy, CSP variation-template, and overlay regression contracts passing without weakening assertions.
2. Assert actual HTML ordering: one core before Migrate/consumers; blocking pair preserved; native Woo strategies retained; `wp-util` before its inline cache initializer and variation consumer. Check actual handles and duplicate payloads.
3. Capture theme-delay and no-JavaScript separately at mobile/desktop sizes. Check product links, GET filters/search, primary PDP content, and exact script URLs held.
4. Exercise valid/unavailable/reset variation states, actual variation IDs/prices/availability, native add-to-cart, quantity/subtotal/removal and announcements, account forms, and protected checkout without placing orders.
5. After releasing enhancements, check duplicate handlers/cart submissions, console/CSP errors, focus return, and layout shifts. Catch-and-ignore error policies do not constitute verification.
6. Compare serial repeated measurements with identical throttling, fixture, media authority, and network policy. Report all primary metrics and verify exact served build assets/package reproducibility.

This appendix is source/HTML diagnosis only and adds no new functional-test or performance-pass claim.
