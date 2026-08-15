# SkyyRose Flagship V2 remodel — design-system census

**Mode:** discovery only; source audit and adoption plan. This is not visual approval, a release decision, or authorization to deploy.

**Historical candidate identity.** The discovery census examined `codex/skyyrose-flagship-v2-prototype` at baseline `0074fc6e6b161266f9181303296941cd474208af`, seeded at `wordpress-theme/skyyrose-flagship-2`. Its “untracked seed” language is retained as historical evidence only.

**Active candidate identity.** The tracked V2 candidate is `refs/heads/codex/skyyrose-v2-marketplace` at source snapshot `a620435d646d89d3f99840350d3151a25dcbec47`, with payload/attestation and generated-asset parity recorded in `.fashion-theme/promotion-manifest.json`. Source/package gates pass; rights, WooCommerce runtime, fresh browser evidence, independent QA, and founder authorization remain release blockers.

## Decision

Adopt **V1’s canon and generated-SOT architecture**, not V2’s independent `--sr2-*` token branch. Retain and elevate V2's Scroll World, cinematic video, scenery, collection lockups, and native rails as a bounded immersion layer. They are allowed only when commerce remains a direct, static, keyboard-operable control and all animation has a static/reduced-data/reduced-motion fallback.

V2 has strong raw experience material and the active candidate has now passed the source/package handoff gates. It remains **BLOCKED for promotion** until the source-of-truth bridge, page/state inventory, asset registration, rights, runtime commerce, independent QA, and hard-fail removals below are evidenced. This audit makes no rendered or logo-off verdict.

## Canonical source-of-truth map

| Concern | Authority to retain | V2 adoption rule |
| --- | --- | --- |
| Global palette and editor settings | `wordpress-theme/skyyrose-flagship/theme.json` | Keep its named palette/layout sizes; V2 may consume aliases, never define competing brand facts. |
| Typography roles | `data/brand/typography.json` | Archivo display, Hanken Grotesk body, Anton utility, Cinzel caps; scripts are collection lockup assets, not page-heading typography. |
| Collection palette, fonts, lockup, story and hero slot | `data/collections/<slug>/identity.json` | Consume generated collection blocks and the collection's one accent. Do not restate collection maps in PHP/CSS. |
| Generated CSS collection regions | V1 `data/gen-design-tokens.py` → V1 `assets/css/design-tokens.css` | Port/adapt the generator to V2's namespace only after it reads the same sources. It must output an explicitly delimited generated region. |
| Collection asset provenance | V1 `data/build-collection-sot.py`, per-collection `sot.json`, `data/visual-manifest.json`, `data/sot-images.json` | Copying files under `assets/sot/` is not proof. Build a candidate-bound manifest with source record, SHA-256, rights/status, collection/SKU role, dimensions, and fallback. |
| Product facts, prices, availability, variations | WooCommerce server state | Scenic V2 data may name only non-product editorial media. Every purchasable card/PDP must bind a verified product/variation/media record. |
| Design constraints | `docs/brand/visual-references.md`, theme-team charter | The Five are Kith, Oaklandish, Culture Kings, Fear of God, Palm Angels. Oakland-rooted American streetwear, not a European-maison register. |

V1’s canonical pipeline is explicit: `data/collections/README.md` defines `identity.json` as the hand-authored canon; `data/gen-design-tokens.py` generates the collection token blocks; `data/verify-collection-sot.py` checks staleness. V2 contains zero `identity.json`, `sot.json`, `visual-manifest.json`, or `sot-images.json` files. Its comment claiming sources at `assets/css/design-tokens.css:2-4` is therefore not an executable provenance chain.

## What is worth translating from V1

- The generated collection cascade: a collection declares exactly one accent, a lockup, and its font/data facts once, then emitted CSS and templates consume them.
- The V1 semantic layer: surfaces, readable text/muted text, focus, motion, depth, safe areas, and a named z-index/spacing scale in `assets/css/design-tokens.css`.
- The system-wide states and routes V1 has already separated into purposeful surfaces: `woocommerce/cart/cart-empty.php`, `woocommerce/checkout/form-checkout.php`, `woocommerce/checkout/thankyou.php`, account/cart/checkout CSS, `search.php`, `404.php`, and `template-parts/content/content-none.php`.
- Editorial cadence rather than uniform retail rows: collection film spine/feature-scroll parts and product-card variants are usable references, subject to a fresh SOT/product-truth audit.
- The existing focus-visible/reduced-motion design contract and the V1 source/build convention in which `.min` is a derived artifact.

## V2 inventory: assets to retain, defects to close

### Retain and elevate

1. **Scroll World, but as progressive enhancement.** The existing controller preserves native scroll position and has rail controls; desktop-only pinning is gated by fine pointer, reduced motion, and `min-width:901px` (`assets/js/theme.js:92-160`). Keep the cinematic chapter progression and the native rail fallback (`:149-200`), but remove the wheel `preventDefault()` conversion (`:187-196`): it creates a scroll-jacking risk. Static chapter links and an always-visible “Shop pieces” route must work without JS.
2. **Cinematic video and scenery.** V2 checks reduced motion and data-saver before video playback (`assets/js/theme.js:14-24`, `:248-297`), uses lockup imagery in the collection template (`template-collection.php:25-28`), and has non-WebGL degradation logic in `assets/js/skyy-3d.js`. Preserve these pieces only as scenery/supporting atmosphere; a paused poster or first meaningful static frame is the guaranteed fallback.
3. **Collection lockup axis.** The collection template currently uses an image lockup with an accessible text H1 (`template-collection.php:19-29`). This is the correct hero pattern to extend—verified garment/editorial imagery as the frame, a lockup as focal axis, small utility copy, and an edge-positioned commerce CTA.
4. **Baseline accessibility mechanics.** V2 has a skip link, `:focus-visible` treatment, keyboard navigation controls, Escape-close menu, and a reduced-motion branch. These are assets, not completion evidence; browser and assistive-tech proof remains required.

### Blocking deficits

1. **Parallel, hand-copied token system.** V2’s `--sr2-*` palette, fonts, spacing, timing, and collection overrides are separately authored (`assets/css/design-tokens.css:16-88`). It even changes Black Rose and Kids display to Cinzel (`:70-87`), contradicting the canon’s Archivo display role. Replace the file through a V1-source-driven adapter; no second set of token facts.
2. **Asset provenance is not real SOT binding.** `skyyrose2_sot_asset_uri()` only concatenates a local path (`functions.php:14-22`); the large hard-coded `skyyrose2_collections()` map owns hero/lockup/scenery assignments (`:111-193`). One referenced hero is absent from the V2 asset tree: `branding/hero/forbidden-midnight-1280w.webp` (`functions.php:145,153`). Treat every V2 bundled scene as unverified until the candidate manifest binds it to a V1 SOT record and rights status.
3. **Commerce and service-state coverage is partial.** V2 supplies only `woocommerce/archive-product.php`, `woocommerce/single-product.php`, and `woocommerce/cart/cart.php`; it has no V2 cart-empty, checkout form, thank-you, account, search, 404, no-results, loading, error, or payment-recovery templates. A default WooCommerce fallback fails the system requirement.
4. **Product-card contract is incomplete.** The reusable V2 card is a link/name/price/availability projection (`functions.php:240-280`), not a stateful add-to-cart/variation contract. It needs default, unavailable, selection-required, loading, added, failed, keyboard, and server-rejected states; exact price/availability must come from WooCommerce on every transition.
5. **Generated-asset discipline is unproven.** `functions.php:59-87` serves `.min.css`/`.min.js` unless `SCRIPT_DEBUG` is true, but V2 has no local build/verification source or byte-parity record. Do not edit minified output; establish build → minification → byte/parity gate before implementation.
6. **Hard-fail patterns are present.** The V2 homepage uses a type-rendered, centered hero headline plus paragraph and two CTAs (`front-page.php:51-62`), repeated `liquid-glass` chrome (18 static matches), and fact/credibility chips (“04 Living worlds”, “2020 Oakland founded”, “Limited”) (`:57-62`). These meet the banned generic hero, arbitrary glass, and fake-metrics/credibility-pattern criteria. Remove them; no exception or lower threshold.
7. **Immersion lacks cancellation/commerce priority contract.** Current JS provides no user-level “reduce/stop cinematic mode” control, no `visibilitychange`/route cleanup ownership for the video loop/Scroll World, no performance telemetry, and no guarantee that media failure exposes the associated commerce facts.

## Enforceable V2 work package

### A. Canon bridge and tokens — owner: token foundations

- Add a V2 adapter/generator whose only inputs are V1 `theme.json`, `typography.json`, and collection `identity.json`; document generated boundaries and prohibit direct edits inside them.
- Preserve V1 semantic names or provide one-way aliases (`--color-*` → V2 component aliases). Component CSS consumes semantic tokens only; no raw collection hex or font literals outside generated font faces/token output.
- Emit: color/surface/text/state, spacing/layout, elevation/borders, radius, z-index, safe areas, typography role, duration/ease, breakpoints, and focus tokens. The only collection variable exposed to a component is the current collection’s single accent/lockup/media reference.
- CI/source scans fail on retired font declarations, undeclared hex literals, a second accent on a collection route, manual collection maps, direct product-image paths, or stale generated/minified output.

### B. Components and commerce states — owner: component/commerce

- Specify and build one shared contract for header/nav, collection rail, editorial card, product card, gallery, variation selector, quantity, primary purchase control, notices, cart line, order summary, forms, modal/drawer, and empty/error/loading surfaces.
- Product/purchase controls require: default; hover/focus/active; disabled; selection required; stock unavailable; submitting; success/added; recoverable server failure; and price/stock change. No client projection authorizes purchase.
- Bring cart-empty, checkout/payment/validation, thank-you, account/auth, search/no-results, 404, order/returns, and contact failure into the same token/component family before visual polish.
- Collection browsing must vary cadence (featured/portrait/editorial interruption; no more than two identical product-row widths) while filters remain subordinate and keyboard usable.

### C. Extraordinary but restrained immersion — owner: motion/responsive

Use V2 scenery and Scroll World to create one meaningful narrative beat per route, not an ambient effect stack:

| Capability | Required behavior | Fail-closed fallback |
| --- | --- | --- |
| Scroll World | Native scroll remains authoritative; buttons and URL anchors move between chapters; no wheel/touch cancellation of page scroll. | Native horizontal rail with visible prior/next controls and chapter links. |
| Cinematic video | Poster is first render; video is decorative and never the only carrier of collection/product facts. Pause on reduced motion, save-data, hidden document, errors, and user cancellation. | Poster plus lockup, descriptive copy, and commerce CTA. |
| 3D/mascot | Load only on intentional interaction or idle budget; no purchase obstruction; WebGL/GPU failure is silent. | Static approved sprite/link, or no mascot. |
| Motion | House ease `cubic-bezier(.16,1,.3,1)`; reveals 0.6–1.2s; one showpiece maximum; meaningful per-section choreography. | `prefers-reduced-motion: reduce` renders the final stable state immediately. |

Proposed performance ceilings (verify, do not assume): 390px is first priority; above-fold media <=1.5 MB; header animation <=400 KB; LCP <=2.5 s excluding separately attributed TTFB; INP <=200 ms; CLS <=0.05; no long task >200 ms during an immersive interaction; 60 fps sustained only where a 3D scene is intentionally enabled. Instrument `visibilitychange`, media `error`, `prefers-reduced-motion`, `saveData`, explicit “reduce motion” cancellation, interaction-to-commerce latency, and the LCP element. A budget breach removes/degrades the scene rather than delaying purchase controls.

### D. Responsive, content, accessibility, and governance — owners: motion/responsive, accessibility/content, DesignOps

- Define layout transformations at 390×844, 768×1024, and 1440×900 for each route/state: crop/focal point, section ordering, rail/pin fallback, CTA persistence, header/menu, gallery, filters, cart and checkout.
- Maintain 44×44 minimum actionable targets, visible focus, focus restoration for menu/modal, semantic headings, no image-only critical text, alt text from verified asset purpose, live region only for real state changes, AA computed contrast, 200% zoom/reflow, forced-colors, and keyboard/touch flows.
- Do not use crimson for body text on dark. The V1 canon’s `#DC143C` on `#0A0A0A` needs measured contrast and is restricted to fills/borders/glows.
- Freeze a candidate manifest with baseline SHA, source-set hashes, V1 identity/SOT versions, each media record/right, generated-file hashes, build command/tool version, and capture paths. A stale/missing screenshot, timeout, skipped test, missing browser, or unavailable source is `UNVERIFIED`, never pass.

## Verification matrix before independent visual QA

| Route / state | 390 | 768 | 1440 | Required proof |
| --- | --- | --- | --- | --- |
| Home: poster/video/Save-Data/reduced motion | required | required | required | LCP/media budget, cancellation, no generic hero/glass/metrics |
| Collection: normal/rail fallback/Scroll World | required | required | required | native scroll, keyboard controls, lockup/one accent/SOT scenery |
| Archive: featured/grid/filter/empty/error/loading | required | required | required | cadence, filter focus, true availability |
| PDP: gallery/variation/stock/add-to-cart/error | required | required | required | product ↔ SKU imagery proof, state machine, commerce facts above fold |
| Cart: populated/quantity/remove/coupon/empty/failure | required | required | required | keyboard, focus, server result, recovery path |
| Checkout: validation/payment/pending/failure/success | required | required | required | no motion-gated checkout, payment/ownership trace |
| Account/search/404/returns/contact | required | required | required | designed empty/error and focus/contrast/overflow proof |

Run source scans for token/font/color/motion drift; then fresh browser captures, computed contrast, keyboard/focus/overflow/reduced-motion/forced-colors checks, and WooCommerce journey traces against the frozen candidate. The independent fashion visual QA red team alone runs logo-off recognition and the distinctiveness score; this author cannot approve pixels.

## Evidence and current gaps

**Evidence consulted:** `.wolf/memory.md`; `docs/theme-team-charter.md`; `docs/brand/visual-references.md`; V1 `theme.json`, `data/brand/typography.json`, four collection identities, generated-token scripts/README; V2 ledger, token/theme/mascot CSS, theme/mascot/3D JS, PHP templates, V2 source asset inventory; the supplied V2 gap-closure procedure.

**Static evidence:** V2 has 101 lines of token CSS, 543 lines of theme CSS, 865 lines of mascot CSS, and only three WooCommerce template overrides. Its minified assets exist and are loaded by default, but no V2 local generator/parity verifier was found. Static PHP lint is **UNVERIFIED** because the available shell does not expose a PHP executable. V1’s Python staleness verifier is also **UNVERIFIED** in this environment: its `str | None` syntax failed under the available Python runtime before verification could execute. Neither outcome is a pass.

**Hard-fail scan:** **FAIL** — generic centered/two-CTA type hero, arbitrary repeated glass, type-rendered hero script direction, and fake fact/credibility chips are present in source. **Token drift:** **BLOCKED** — V2 has a parallel manual token system and no generator/provenance binding. **SOT imagery:** **BLOCKED** — no V2 asset manifest and a referenced Black Rose hero is missing. **Accessibility verdict:** **UNVERIFIED** — static hooks exist, but no browser/AT/contrast/overflow evidence exists. **Captures:** none; required capture paths have not been created. **Logo-off verdict:** not run; reserved for independent reviewer.

**Independent approver:** none assigned/run. **Promotion handoff: BLOCKED.** Historical blockers included an untracked candidate snapshot and missing minified parity. Current blockers are rights and SKU/media binding, unresolved WooCommerce runtime states, motion/contrast remediation, route/state coverage, fresh browser/accessibility/commerce/performance evidence, and independent visual approval.
