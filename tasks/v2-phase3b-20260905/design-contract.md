# SkyyRose V2 Phase 3B — Flagship Experience and Commerce Art Direction

**Mode:** local source implementation and isolated verification only.
**Starting runtime:** `75ce80b90380d0e5915be79dab33484eda6d3fca`.
**Starting evidence:** `06f68bd11a08aeab608b6519f2f80a5709476f9b`.
**Branch:** `codex/v2-phase3b-flagship-commerce-20260905`.
**Specification:** founder attachment `d1afc6fe-6c6a-4ca6-ac93-6236ac9d609c/pasted-text.txt`, read in full.
**Contract status:** first-card implementation contract ready; no rendered acceptance or new media approval.

The lead reports the accepted baseline suite passed, 14 baseline screenshots and five
Lighthouse observations were captured before implementation. At actual 390px, baseline
Shop imagery only begins around 700px and the statue treatment makes the garment small;
baseline PDP notices/gallery consume the initial 844px with price and purchase below.
Immediate composition targets are a compact Shop head/filter/sort band, dominant
readable card photography, and PDP identity/price/size/purchase near the first viewport.
Detailed measurements remain in the lead's evidence; this document does not replace them.

The lead owns baseline reproduction, exclusive browser/performance profiling, builds,
runtime changes and evidence. This contract's author performed a read-only source
census and owns this document only. No runtime files, databases, media or history were
changed by this census. Do not begin visual complexity until the mobile LCP path has
been profiled. Do not substitute an earlier implementation baseline.

## Scope and protected invariants

Authorized: card, Shop/PLP, PDP, collection architecture, all four collection worlds,
editorial merchandising, homepage, responsive composition, performance recovery and
restrained page motion. Checkout may inherit compatible tokens but is not redesigned.

Prohibited: production or whole-package staging deployment, payments, inventory or
fulfillment changes, transactional pre-order changes, custom authentication, replacing
Woo authority, WebGL/Three.js, heavy page-transition routers, persistent smooth-scroll
replacement, final Town Line motion and final Black Rose immersive salon infrastructure.
No new media generation, upload, substitution of rejected work, or approval promotion.

Preserve Phase 2/3A native size and variation IDs, validation, quantity, prices, cart
identity, stock truth, payment/status messaging, account forms, overlay coordination,
search result grouping, media rejection behavior, catalog/SOT and generated registry,
editor/runtime token parity, native URL behavior and reproducible build/package gates.

## Design thesis and non-interchangeable house identity

SkyyRose is an Oakland family-founded fashion house whose products carry its story.
Pages should resemble a deliberately edited archive of garments and places, not a
fashion-template grid punctuated by interchangeable slogans. The static frame must
already work before JavaScript arrives.

Retain five connected recognition devices from Phase 3A, now extended to products:

1. An ordinal archive gutter connects collection markers, product sequences and story
   chapters without presenting arbitrary numbers as inventory or exclusivity claims.
2. Athletic Archivo titles oppose quieter Hanken commerce facts and monospace indices.
   Existing lockup artwork remains artwork; no script-font hero reconstruction.
3. House rose-metal rules connect otherwise varied compositions; each collection owns
   exactly one accent and keeps the same usable purchase language beneath its world.
4. Full, readable garments dominate commerce frames. Local monuments and family material
   supply context in adjacent editorial space, not repeated obstructive decoration.
5. Father/daughter/Oakland provenance appears in existing source-backed language and
   imagery. No invented coordinates, civic symbols, awards, materials or craftsmanship.

Use existing `assets/css/design-tokens.css`, `theme.json` and
`tools/v2-runtime/sync-token-contract.cjs`. New components consume `--sr2-type-*`,
`--sr2-leading-*`, `--sr2-color-*`, `--sr2-surface-*`, `--sr2-space-*`, shared width,
focus, control and motion tokens. Add a semantic role only where reuse justifies it;
no parallel page palette or font inventory. CSS variables that change by collection
must resolve at the collection context, preserving Phase 3A's inheritance repair.

## Source architecture census

All theme paths below are relative to `wordpress-theme/skyyrose-flagship-2/`.

| Surface | Existing source | Preserve / change boundary |
| --- | --- | --- |
| Canonical card | `template-parts/commerce/product-card.php`; `woocommerce/content-product.php`; `functions.php` card helpers | One renderer already serves Woo and editorial loops. Evolve it rather than adding Home/Shop copies. Preserve its scoped global product identity and native loop action. |
| Shop | `woocommerce/archive-product.php` | Native main loop, sorting/notices/pagination hooks exist. Collection links currently lead to collection worlds, not archive filtering. Add explicit filter URL behavior without conflating the two. |
| PDP | `woocommerce/single-product.php`; `template-parts/commerce/product-hero.php` | Native before/summary/after hooks and product-type forms exist. Current house note, fit button and decorative context precede the native title/price; restore purchase-first hierarchy. |
| Media | `inc/approved-card-fronts.php`, `data/approved-card-fronts.json`; separately `functions.php:skyyrose2_product_commerce_media`, schema/variation filters and `data/opening-product-media.json` | Preserve both media contexts: 33 accepted card fronts and the unchanged editorial/PDP state contract. Do not apply one manifest's rejection to an independently accepted card asset. |
| Collections | `template-collection.php`; `functions.php:skyyrose2_collections()` | One data-driven template already exists. Replace its repeated hero/story/pinned-world stack with shared compositional modules; differentiate through controlled variants. |
| Homepage | `front-page.php`; `template-parts/home/kids-capsule-reveal.php` | Current code has hero, rotating duplicated filmstrip, product portals, Heir procession, feature product, Town Line film, confidence row and legacy. New Home consumes the mature shared modules last. |
| Town Line | `functions.php:skyyrose2_render_black_rose_jersey_series()` | Registry-driven jersey order and live Woo lookup are reusable. Current film is explicitly a review candidate/previsualization. Do not promote it. |
| Global shell | `inc/global-shell.php`, `assets/css/global-shell.css`, `assets/css/controls.css`, overlay coordinator in `assets/js/theme.js` | Accepted shell is shared infrastructure. Use its controls/dialogs; do not build competing mobile-filter or gallery overlay managers. |
| Page assets | `assets/css/theme.css` plus page/commerce CSS, existing JS and enqueue/build manifests | Remove obsolete rules after each replacement; conditionally serve page behavior. No large additive global stylesheet by default. |

### Concrete issues to address, not copy forward

- The card currently chooses the separately accepted card front, then raw Woo image,
  then fallback; quick view follows the same card context. Preserve the accepted front
  before considering contextual fallbacks. If no accepted card front resolves, raw Woo
  fallback must pass the preserved PDP rejection guard; quick view must not bypass
  that guard either. A shared media API, if introduced, must take an explicit usage
  context and may not
  collapse these distinct truth contracts or downgrade the 33 accepted fronts.
- Card index alone makes the first four cards eager and the first high priority,
  even in below-fold Home/collection modules. Replace that assumption with explicit
  placement/loading intent so each module does not compete with the actual LCP.
- Repeating statue frames around every card adds a second substantial image per item.
  The manifest separately records approval of collection frames, adult-top crops and
  bold single-line inscriptions. Evolve that treatment so the garment is dominant and
  fully readable; profile the frame payload and keep recognizable archive/nameplate
  behavior rather than assuming the approval can be discarded as generic decoration.
- Home BR-004 and the rejected editorial-opening state require context separation,
  not automatic removal: the Home source hash exactly matches the approved card
  manifest's BR-004 source hash. The accepted derived card has its own hash and approval.
  Root owns pixel/provenance review before deciding Home presentation.
- Love Hurts' configured mobile hero path names a Golden Gate composition while its
  desktop path names the rose aisle. Verify actual pixels/provenance before reusing
  this responsive pair; do not infer a matching crop from filenames.
- Town Line markup declares `founder-review-candidate`; its transcript says the film
  is not product-media approval. Existing inclusion does not authorize broader reuse.
- Collection pages contain pinned/horizontal world attributes, extra effects and
  multiple competing entry CTAs. The replacement needs natural vertical reading,
  immediate shopping access and no mandatory horizontal navigation.

## Source-backed content and media boundaries

| Material | Source support | Permitted use / exact limit |
| --- | --- | --- |
| Oakland origin | Existing Home line “Luxury grows from concrete.” and father/daughter/Town statement; `functions.php` collection copy | Existing provenance can anchor Home. Do not invent geographic facts or turn the fictional journey into a real transit claim. |
| Signature | Existing “The House, Signed” copy, Oakland-origin manifesto and Golden Gate responsive hero references | Foundational geometry and origin story. Verify referenced imagery against current source evidence before reuse; no new scene claims. |
| Black Rose | Existing “Beauty Without Permission” copy, protection/depth manifesto, Bay Bridge two-monument references | Silver, nocturnal contrast and sharp pacing. Preserve separately approved card assets; do not promote rejected editorial scene assets into campaign proof. |
| Love Hurts | Existing “The Beast Speaks” and “Every wound protects a rose” copy; rose-aisle/cathedral references | Controlled disruption around stable commerce. Do not replace its world with unrelated bridge material based on filename assumptions. |
| Kids / The Heir | Existing “The throne is already hers”/next-generation manifesto; `kids-capsule-reveal.php`; live lookup of kids-001 and kids-002 | Inheritance and confident future ownership, never babyish/cartoon UI. Existing guardian/procession imagery remains subject to its real authority state. |
| Legacy | Home's existing “A legacy in bloom” and father/daughter promise; `images/about/skyy-rose-founder-hero.webp` reference | Existing narrative may be edited for brevity without changing its factual meaning. Referenced portrait identity/rights must remain verified; do not call it a father/daughter pair if it only shows one person. |
| Town Line | Registry `presentation=jersey-series`, `series_order`, actual Woo product names; existing fictional Oakland/San Francisco/Bay/San Jose narrative | Static chapter directory with real product links is supported. Film/poster candidate status is not elevated. No advanced transport animation, invented schedule, station or affiliation. |
| General brand narrative | `docs/brand/visual-references.md`, `docs/brand/collection-stories.md`, collection identity files | Narrative references only where traceable. Old story-document typography instructions do not override the accepted Phase 3A font system. |

The user explicitly preserves **16 stale / 9 missing-front / 5 rejected / 3 approved**
editorial-opening/PDP resolver states. Independently, `data/approved-card-fronts.json`
contains **33 accepted card-front records** and the founder's wiring authorization.
Its `card_treatment_approval` records all 33 fronts, collection frames, adult top crops,
and bold single-line inscriptions centered inside their nameplates. These scopes are
not interchangeable. Record unchanged manifest/registry hashes and resolver outcomes.
A stale editorial approval may coexist with valid assigned commerce imagery according
to the accepted PDP resolver; that resolver's explicit rejection cannot be bypassed.
But it must not invalidate a separate approved card asset merely by matching the SKU.
Missing media gets a compact intentional treatment, never an invented image.

Exact BR-004 census evidence: Home path
`assets/sot/images/home/on-model/black-rose-br-004.webp` hashes to
`e6cd53ffeadf828c2b6cbda9b2e3d33f0f255bbeef0510fe8f353ebd879d1feb`, exactly the
`source_sha256` of the approved BR-004-A card record. Its accepted derivative
`assets/approved-card-fronts/br-004-onmodel.webp` hashes to
`8c415f0fe1e5ab113e74f7a7040563b5396f2672f400cf9d30f99db839de32c8`, matching the
manifest's `sha256`, with `scene_status=FOUNDER_APPROVED_V2_CARD`. The separate opening
record is `REJECTED_AUTHENTICITY` with empty views and does not identify that accepted
card derivative as rejected. There is no basis here to call that approved card
unauthorized. Root owns actual-pixel review and any Home reuse decision. The source-hash link
supports provenance; it does not expand the approved card usage into blanket Home
editorial approval. No new visual approval is claimed by this document.

## Canonical components and rendering contracts

### A. Product card — implement first

Use the existing PHP partial with a small explicit argument contract:

- `product`: real visible `WC_Product`; no duplicated price/name/status payload.
- `index`: editorial ordering only, not automatic eager-load permission.
- `variant`: bounded standard, feature or compact presentation using one DOM contract.
- `media_priority`: explicit placement intent; default lazy/auto outside initial view.
- `heading_level`: context-correct h2/h3 without skipping the host section hierarchy.

Render media → collection/index → linked name → native price/status → native action.
Keep name, price and status visible without hover. Show native selection-required action
for variable products; simple-product actions remain Woo-owned. Quick view is optional
secondary access and must use the same accepted card-context image policy. Avoid redundant nested
links or three equal-priority CTAs.

Standard image ratio should favor full readable garments, with contain behavior and
reserved dimensions. Retain a legible archive index/nameplate and evolve the existing
approved frame/crop system so it does not shrink the garment into a tiny inner window.
Feature variant changes footprint/adjacent text rather than cropping away construction. A verified secondary view can be loaded on deliberate
fine-pointer interaction, never eagerly for the entire grid. Keyboard focus receives
an equivalent clear response without requiring hover to discover information. Default
reaction: frame/rule change and small metadata emphasis; no generic zoom dependency.

Required states: available, unavailable, selection required, submitting, added, failure,
media ready, media absent/rejected, keyboard focus, reduced motion and no-JS. Native
validation and notices remain authoritative. No fabricated badges, scarcity or savings.

### B. Shop / PLP

Preserve the main Woo query and hooks. Build a compact archive introduction, indexed
collection/category navigation, native result count and sorting, then a comparison-first
product grid. Category/collection/status filters must use validated native URL state
and existing terms; render only real available dimensions. Default browser GET forms
must work without JS; preserve active parameters through pagination/sort and clear them
explicitly. Refresh and Back/Forward must reconstruct identical filters.

Desktop: 3 columns as default, 4 where widths fit, with a bounded feature or editorial
interruption between product groups. Mobile: 2 columns only where garment/name/control
legibility remains sound; use 1 at very narrow widths rather than squeezing controls.
A feature may span columns without silently changing product order. Filter drawer uses
the shared overlay lifecycle; desktop filters remain subordinate. Show active count,
clear filters, empty-result explanation and a real reset URL. No custom client catalog.

### C. PDP — product first, story after

Desktop: large primary media and calm purchase column; optional verified details form
an editorial sequence below. Mobile: compact primary view, name/price/size/purchase near
the opening viewport; additional imagery and storytelling follow. Do not put a long
manifesto or decorative artifact ahead of the title and form. Preserve the native hook
sequence and scoped media filters with restoration even on missing imagery/errors.

Information order: name, collection, price, stock/variation, quantity/Add to Bag, fit
help, supported material/construction/care details, shipping/returns route, product
story, collection context, related pieces. Populate product facts only from actual Woo
content/catalog authority; omit unsupported sections instead of generating filler.

Keep native gallery behavior if it meets the composition; avoid new carousels by
preference. Primary media gets the correct responsive dimensions and sole justified
priority; remaining media lazy-loads. No sticky purchase bar unless overlap, focus,
virtual keyboard and shared-overlay behavior are verified. Native commerce controls
are not duplicated into a competing JS transaction state.

### D–H. Shared collection system, Signature first

Implement a shared set of PHP modules: collection arrival, story spread, product edit,
editorial interruption, optional related-story link and next-world index. Data comes
from existing collection definitions and live Woo queries. Variants choose composition,
not duplicated template stacks or independent CSS systems.

| World | Composition | Mobile transformation | Signature restraint |
| --- | --- | --- | --- |
| Signature | Precise grid; existing monument image framed beside compact origin copy; aligned product feature and quieter supporting pieces | Identity and shop access accompany a bounded image; origin text follows first products | Strong stable geometry, gold accent, minimal motion. |
| Black Rose | Asymmetric dark/light photographic mass; narrow text measure; one larger garment frame interrupts the cadence | Keep product legibility and silver rules; compress empty space rather than crushing dark detail | No red-on-black gothic wallpaper, roses as decoration or final salon system. |
| Love Hurts | One controlled offset or broken alignment in an editorial block; bold image/text tension; commerce tracks aligned | Reset risky overlap into intentional staggered blocks; all controls fully aligned and reachable | Crimson only; visible disruption cannot become clipping, focus loss or illegible text. |
| Kids / The Heir | Confident future-owner portrait/context plus generous garment frames; warm house rhythm with rose accent | Image, identity, practical sizes, then inheritance story | No cartoon controls, toy styling, invented family facts or automatic rejected-media reuse. |

All worlds share the same card, filter/control, type-role and media contracts. Collection
lockup artwork may establish arrival when already verified. Do not type-render new hero
scripts. Keep one obvious Shop collection destination/anchor and a quieter story link.
Render/refine/review Signature before propagating any new structure to the other worlds.

### I. Homepage — consume mature components last

The eight acts are a narrative sequence, not eight identical full-screen panels.
Reuse shared cards, collection index, story spread and product feature modules.

| Act | Job / source-backed content | Composition contract |
| --- | --- | --- |
| Arrival | SkyyRose, Oakland, Living Archive | Existing house artwork/verified context as a bounded frame; meaningful identity immediately visible; one primary exploration action and direct commerce path. No centered generic video/two-pill hero. |
| Four Worlds | All four collections | A coherent indexed composition with varied feature scale/crop; all four accessible without mandatory horizontal scrolling. |
| Oakland | “Luxury grows from concrete” and existing provenance | Typographic/photographic rhythm break using verified existing content, not added concrete texture or graffiti costume. |
| Product as Artifact | Real purchasable pieces | Shared feature card/spread plus quieter supporting products. Current name, price and status from Woo. |
| The Heir | Existing next-generation narrative | Refined shared Kids story/product module; no duplicate Home-only commerce renderer. |
| Town Line prelude | Fictional jersey chapter sequence | Static registry-ordered product/name directory with archive numbering; candidate film excluded from any approval claim. |
| Legacy | Existing father/daughter/house promise | Intimate reading measure and verified existing portrait if supported; no invented testimonial or provenance. |
| Continue | Collections, Shop, Journal, Pre-Order, Client Services | A deliberate closing directory integrated with the accepted footer, not another generic CTA card. |

Every act must answer identity, origin, making, distinction, exploration or purchase.
Remove any decorative sequence that answers none. Do not autoplay a duplicated product
filmstrip or repeat eagerly loaded cards merely to fill the narrative.

## Responsive, static-first and performance gates

Actual test widths: **320, 360, 375, 390, 414, 768, 1024, 1280, 1440, 1728**.
Use intrinsic/fluid layout; these are verification points, not ten breakpoint patches.
Primary captures at 390 and 1440 include full-page composition and crucial states;
768 demonstrates tablet transformation. Mobile order prioritizes image, identity,
commerce, then story where appropriate. Never hide purchase facts behind motion.

Before visual additions, the lead records mobile LCP element, request chain, responsive
selection, dimensions/format, priority/preload, font/CSS blocking, server/JS/third-party
contribution and layout instability. Report comparable observations for Home mobile,
Home desktop, Shop mobile, PDP mobile and PDP desktop: score, LCP, CLS, transferred
bytes, CSS, JS and fonts. Phase 3A reference observations are 12.01s Home mobile LCP,
12.39s PDP mobile, 2.57s Home desktop; 267,962B minified theme CSS, 62,287B JS and
211,100B observed fonts. They are lab references, not field CWV.

For every flagship route, inventory above-fold/eager/lazy images, largest request,
selected mobile/desktop source, dimensions, transferred bytes and purpose. Use
`srcset`/`sizes` for actual rendered slots; no eager secondary-view grid payload, no
full originals when suitable derivatives exist, no desktop media unnecessarily on mobile.
Record existing-image derivative provenance if responsive resizing is authorized in
implementation; no generative imagery or authority change is implied.

Fix Home's pre-existing hero CLS through reserved geometry and stable typography.
Remove/replace obsolete page selectors as new rules land; track net CSS growth and
page-level JS instead of simply appending global code. Page-specific enhancement is
conditionally enqueued. No new font or animation library is expected.

Capture each redesigned page with reduced motion and delayed/disabled JS. Essential
content starts visible. Allowed motion uses existing tokens, CSS/WAAPI/native observers
only where justified. No scroll-jacking, mandatory horizontal scroll or touch hijack.
Scroll-linked moments must survive reverse/fast scroll, reload mid-page and reduced motion.

**Performance certification hard gate:** do not certify a materially worse mobile
loading path without compelling evidence-backed justification and an explicit remediation
plan. Target substantial improvement from the comparable Phase 3A local baseline;
never trade real meaningful rendering for a Lighthouse score trick.

## Delivery sequence and acceptance matrix

1. Canonical card → build/render/refine and verify media/commerce semantics.
2. Shop/PLP → filter/sort/query/URL/no-JS/empty/mobile review.
3. PDP → media states, native size/purchase hierarchy and performance review.
4. Signature → render, independent collection-system review, refine shared architecture.
5. Propagate shared architecture deliberately to Black Rose, Love Hurts, Kids/The Heir,
   reviewing each world's responsive differences before continuing.
6. Home → assemble mature shared modules; profile/repair hero CLS and mobile LCP.
7. Cross-surface regression, clean build, independent review and final report. No 3C.

For each substantial batch: build, V2 verification, relevant regression tests, generated
drift check, desktop/mobile capture, keyboard, reduced motion, console/network and
independent review as required. Commit coherent reviewed batches; do not rewrite 3A.

| Surface | Required interaction/state evidence |
| --- | --- |
| Card | Available/unavailable/selection-required/loading/failure; focus; missing/rejected image; secondary view if implemented; no-JS. |
| Shop | Sort + filters + category/collection + URL preservation + Back/Forward + refresh + pagination + mobile drawer + no results. |
| PDP | Approved/stale/missing/rejected resolver outcomes; valid/invalid size; quantity; native Add to Bag; related card identity; gallery keyboard. |
| Collections | All four with shared structure, distinct pacing, responsive image choice, clear commerce and next-world links; no horizontal trap. |
| Home | Eight narrative jobs where supported; all four worlds; mature product modules; meaningful pre-JS arrival; stable dimensions; media inventory. |
| Protected surfaces | Account form labels/toggle, search product/story/page grouping, bag state/focus, native checkout shell and Select2 contrast. |

End-to-end: **Shop → filter/sort → product → valid size → Add to Bag → Bag → Cart →
Checkout shell**. Record actual product, variation ID, quantity, price and subtotal.
No real payment/order completion. Test fixture evidence does not certify live inventory,
authenticated account, integrations or payment-provider behavior.

## Independent visual review — exact founder rubric

Review Home, PLP, PDP and the collection system independently. Use this exact weighting:

| Dimension | Points |
| --- | ---: |
| Brand specificity | 20 |
| Composition | 15 |
| Typography | 15 |
| Imagery | 15 |
| Commerce clarity | 15 |
| Responsive quality | 10 |
| Motion restraint | 5 |
| Technical finish | 5 |
| **Total** | **100** |

The target is a material improvement over the accepted 3A 88/100, not an automatic
90+ label. Give criticism and concrete evidence. Retain governance's minimum 85/100,
70% in every category, zero hard failures and zero unverified claims for approval;
a score below or merely equal to 88 does not demonstrate the requested quality uplift.
Do not change the weights to the older global-shell rubric.

Ask the hard question: could another fashion logo replace SkyyRose with equal sense?
If yes, strengthen the source-backed recognition devices. Independent logo-off review
must distinguish the authored page from generic fashion retail. Scores and approval
are UNVERIFIED until the independent reviewer supplies fresh captured evidence.

Hard failures include arbitrary glass, generic centered hero/two-pill structure,
monotonous product grids, equal decorative cards, multiple accents, new/retired fonts,
type-rendered lockups, unverified garments, fake scarcity/metrics, urgency/pressure copy,
uniform reveal stacks, unrelated startup/SaaS composition, cinematic obstruction of
commerce and intentional disorder that becomes actual broken responsive behavior.

Carry forward launch blockers from 3A/2: full-package staging parity, Klaviyo/integration
uncertainty, authenticated account and real payment/provider flows, inventory/shipping
unknowns and existing media approval deficits. This contract authorizes none of those
gates. Stop after Phase 3B delivery; do not begin final signature motion in Phase 3C.
