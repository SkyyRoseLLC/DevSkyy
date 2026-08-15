# V1 authority benchmark for the V2 remodel

**Purpose.** Historical discovery comparison of the existing Flagship (V1) theme and the original `skyyrose-flagship-2` seed. This is a redesign decision input, not visual approval, implementation authorization, or a conversion claim. The active candidate is now the tracked `codex/skyyrose-v2-marketplace` branch at `/Users/theceo/DevSkyy`; see `.fashion-theme/promotion-manifest.json` for current lineage and gates.

## Candidate identity and confidence

| Field | Evidence | Status |
|---|---|---|
| Repository baseline | `0074fc6e6b161266f9181303296941cd474208af` (`chore(sot): add lookbook SOT watch chain and checks`) | HISTORICAL |
| V2 seed | `wordpress-theme/skyyrose-flagship-2/` was untracked in the prototype snapshot | HISTORICAL |
| Active V2 candidate | `codex/skyyrose-v2-marketplace` at `9b4bb5acb1e61f55c8a116d6937a316e16fe9595`; scoped V2 paths are tracked | OBSERVED |
| Immutable candidate manifest / package hash | `.fashion-theme/promotion-manifest.json`; payload/package evidence is recorded and promotion remains fail-closed | OBSERVED / BLOCKED |
| Browser, responsive, screen-reader, payment, order, and independent visual evidence | Not present in the V2 seed | UNVERIFIED |
| SOT authority | [SOT.md](../../../../SOT.md) makes catalog/dossiers product authority, `sot-images.json` the generated product-image contract, and V1 `data/visual-manifest.json` the non-product-image authority. | OBSERVED |

The V2 gap contract requires a single candidate ID on evidence and calls missing interaction, responsive, commerce, rights, and independent-review proof `UNVERIFIED`; static source inspection cannot substitute for those gates ([v2-gap-closure.html](../../../../../plugins/fashion-theme-team/skills/fashion-theme-team/brain/showcase/v2-gap-closure.html)).

## What V1 establishes

| Area | Evidence | Benchmark finding | Status |
|---|---|---|---|
| Editorial authority | V1 composes the homepage as named acts—hero, ticker, rooms, conditional drop, receipts, origin, letter, close—rather than a generic card stack ([front-page.php](../../../../wordpress-theme/skyyrose-flagship/front-page.php)). | A house-level narrative can precede browsing while each act has a distinct job. | OBSERVED |
| World-to-commerce sequence | The V1 rooms module is a CSS scroll-snap collection chooser; the V1 PDP begins with product information, size guidance, stock/pre-order state, and WooCommerce add-to-cart ([index-rooms.php](../../../../wordpress-theme/skyyrose-flagship/template-parts/home/index-rooms.php), [single-product.php](../../../../wordpress-theme/skyyrose-flagship/woocommerce/single-product.php)). | Visual world-building and purchase readiness coexist. | OBSERVED |
| Cinematic restraint | V1 keeps video in hero/origin modules and limits 3D to a specific decorative Kids section ([hero.php](../../../../wordpress-theme/skyyrose-flagship/template-parts/home/hero.php), [origin.php](../../../../wordpress-theme/skyyrose-flagship/template-parts/home/origin.php), [letter.php](../../../../wordpress-theme/skyyrose-flagship/template-parts/home/letter.php)). | Motion is most authoritative as a composed layer around content, not the sole route to it. | OBSERVED |
| Purchase continuity | V1 customizes cart and checkout with cart variations, quantity, coupon/update, totals, checkout handoff, and WooCommerce hooks ([cart.php](../../../../wordpress-theme/skyyrose-flagship/woocommerce/cart/cart.php), [form-checkout.php](../../../../wordpress-theme/skyyrose-flagship/woocommerce/checkout/form-checkout.php)). | V1 has a fuller source-level browse-to-checkout surface than the V2 seed. Payment/order behavior was not exercised. | OBSERVED / UNVERIFIED |
| Product truth | V1 display helpers resolve catalog projections to live WooCommerce products when possible and use catalog/SOT image precedence ([product-catalog-display.php](../../../../wordpress-theme/skyyrose-flagship/inc/product-catalog-display.php)). | V2 must retain SOT-first, WooCommerce-authoritative rendering, not hard-code product facts in its editorial layer. | OBSERVED / RECOMMENDED |

## V2 seed: authority retained and gaps exposed

| Surface | Seed evidence | Assessment | Status |
|---|---|---|---|
| Cinematic home | V2 has a video hero, house navigation, collection rail, Bay-area pre-order scene, WooCommerce-backed product cards, founder/origin content, house index, FAQ, and shop CTAs ([front-page.php](../../../../wordpress-theme/skyyrose-flagship-2/front-page.php)). | Scenic density coexists with shop paths before, within, and after the cinematic material. | OBSERVED |
| Scroll World | The seed has a desktop fine-pointer pinned scroll transform; reduced-motion, non-fine-pointer, and smaller-screen cases use native horizontal rails with controls ([theme.js](../../../../wordpress-theme/skyyrose-flagship-2/assets/js/theme.js), [theme.css](../../../../wordpress-theme/skyyrose-flagship-2/assets/css/theme.css)). | Retain and elevate it as a signature interaction; native rail remains the canonical non-cinematic path. | OBSERVED / RECOMMENDED |
| Collection world | Each collection combines a SOT-path hero, product section, scenic story rail, and return-to-shop link ([template-collection.php](../../../../wordpress-theme/skyyrose-flagship-2/template-collection.php)). | The intended “world then garment” grammar is present. Final media/right binding and product availability relationships are UNVERIFIED. | OBSERVED / UNVERIFIED |
| Product and cart | V2 delegates product decision content to WooCommerce and preserves variation display, quantity, remove, coupon/update, nonce, and cart-collaterals in its cart ([single-product.php](../../../../wordpress-theme/skyyrose-flagship-2/woocommerce/single-product.php), [cart.php](../../../../wordpress-theme/skyyrose-flagship-2/woocommerce/cart/cart.php)). | V2 has no checkout override and no checkout/payment/order test evidence. | OBSERVED / UNVERIFIED |
| Catalog cards | The V2 helper derives card name, price, image, category, and stock label from published WooCommerce products ([functions.php](../../../../wordpress-theme/skyyrose-flagship-2/functions.php)). | This is not the V1 SOT/catalog binding model; it needs explicit SKU → variation → approved media mapping. | OBSERVED / RECOMMENDED |
| Asset bindings | `skyyrose2_sot_asset_uri()` only concatenates a local path. `branding/hero/forbidden-midnight-1280w.webp` and `branding/hero/luxury-nighttime-1280w.webp` are absent from V2 `assets/sot/`, although matching V1 visual-manifest records are `verified` ([functions.php](../../../../wordpress-theme/skyyrose-flagship-2/functions.php), [page.php](../../../../wordpress-theme/skyyrose-flagship-2/page.php), [visual-manifest.json](../../../../wordpress-theme/skyyrose-flagship/data/visual-manifest.json)). | Do not infer identity or rights from a copied filename. Bind each V2 reference to manifest, rights, and candidate records; absent packaged files block release review. | OBSERVED / RECOMMENDED |

## V1 principles to reuse — without copying V1 code or assets

1. **Legible house sequence:** thesis, collection choice, garment discovery, service/terms, invitation. V2 can express this with scenic Scroll World and video rather than V1’s component sequence. **RECOMMENDED.**
2. **Distinct world + real collection route:** use current collection `identity.json` and generated collection SOT; keep narrative, type/accent, and scenic pacing adjacent to purchasable products. **RECOMMENDED.**
3. **Garment truth at decision time:** approved media, identity, price, variation/material, size/fit, availability/fulfilment, return summary, then real WooCommerce action. **RECOMMENDED.**
4. **Service language tempers spectacle:** retain direct fit, shipping/returns, and support paths, rendering actual policy/store facts only. **RECOMMENDED.**
5. **Progressive enhancement:** semantic links and content, static poster/image, native rail, and normal product/cart routes must exist before pinned transform, autoplay, depth, 3D, or hotspots. **RECOMMENDED.**

Do **not** copy V1 templates, CSS, JS, visual assets, hard-coded policy language, or custom checkout implementation by default. They are historical source material, not V2 canon; current compatibility, candidate rights, visual behavior, and conversion outcomes are unproven. **RECOMMENDED / UNVERIFIED.**

## Founder-mandated V2 acceptance criteria

These are non-negotiable scope criteria for a V2 candidate. They are not approved visual outcomes and require candidate-bound evidence before any acceptance decision.

| Criterion | Source-backed path today | Missing SOT / evidence that blocks acceptance | Status |
|---|---|---|---|
| Each collection is its own story-world experience. | `skyyrose2_collections()` declares four collection-specific narrative records; `template-collection.php` renders a hero, product section, scenic story rail, manifesto, and cross-navigation per collection. Scroll World code provides a desktop pinned treatment with native-rail fallback. | Each story asset needs an explicit current identity/manifest/right record, local package resolution, and 390/768/1440 visual evidence. The V2 seed does not contain a candidate manifest or independent review. | OBSERVED / UNVERIFIED |
| Cards and PDPs are collection-specific. | V2 cards obtain product data from WooCommerce; V2 PDP determines a matching collection from product-category terms and renders the associated collection context. | No demonstrated SKU → collection → variation → `sot-images.json` binding, no required high-fidelity product-media sequence, and no proof that every card/PDP rejects a mismatched or unresolved collection/media record. | OBSERVED / UNVERIFIED |
| Cards and PDPs use high-fidelity imagery. | The root SOT identifies `sot-images.json` as the front-first product-imagery contract and V1 visual manifest as non-product authority; V2 contains local scenic asset copies and uses WooCommerce attachment images for cards. | High fidelity is not established by a filename, attachment, or local copy. Require SOT resolution, eyes-on pixel review, approved rights/provenance, proper crop/focal point, and product/variation identity at each displayed frame. | OBSERVED / UNVERIFIED |
| Pre-order items have collection-specific immersive shopping scenes. | V2 supplies a Black Rose pre-order interactive-scene template with linked hotspots/cards and a dedicated pre-order route. | No evidence that every active pre-order item maps to its collection-specific scene, source SKU, variation, availability, rights-cleared imagery, or fallback card/list. The exact availability data was not exercised. | OBSERVED / UNVERIFIED |

Acceptance must fail closed if a collection is reduced to a generic shell, a card/PDP uses unresolved or cross-collection imagery, a pre-order item has no corresponding collection scene and ordinary product route, or the scene blocks touch, keyboard, reduced-motion, no-JS, or checkout access. **RECOMMENDED.**

## Priority V2 redesign decisions

| Priority | Decision | Required safeguard / proof | Status |
|---|---|---|---|
| P0 | Explicitly bind product, variation, media, and rights for every purchasable CTA and scenic shoppable link. | Candidate manifest plus SKU → product → variation → media → rights; fail closed on unresolved media/availability. | RECOMMENDED |
| P0 | Repair or remove the two missing packaged SOT hero paths before visual review. | Exact manifest record, local-file existence, rights freshness, and rendered-route proof on one candidate. | RECOMMENDED |
| P0 | Preserve a full commerce path while V2 is cinematic. | Shop, collection products, PDP facts, cart, and WooCommerce checkout work without video, Scroll World, WebGL, hover, mouse wheel, or JS-only state. Browser/payment evidence remains UNVERIFIED. | RECOMMENDED / UNVERIFIED |
| P1 | Elevate Scroll World as editorial transport, not a purchase gate. | Desktop fine-pointer may pin; touch, keyboard, reduced-motion, save-data, failed-JS, and 390/768 views retain native scroll/controls and visible CTAs. Do not trap vertical progress or hide product facts in transformed panels. | RECOMMENDED |
| P1 | Elevate video as scenic atmosphere, not message delivery. | Approved poster, equivalent text/context, no autoplay dependency for navigation/commerce, pause/save-data/reduced-motion behavior, and LCP/overflow checks. | RECOMMENDED |
| P1 | Move V1 decision-zone discipline into V2 WooCommerce content, not a bespoke visual summary. | Test simple/variable products, stock/pre-order changes, validation/errors, price changes, cart persistence, payment failure, guest/auth ownership, and confirmation. | RECOMMENDED / UNVERIFIED |
| P2 | Make collections recognisable with logos removed. | Test crop family, Oakland/concrete spatial cues, garment-first framing, one controlled accent, type rhythm, narrative labels, and service transparency—not lockups, rose marks, or branded video. | RECOMMENDED |

## Cinema, accessibility, and commerce contract

The founder direction is to retain and elevate Scroll World, video, and scenic immersion. The contract below prevents cinema becoming a gate.

| Layer | May enhance | Must never control | Required fallback |
|---|---|---|---|
| Scroll World | Desktop fine-pointer pacing, chapter reveal, scenic continuity | Collection links, products, price, fit, availability, cart, checkout, escape/back navigation | Native horizontal rail, visible previous/next controls, ordinary document scroll |
| Video | Atmosphere and temporal texture | Brand identification, route discovery, policy/fulfilment, CTA visibility | Poster/still, equivalent text, semantic links |
| Hotspots / 3D / depth | Optional spatial orientation and discovery | Sole route to product, variation, add-to-cart, or support | Linked product cards/list, keyboard focus, touch targets, static scenes, no-WebGL behavior |
| Motion | Transition and feedback | Focus visibility, state announcement, error recovery, payment completion | Reduced-motion static equivalence, focus restoration, no-JS baseline |

## Logo-off test (required before visual approval)

Hide wordmarks, lockups, monograms, rose marks, named collection labels, and video branding. Independent reviewers receive home, one collection, shop, PDP, cart, and service at 390×844, 768×1024, and 1440×900.

Pass only if reviewers describe one system: Oakland-rooted, garment-first luxury streetwear; a sequence from concrete/city context to distinct worlds; a restrained dark field with one collection accent; and a purchase path with fit, availability, fulfilment, and returns legible. They must complete browse → PDP → cart → checkout entry with reduced motion and without video, Scroll World, or 3D. This is a proposed test, not a passed result. **RECOMMENDED / UNVERIFIED.**

## Uncertainties and blockers

- No immutable V2 candidate, manifest, rights review, or provenance binding: **UNVERIFIED.**
- V2 uses WooCommerce attachment images while SOT requires generated `sot-images.json` resolution; runtime binding is not demonstrated: **OBSERVED / UNVERIFIED.**
- No browser evidence for pixels, responsiveness, overflow, video fallback, keyboard, screen reader, contrast, performance, or logo-off recognition: **UNVERIFIED.**
- No staging/payment/order evidence for stock races, idempotency, checkout recovery, guest/account ownership, returns, or policy accuracy: **UNVERIFIED.**
- V2 plan lists collection media as `interim-pending-mj, non-shippable`; no approved replacement/exception was supplied: **OBSERVED / UNVERIFIED.**

## Files read

Canon/SOT: `SOT.md`; V1 `front-page.php`, homepage parts, PDP/archive/cart/checkout templates, `inc/product-catalog-display.php`, and `data/visual-manifest.json`.

V2: `README.md`, `functions.php`, `front-page.php`, `page.php`, `template-collection.php`, header/footer, product/archive/cart templates, `assets/js/theme.js`, and `assets/css/theme.css`.

Planning: V2 gap-closure procedure, V2 page-and-imagery plan, and Fashion Theme Brain taxonomy/merchandising pack. The prototype worktree lacks `.wolf/memory.md` and `docs/design/fashion-design-system-team.md`; neither was substituted with a non-canonical equivalent.
