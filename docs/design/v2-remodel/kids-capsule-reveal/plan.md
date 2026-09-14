# Kids Capsule Chapter 04 — Royal Procession

Status: **implemented and locally verified; founder visual approval and release gates remain**  
Candidate: `codex/skyyrose-flagship-v2-prototype`  
Theme: `wordpress-theme/skyyrose-flagship-2`

## Selected direction

The founder selected Direction C, **The Royal Procession**. Kids Capsule does not appear in the opening homepage hero. The homepage sequence is now:

```text
Signature / Black Rose / Love Hurts opening worlds
        ↓
Chapter 04 invitation
        ↓
The red and purple guardians
        ↓
The heir on her throne
        ↓
Full collection and commerce journey
```

This is a short, native-scroll story rather than an autoplay film or WebGL scene. It uses three server-rendered chapters with explicit previous/next controls. JavaScript only synchronizes the active chapter and controls; it never captures wheel/touch input, locks the page, or gates links.

## Story and visual system

### Chapter 01 — The Invitation

- The existing Kids Capsule playroom opens the fourth world.
- Copy establishes inheritance, imagination, and the reason Kids was withheld from the first hero.
- The panel remains useful as a static poster with JavaScript disabled, reduced motion, or Save-Data.

### Chapter 02 — The Guardians

- The red guard uses the archive mascot wearing the exact red/black/white colorblock set.
- The purple guard uses the archive mascot wearing the exact purple/pink colorblock set.
- Each guard is paired with a separate WooCommerce product-proof card.
- `kids-001` and `kids-002` are resolved independently. A missing or invalid product removes that product's price, availability, and purchase claim instead of inventing a substitute.

### Chapter 03 — The Heir

- The founder-directed throne scene centers the SkyyRose mascot as the heir.
- The final collection CTA is a normal crawlable link to the Kids Capsule collection route.
- No price, stock, size, shipping, or preorder promise is emitted from collection narrative data.

## Commerce authority

`skyyrose2_get_products_by_skus( array( 'kids-001', 'kids-002' ), 'kids-capsule' )` is the only reveal-specific product resolver. It requires an exact SKU, a visible and published `WC_Product`, and the `kids-capsule` product category. The template reads product name, price HTML, stock state, image, and permalink from the resolved Woo product.

The reveal intentionally emits no Product or Offer schema. WooCommerce/PDP remains structured-data authority.

## Responsive behavior

### 1440px

- Native horizontal story rail with a 72vw authored panel and adjacent chapter preview.
- Invitation and heir panels retain cinematic wide scenery.
- Guardians share the scene stage and stay fully visible beside the Woo proof ledger.
- Explicit chapter controls remain available to pointer and keyboard users.

### 768px

- Panels expand to 84vw.
- Guardians occupy a full upper stage.
- Exact red and purple product proof remains a two-column ledger after measured fit.
- No page-level horizontal overflow.

### 390px

- The horizontal story transforms into normal vertical document order.
- Invitation poster, full-body red guard, full-body purple guard, Woo proof cards, and heir poster all remain present.
- Controls remain available, but the content never depends on them.
- No sticky panel, center-crop loss, scroll capture, or page overflow.

## Motion and fallback contract

- Native browser scrolling only; no `preventDefault`, wheel listener, touch interception, canvas, WebGL, or autoplay media.
- Active scenes use transform/opacity transitions only.
- One `AbortController` owns listeners and observer cleanup on `pagehide`.
- Reduced motion and Save-Data remove scroll snapping and all scene transitions.
- No JavaScript renders the full three-chapter sequence and hides enhancement-only controls.
- A failed image receives an `is-media-missing` state while the heading, story, product proof, and links remain.
- A missing Woo product renders a factual unavailable state with no price or stock claim.

## Measured local candidate evidence

- Reveal media: `470,300` bytes total (`459.3 KiB`), below the `550 KiB` desktop budget.
- Guardian layers: `93,008` and `96,032` bytes, each below `150 KiB`.
- Throne poster: `142,022` bytes, below `200 KiB`.
- Reveal JavaScript: `2,224` bytes minified / `1,019` bytes gzip, below `6 KiB` gzip.
- Component CSS: below `10 KiB` minified after the final budget pass; source and minified bundles are rebuilt together.
- Desktop performance trace: LCP `164 ms`, CLS `0.02`, no network or CPU throttling.
- A 120-frame chapter transition sample averaged `16.56 ms`, maxed at `18.7 ms`, dropped no full frames, and recorded no task over `50 ms`.
- Lighthouse local template audit: Accessibility `100`, Best Practices `100`, SEO `100`, Agentic Browsing `99`.
- Eyes-on and DOM checks cover 1440x1000, 768x1024, and 390x844; all three have zero page overflow.
- Keyboard activation advances chapters without focus theft. Reduced-motion, Save-Data, no-JS, media-failure, and missing-product states preserve equivalent facts and purchase reachability.

## Release boundary

This candidate is complete for founder review but is not yet a production release. Remaining gates are:

1. founder visual approval of the finished Royal Procession;
2. rights/usage records for the archive mascot and active throne/playroom scene bytes;
3. independent visual, accessibility, and performance review on the same candidate;
4. real staging WooCommerce identity, variation, cart, and purchase-path proof;
5. explicit founder authorization before any staging or production deployment.

No deployment occurred as part of this implementation.
