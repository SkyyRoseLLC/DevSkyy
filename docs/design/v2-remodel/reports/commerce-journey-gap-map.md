# V1 → V2 commerce journey gap map

**Historical candidate:** `codex/skyyrose-flagship-v2-prototype` at `0074fc6e6b161266f9181303296941cd474208af` (preserved by capture `1982efe66`).

**Active candidate:** `codex/skyyrose-v2-marketplace` at `a620435d646d89d3f99840350d3151a25dcbec47`, with tracked V2 paths and explicit payload/attestation evidence in `.fashion-theme/promotion-manifest.json`. The gap classifications below remain valid as release gates; they must not be read as evidence that the active candidate is untracked.

**Decision rule:** `OBSERVED` means source inspection only; `RECOMMENDED` is a contract input, not implemented behavior; `UNKNOWN` has no source or live proof. A client projection is never purchase authority. Product, variation, price, stock, cart, checkout, and order ownership remain WooCommerce authority.

## Executive gap statement

V2 preserves and elevates the strongest V1 differentiator: an editorial journey through Scroll World, video, collection scenery, and collection-aware PDP context. Its commercial layer is incomplete: it has custom archive, PDP shell, and cart templates, but no V2 overrides for product-loop cards, checkout, thank-you, account, search, wishlist, or returns/service. Consequently WooCommerce core/plugin behavior will render those surfaces, without evidence that the V2 visual, state, accessibility, or product-truth contract reaches them.

The immediate work is not to flatten V2 into a conventional grid. It is to make every editorial moment a truthful, progressive-enhancement bridge into a purchasable WooCommerce state—and to prove that a media failure, reduced motion, stock race, or payment failure cannot make purchase unavailable or misleading.

## Evidence and limits

| Class | Evidence read | Finding |
|---|---|---|
| `OBSERVED` V1 commerce | `wordpress-theme/skyyrose-flagship/woocommerce/{archive-product.php,content-product.php,single-product.php,cart/cart.php,cart/cart-empty.php,checkout/form-checkout.php,checkout/thankyou.php}`, `search.php`, `searchform.php`, `page-wishlist.php`, and `inc/{woocommerce.php,woocommerce-preorder.php,customer-enhancements.php,wishlist-functions.php}` | V1 has custom loop/PDP/cart/checkout/thank-you/search/wishlist surfaces, a shortcode fallback for account, preorder hooks, variation UI synchronization, and cart fragments. |
| `OBSERVED` V2 commerce | `wordpress-theme/skyyrose-flagship-2/{front-page.php,functions.php,page.php,template-collection.php,woocommerce/archive-product.php,woocommerce/single-product.php,woocommerce/cart/cart.php,assets/js/theme.js}` | V2 has collection scenes, native-scroll rails, video/scenery, a custom archive and product shell, product-card helper, and a custom cart. Only three WooCommerce template overrides exist. |
| `OBSERVED` gap procedure | `/Users/theceo/plugins/fashion-theme-team/skills/fashion-theme-team/brain/showcase/v2-gap-closure.html` | P1 requires product/variation/media/rights binding; P4 requires stock-race, payment, idempotency, guest/auth ownership, returns, appointment/gift/loyalty/waitlist rules; P5 requires browser/a11y/performance proof. |
| `OBSERVED` design-system charter | `/Users/theceo/DevSkyy/docs/design/fashion-design-system-team.md` | Complete coverage includes every commerce state, framework adapters, accessibility, performance, governance, and independent QA; polished home with default commerce states is explicitly incomplete. |
| `OBSERVED` official WooCommerce references | [template structure](https://developer.woocommerce.com/docs/theming/theme-development/template-structure), [Store API](https://developer.woocommerce.com/docs/category/store-api), [cart API](https://developer.woocommerce.com/docs/apis/store-api/resources-endpoints/cart/), [checkout API](https://developer.woocommerce.com/docs/apis/store-api/resources-endpoints/checkout), [variable-product reference](https://woocommerce.github.io/code-reference/classes/WC-Product-Variable.html) | Template override version compatibility must be maintained. Store API represents the current session; write operations require a Nonce or Cart Token; variations are excluded from product responses by default and must be selected/validated server-side. |
| `UNKNOWN` | `.wolf/memory.md` in this worktree; an authenticated official-docs session; a running WooCommerce staging site | The required session file is absent locally. Public official documentation was accessible, but no authenticated documentation connector/session was configured. No live behavior is asserted. |

## Journey comparison and priorities

| Priority | Journey | V1 observed | V2 observed | Gap / contract outcome | Live WooCommerce staging gate |
|---|---|---|---|---|---|
| P0 | Global entry → bag | Header/cart fragments and custom commerce scripts are present. | Header shows server-rendered `WC()->cart` count and a bag link; no observed fragment or Store API refresh. | **RECOMMENDED:** one CartState authority contract drives header, cart, and checkout-adjacent summary; announce confirmed count changes. | Add, remove, variation add, expired session, guest→login merge, cache/CDN behavior. |
| P0 | Home → world → product | V1 has conventional storefront composition. | V2 home and collection rails are first-class Scroll World/video/scenery journeys; product helper uses `wc_get_products`, name, price HTML, and parent `is_in_stock`. | **RECOMMENDED:** retain scenery but expose an always-present semantic product rail/list and CTA per scene; do not make video, horizontal scroll, or WebGL a purchase prerequisite. Parent stock label must not claim a selectable variation is purchasable. | Media blocked/404/slow, JS disabled, reduced motion, keyboard/touch bridge, mobile widths, each CTA resolves an eligible product. |
| P0 | Shop/search → product | V1 custom archive, loop card, search and search form exist. | V2 archive delegates loop card to WooCommerce because it has no V2 `content-product.php`; no V2 search override. | **RECOMMENDED:** canonical ProductCard adapter with search/filter/result/empty/error/loading states, then deliberately reuse it in V2 archive and scene bridges. | Query, filter, sort, pagination, no results, stale result after stock/price change, keyboard focus after update. |
| P0 | PDP → variation/fit → add | V1 has a custom PDP and variation event synchronization; V1 fit enhancements exist. | V2 wraps WooCommerce `content-single-product`; collection context and preorder copy surround it. No V2 fit, variation, add-to-cart feedback, or availability adapter is observed. | **RECOMMENDED:** VariationResolver is authoritative for selected attributes, variation ID, display price, availability/backorder/preorder, media, and addability. Fit must be a separate disclosed guidance contract, never a claimed fact when source data is absent. | Complete/incomplete attributes, invalid combination, sold-out child, low stock, backorder, price/media change, quantity limits, double submit, server rejection. |
| P0 | Cart → checkout-adjacent | V1 has custom cart and checkout/thank-you templates. | V2 custom cart retains core hooks/collaterals and renders an empty state, quantity, remove, coupon, and update. No V2 checkout or thank-you override. | **RECOMMENDED:** a CheckoutReadiness summary must display server-calculated totals, shipping/tax/payment eligibility and fulfillment disclosure before final submission; it may link to core checkout but cannot invent its status. | Coupon accepted/rejected, shipping recalculation, tax/address change, item removed or stock changed, payment pending/failed/cancelled, idempotency/order ownership. |
| P1 | Account/order service | V1 injects `[woocommerce_my_account]` when account content is otherwise empty; no V1 account override is shipped. | No V2 account override or account link is observed. | **RECOMMENDED:** use WooCommerce account endpoints as the data authority behind a V2 AccountShell; cover signed-out, login/reset, dashboard, orders, order detail, addresses, downloads if enabled, and authorization failures. | Guest order lookup, logged-in ownership, cross-account access denial, session expiry, password/reset error paths, order-status updates. |
| P1 | Returns, support, fit, policy | V1 has wishlist/support-related facilities; return workflow implementation was not established by this read. | V2 footer/contact link to shipping-returns, size guide and support; contact form has success/error notices. No returns case workflow or policy data contract is observed. | **RECOMMENDED:** ServiceCase and ReturnEligibility are server-backed or explicitly policy-page-only. Do not show eligibility, dates, fees, labels, or appointment capacity without authoritative data. | Eligibility boundary, RMA create/cancel, authorization, failed submission/retry, rate limiting, privacy, notification and status updates. |

## Editorial-to-product bridge moments

These are `RECOMMENDED` composition points that protect V2's Scroll World/video/scenery rather than replacing them.

| Moment | Product bridge | Required fallback and truth boundary |
|---|---|---|
| Home hero video | Persistent “Shop new arrivals” link plus compact live ProductRail below/after the hero. | Poster or static image failure still leaves the link and product rail in document order; no auto-play dependency. |
| Collection rail / Scroll World chapter | Each scene/card carries a semantic product link or “Shop this world” link with collection context. | Native horizontal scroll remains usable; if JS/scenery fails, a vertical list of links and collection product rail remains. |
| Pre-order salon/hotspot | Hotspot opens/navigates to a real product or collection only after mapping to its canonical Woo product ID/URL. | Decorative imagery cannot imply an item is sellable; unmapped/unknown hotspot becomes non-purchasing editorial content with no stock or price claim. |
| Collection manifesto/lookbook | “Shop the pieces in this story” ProductRail derived from current catalog query. | Empty collection shows a truthful empty state and a shop fallback, never substituted products unless explicitly labeled and authorized. |
| PDP collection-world aside | Keep the world context after `content-single-product`, paired with a return-to-collection link. | Add-to-cart, variation form, price, availability, and errors remain in the core WooCommerce form, ahead of optional media. |
| Cart and order confirmation | Small still/context tag can preserve provenance for the chosen collection. | Cart/order facts, totals, fulfillment and payment state come from WooCommerce; imagery never substitutes for line-item data. |

## Prioritized state matrix

`✓` means source evidence exists for part of the state only, not that it has been browser-tested. `—` means no V2 implementation was observed. All rows marked **GATE** require a live staging proof before closure.

| Surface/component | Default / hover / focus / active / selected / disabled | Loading / empty / success / warning / error | Stock / variation / media-failure / reduced-motion | Classification and next gate |
|---|---|---|---|---|
| Header + menu + bag | `✓` menu `aria-expanded`, Escape close, server count; hover/focus visuals unverified | — | **RECOMMENDED:** count must update only after confirmed cart response; video has poster | `OBSERVED` partial. **GATE:** keyboard, focus restore, cart mutation, media-off. |
| Home hero/video + rail | Links/controls present; rail focusable; scene controls exist | — | Poster exists; **RECOMMENDED:** timeout/error static fallback and CTA never blocked by motion | `OBSERVED` partial. **GATE:** 390/768/1440, JS off, video fail, prefers-reduced-motion, touch. |
| ProductCard / archive / search | V2 helper has links; archive delegates card state to core | Empty helper text exists; archive/search loading/error absent | Parent `is_in_stock` shown; variation-level state unknown; image empty span exists | `OBSERVED` partial. **GATE:** actual core loop/search/filter pages and stock transitions. |
| PDP / VariationResolver / Fit | Core template likely supplies controls; V2 wrapper adds breadcrumb/world | Core notices may render; V2-specific errors absent | V1 has variation sync and fit work; V2 has no adapter. **RECOMMENDED:** locked add button until valid selection; explicit unavailable/backorder/preorder states; factual fit source/status. | P0 `RECOMMENDED`. **GATE:** all variation combinations, add failure, focused error summary, image/media fallback. |
| Add-to-cart feedback | Core action inside delegated single-product template | V2 no observed pending/success/error feedback | **RECOMMENDED:** button uses `aria-busy`, prevents duplicate submit, then announces confirmed cart change | P0. **GATE:** latency, duplicate click, out-of-stock race, invalid variation. |
| Cart + summary | Quantity/remove/update/coupon markup and core hooks `✓` | Empty cart `✓`; coupon/update error feedback relies on core notices | Cart values use Woo APIs; no observed stale-stock/recalculation UI | `OBSERVED` partial. **GATE:** update/remove/coupon/totals/shipping/stock change and focus/live notice. |
| Checkout + payment + thank-you | V1 custom surfaces; V2 is core fallback | V2 state design is absent | Payment/order state must remain Woo/payment-gateway authority | P0 `UNKNOWN`. **GATE:** guest/auth, validation, 3DS/redirect, pending/failed/cancelled/success, duplicate order prevention. |
| Account + order detail | V1 shortcode fallback; V2 none | V2 states absent | Ownership/access boundaries unverified | P1 `UNKNOWN`. **GATE:** sign-in/out, reset, order visibility, address validation, session expiry. |
| Returns/service/contact | Contact success/error markup `✓` | Contact success/error `✓`; return state absent | Shipping/return/fit pages only linked; their facts unverified | P1 `OBSERVED` contact only. **GATE:** policy/version, RMA eligibility and privacy flow. |
| Wishlist / save | V1 feature exists | No V2 bridge/override observed | Guest persistence and account merge require authority proof | P2 `UNKNOWN`. **GATE:** guest/auth persistence, cap/error/removed item. |

## Component contract inputs

### Canonical components to reuse before making new ones

1. **WooCommerce product loop and single-product form** are the default authority adapters. V2 must wrap/slot them before duplicating price, availability, variation selection, or add-to-cart logic.
2. **WooCommerce cart collaterals, notices, quantity input, coupon, and checkout/account endpoints** remain the default transaction components. V2 supplies shell, slots, and presentation only after state parity is demonstrated.
3. **V2 ProductCard helper, collection rail, scene hotspot/card, header, cart, and contact form** are V2-specific candidates to normalize. Their current divergence must be closed through contracts, not another parallel card/form system.

| Component | Anatomy and slots | Variants / sizes | Required state contract | Authority/data boundary |
|---|---|---|---|---|
| `EditorialWorld` | `media`, `poster`, `eyebrow`, `title`, `story`, `bridge`, `productRail` | `home`, `collection`, `preorder`; `full`, `compact` | idle, playing, paused, loading, failed, reduced-motion, no-JS | Scenery/media are editorial. `bridge` resolves canonical product/collection URL; it never owns price/stock. |
| `ProductCard` | `image`, `badges`, `collection`, `name`, `price`, `availability`, `fitSummary?`, `action` | `rail`, `grid`, `search`, `related`; `sm/md/lg` | default, hover, keyboard focus, unavailable, pre-order, loading, empty image, price/stock changed | Product/price/availability from current Woo product response. Parent product is not a selected variation. |
| `VariationResolver` | `attributeGroup[]`, `selection`, `price`, `availability`, `media`, `notice`, `reset`, `addAction` | `select` or accessible chips; `inline/sticky` | unselected, partial, resolving, valid, invalid, unavailable, backorder/preorder, error | Server/Woo variation resolution is authoritative. Submit includes product/variation ID and selected attributes; never infer availability from a scene. |
| `FitGuidance` | `summary`, `sizeGuide`, `measurements?`, `modelContext?`, `disclaimer`, `helpLink` | `inline`, `drawer`, `page` | available, unavailable, loading, source-stale, error | Only render factual fields with a cited product/SOT source; otherwise show neutral size-guide/support recovery. |
| `CartSummary` | `lineItems`, `quantity`, `coupon`, `totals`, `shipping`, `notices`, `checkoutAction` | `page`, `drawer`, `checkout-adjacent` | updating, recalculating, empty, valid, stock changed, coupon error, server error | Woo cart/session totals and constraints win. Mutations return confirmed cart; token/nonce is never exposed in markup/logs. |
| `CheckoutReadiness` | `address`, `shipping`, `payment`, `disclosures`, `orderSummary`, `submit`, `errors` | `core`, `V2 shell` | incomplete, validating, ready, submitting, payment-action-required, failed, cancelled, success | WooCommerce/payment gateway creates and owns order/payment state. |
| `AccountShell` | `identity`, `nav`, `endpoint`, `notices`, `support` | `guest`, `authenticated` | sign-in, reset, dashboard, no orders, order detail, address error, unauthorized/session expired | Current user and Woo endpoints enforce ownership; never fetch another customer's order by ID client-side. |
| `ServiceCase` | `policyLink`, `orderLookup`, `reason`, `evidence`, `status`, `contactFallback` | `return`, `exchange`, `order-support` | eligible, ineligible, pending, submitted, rejected, error, cancelled | No return eligibility or label without authoritative server/policy result; Contact is fallback, not a fabricated RMA flow. |

## API and event contracts

The following is a design contract, not an implemented endpoint inventory. For a headless or enhanced client, prefer the current-session WooCommerce Store API; its cart and checkout writes require a nonce or Cart Token. Store product responses omit variations by default, so selected variation data must be requested/validated deliberately. Classic form posts must preserve the same server authority.

```ts
type ProductProjection = {
  productId: number;
  permalink: string;
  name: string;
  priceHtml: string;          // rendered by WooCommerce; not client-calculated
  productType: 'simple' | 'variable' | 'other';
  parentAvailability: 'in_stock' | 'out_of_stock' | 'backorder' | 'unknown';
  media: Array<{ src: string; alt: string; status: 'ready' | 'missing' }>;
  collectionSlug?: string;
  fulfillmentDisclosure?: string; // only where authoritative
};

type VariationSelection = {
  productId: number;
  attributes: Record<string, string>;
  variationId?: number;
  status: 'incomplete' | 'resolving' | 'valid' | 'invalid' | 'unavailable' | 'error';
};

type CartMutationResult = {
  status: 'success' | 'error';
  cart: unknown;              // full current-session server result
  notice?: { level: 'success' | 'warning' | 'error'; message: string };
};
```

| Intent | Request boundary | Confirmed response/UI behavior |
|---|---|---|
| Browse/filter/search | `GET /wc/store/v1/products` or server-rendered Woo archive | Render current product projection; distinguish no results from fetch failure; never synthesize variations. |
| Resolve variation | Core variation form/Woo resolver, or variation-aware server query | Update selection only from resolved product/variation; invalid and unavailable combinations remain actionable/recoverable. |
| Add/update/remove cart | Core form or Store API cart endpoint with nonce/Cart Token | Disable duplicate action while pending, then replace with returned cart and announce confirmed result. On error preserve selection/input and focus a perceivable notice. |
| Recalculate shipping/totals | Woo cart/checkout only | Replace totals only from response; disclose that final availability/payment may still be validated on submit. |
| Submit checkout | Woo checkout/payment gateway only | Treat redirect/3DS, payment pending, failure, cancellation, and success as distinct states; deduplicate through server/order idempotency behavior. |
| Account/return service | Woo account endpoints or authorized service endpoint | Scope records to current authenticated owner or permitted guest lookup; never use a public order-ID lookup. |

## Accessibility, media, and escape-hatch policy

- **Media is additive.** Video must have a poster; scenery/hotspots need text-equivalent links; content and buying CTA remain in the DOM before media completes. Supply a static image/vertical list for media error, unavailable WebGL, JS off, or `prefers-reduced-motion`.
- **Reachability.** Every bridge is a native link or button with visible focus, a logical tab order, target size appropriate to touch, and no scroll-jacking. The menu's observed Escape/focus behavior must be extended to drawers/dialogs with focus restoration.
- **Status.** Use `aria-busy` for pending operations, `role=status` for confirmed non-critical updates, and an error summary with field associations/focus for blocked checkout and variation states. Do not rely only on color, motion, or a scene badge.
- **Motion.** Reduced motion replaces autoplay/parallax/pinning with static media and ordinary scroll; it does not remove product cards, fit, cart controls, or checkout access.
- **Responsive proof.** Capture 390×844, 768×1024, and 1440×900 for default, focus, selected variation, out-of-stock, cart-update, error, no-media, and reduced-motion states.
- **Escape hatches.** A surface may add a V2 wrapper/slot, not fork the transactional authority. A new custom cart, checkout, account, or returns component needs: (1) a written reason core hooks/blocks cannot meet the need, (2) product/cart/order authority mapping, (3) state and a11y parity fixtures, (4) source and generated-asset ownership, (5) compatibility/version plan, (6) staging evidence and independent review, and (7) rollback to the canonical WooCommerce template. No exception permits client-authorized purchase, invented inventory/fulfillment/fit facts, or obscured recovery.

## Closure checklist / handoff

1. **P0:** bind a candidate snapshot of SKU → Woo product → variation → approved media → rights/policy record; map every Scroll World hotspot and product bridge.
2. **P0:** implement `ProductCard`, `VariationResolver`, cart-header synchronization, and media/reduced-motion fallback contracts by adapting canonical Woo templates/hooks first.
3. **P0 staging:** execute purchase flows for simple/variable/preorder (only where live fixtures exist), stock race, coupon, shipping, payment success/failure/cancel, guest/auth ownership, and duplicate-submission protection.
4. **P1:** extend V2 shell coverage to search, checkout/thank-you, account/order, fit, returns/service, and wishlist only when contracts and authoritative sources exist.
5. **P5/P6:** capture the required responsive/a11y/performance states and obtain independent visual/security/content review. No deployment decision follows from this report.

**Files read:** listed in the Evidence and limits table, plus V2 `header.php`, `footer.php`, `index.php`, and file inventories under both V1/V2 theme roots. No production files were changed.

## 2026-08-13 candidate journey update

- Homepage entry opens only Signature, Black Rose, and Love Hurts; Kids Capsule is disclosed later through the three-chapter Royal Procession and exact `kids-001` / `kids-002` Woo bridge.
- ProductCard is one collection-owned `house-portal` adapter across homepage, archives, search, and collection rails. Scenery is presentation; Woo product media, names, prices, availability, and actions remain authority.
- The eight Jersey Series SKUs are isolated from Black Rose ProductCards and collection markup. The BART film is the series entry chapter on the homepage; individual jersey truth resolves on Woo product routes.
- Every collection statue hero is a scene-only frame. Typography, graphic artifacts, SEO heading, body copy, and navigation are not duplicated over the composed image; the semantic identity chapter follows in normal document flow.
- Candidate evidence: source gate PASS; PHP/JS syntax PASS; all four collection routes zero-overlay/zero-Jersey checks PASS; 390/768/1440 overflow and media checks PASS; desktop local Lighthouse 100/100/100/100. Deployment and real staging purchase-state evidence remain blocked pending founder approval.
