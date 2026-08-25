# V2 cinematic motion and responsive contract

**Status:** `RECOMMENDED / UNAPPROVED` — a build and QA contract, not a visual approval or evidence of live behaviour.

**Scope:** `wordpress-theme/skyyrose-flagship-2` prototype. This preserves the founder direction: Scroll World, ambient video, scenic collection experiences, collection-specific cards/PDPs, and collection pre-order scenes become one cinematic commerce system. Cinema is progressive enhancement over native document and WooCommerce behaviour; it cannot be the only way to discover, select, or buy a garment.

**Evidence read:** V2 `front-page.php`, `template-collection.php`, WooCommerce archive/PDP templates, `assets/js/theme.js`, `assets/css/theme.css`, and asset/enqueue inventory; V1 Scroll World, collection feature-scroll/motion, immersive templates, and the V2 commerce-gap report. No route was browser-tested and no asset, product, variation, rights, or visual decision is approved here.

## Binding principles

1. **Purpose before motion.** Each movement must orient the visitor, preserve a spatial relationship, acknowledge an input, or clarify an actual commerce state. No uniform “everything rises and fades in” reveal system; V2's current broad `sr2-reveal` selector is an exact remediation candidate.
2. **Native scroll remains sovereign.** No global smooth-scroll engine, scroll-position rewriting, `overflow: hidden` on page content, wheel capture, synthetic scroll inertia, or pin that traps a keyboard/touch visitor. Pinning is a desktop enhancement only and must degrade to ordinary vertical reading and a native horizontal rail.
3. **The garment wins.** Scenic motion may establish a collection's point of view but cannot obscure a product image, price, selected variation, availability, fit/service truth, add-to-bag, cart, or recovery action. No garment-competing particles, noise loops, cursor effects, gratuitous glare, or ambient animation in the purchase decision zone.
4. **Media is never a gate.** The hero and each scene have a sized poster/static image, semantic heading/story, ordinary collection/PDP links, and a visible product bridge in initial DOM. A WebGL/3D treatment is an optional layer above the same image/card/link structure.
5. **User agency wins.** No autoplay audio. Any optional sound is muted by default, starts only from an explicit labelled control, and has an equally obvious stop/mute control. Motion is cancellable by preference, page exit, visibility, and input; reduced motion preserves the narrative and all commerce actions as static equivalents.

## Tokens and motion grammar

These are proposed system tokens. Collection identity comes from crop, setting, typography, copy, one accent, and garment framing—not from a generic animation template.

| Token | Value | Use / prohibition |
|---|---:|---|
| `--sr2-motion-instant` | `0ms` | reduced motion, error/price/stock truth, user-controlled direct state |
| `--sr2-motion-feedback` | `120ms` | press, focus-linked state, add-to-bag acknowledgement; never delays navigation |
| `--sr2-motion-ui` | `180ms` | menu, card-state, scene-card swap |
| `--sr2-motion-scene` | `420ms` | one scene handoff/crossfade after an intentional input |
| `--sr2-motion-editorial` | `700ms max` | one chapter arrival or image mask; one per chapter, no repeating reveal cascade |
| `--sr2-ease-enter` | `cubic-bezier(.22,.61,.36,1)` | arrivals only |
| `--sr2-ease-exit` | `cubic-bezier(.4,0,.2,1)` | departures only |
| `--sr2-distance-ui` | `8px max` | UI feedback; no large slide-ins for purchase controls |
| `--sr2-distance-editorial` | `min(3vw,32px)` | scenic depth only; never moves readable content off target |
| `--sr2-focus-ring` | `2px + 3px offset` | persistent visible focus; cannot be replaced by motion |

Only `transform` and `opacity` may animate continuously. A single opacity crossfade is acceptable for an image/scene handoff; avoid blur, filter, layout-property animation, perpetual video seeking, canvas grain, and unbounded `will-change`. Apply `will-change` immediately before an approved transition and clear it on `transitionend`/completion.

## Route-level choreography and product bridge

| Route / layer | Purpose and distinctive choreography | DOM-first commerce bridge and sequence | Native/cancellation contract |
|---|---|---|---|
| `/` House entry | The muted hero is atmospheric only: poster first, then a single 420ms opacity arrival if playback is available. The title and “Shop New Arrivals” link are static/legible from first paint. | Hero CTA is a native link. The world entry contains a native list of the four collection links before enhancement; first product rail is not deferred behind video. | `preload=metadata`; do not set hero video `src`/play for Save-Data, reduced motion, failed decode, or 390 compact path. Cancel fade and pause on `visibilitychange`, route/pagehide, error, or preference change. Never autoplay audio. |
| `/` Scroll World | A documentary route through four worlds: wide image sequence, chapter count, and one active chapter change. Each chapter's composition differentiates by collection; do not stagger all words/cards alike. | Every chapter has in-flow collection link plus 2–4 real product cards (name, current price, availability, image alt, PDP link). “Shop this world” is reachable without completing a scroll segment. | At 1440 fine-pointer only, the stage may be `position: sticky` while the document scroll maps to progress. It must exit naturally at the last chapter and retain `Previous/Next` buttons. At any other capability, use overflow-x auto with scroll-snap proximity—not mandatory snap—and normal page scroll. |
| Collection landing | Arrival gives a single scenic handoff into a collection-specific editorial composition, never a template fade. | First viewport contains collection name/context plus an ordinary product rail/list. Cards use that collection's image/crop and real PDP URL; price/stock are WooCommerce authority. | Scroll carries reading; no locked chapter. If scene media fails, retain image, copy, product cards, and collection archive CTA. |
| Collection PDP | Product is the hero; scene media can communicate provenance behind/beside it but variation selection and product facts never animate away. | Native variation form resolves selected variation, price, availability, media, and addability server-side. Sticky buy bar is a duplicate control only after the primary form is visible, and submits/focuses the same authoritative form. | Selection update is instant; at most a 180ms image crossfade. Abort stale client media requests on a new selection. In error/invalid/out-of-stock states, focus the associated error summary; no celebratory motion. |
| Collection pre-order scene | An optional “shop the room”: each collection gets one recognisable scene grammar and corresponding pre-order products; the scene is a discovery layer, not a selector. | Static scene image, labelled hotspot links, and an adjacent/after-scene accessible product list are rendered server-side. Hotspot, card, and list link to the same canonical product/PDP. Scene cannot claim stock, price, or a selected variant. | A hotspot activates on click/Enter/Space, not hover alone; touch uses a 44×44 CSS-pixel minimum target and first tap must never be hover-only. Escape closes any modal/drawer and restores trigger focus. No-WebGL gives the static scene + list with zero missing information. |
| Cart / checkout entry | Confirmation and recovery, not spectacle. | Confirmed cart mutation updates the server-backed bag count, `role=status`, and bag/cart links. Checkout remains WooCommerce-authoritative. | No scroll/chapter animation. Pending state uses `aria-busy`, disables duplicate submit only while a request is actually outstanding, and returns focus to a readable server error on failure. |

### Collection-specific story grammar

These are **recommendations requiring founder/brand and independent visual-QA review**, not approved creative.

| Collection | Scenic grammar | Permitted motion | Card / PDP / pre-order distinction |
|---|---|---|---|
| Black Rose | Monumental concrete, night, silver restraint; a deliberate cut from city scale to garment detail. | One slow scene crossfade on explicit chapter advance; fine-pointer tilt is optional and capped at 2°. | Dense editorial crop with silver information rule; garment card and PDP lead with product texture/fit. Pre-order scene resolves to Black Rose products only. |
| Love Hurts | Intimate crimson/cathedral shadow; editorial framing narrows rather than accelerates. | One controlled ink/mask reveal on direct section entry, then stillness; no tap-generated particles in buying zones. | Crimson is a sparse state accent, not an availability signal. Cards/PDPs use intimate crop and clear data. Pre-order scene maps only to Love Hurts product IDs. |
| Signature | Open Oakland/water/architecture; measured horizon and craftsmanship. | Single lateral image drift under fine pointer, maximum 8px, stopped on leave/blur. | Airier grid/crop and craftsmanship modules; product image and price remain foreground. Pre-order scene maps only to Signature product IDs. |
| Kids Capsule | Playful but composed, heirloom/runway setting; no visual noise. | One input-triggered 120ms acknowledgement on a safe non-commerce detail; no autonomous bouncing field or confetti near product actions. | Generous targets, simple card hierarchy, accurate adult/child association. Pre-order scene maps only to eligible Kids Capsule products. |

## Input, focus, and cancellation behaviour

| Interaction / state | Required behaviour |
|---|---|
| Keyboard | Tab order follows DOM reading order: skip link → header → copy/CTA → chapter controls → chapter/product links → next section. `Enter` activates links/buttons; `Space` activates buttons only. Left/right is optional only when a focused rail/control advertises it and must not prevent ordinary page keys. Escape closes menu/dialog/drawer, cancels an in-progress visual scene state, and restores focus to its trigger. |
| Focus | `:focus-visible` is a stable ring, not a transient animation. Focused rail item/hotspot synchronises the visual scene within 180ms but does not scroll the document unexpectedly. When a chapter control does scroll, move focus to the chapter heading with `tabindex=-1` only after destination content is present. |
| Touch | Preserve vertical swipe in every rail/stage. Horizontal rails use native finger panning with `touch-action: pan-x pan-y`; no first-tap hover dependency, 44×44 minimum targets, no drag-only discovery, no double-tap requirement. Native momentum and pinch zoom remain available. |
| Pointer | Fine-pointer depth/tilt is optional, nonessential, and reset on pointerleave, pointercancel, blur, `visibilitychange`, and reduced-motion change. It is never attached to primary product image/CTA on PDP. |
| Menu/modal/drawer | Open state must be a semantic dialog only when modal; trap focus only inside an actual modal, close via Escape/close button/backdrop where appropriate, lock only the background during that modal, and restore origin focus. Do not use a full-page overlay as an editorial loading state. |
| Route change / bfcache | Register one route-owned cleanup function: cancel RAF/timeouts, abort controllers, observers, media listeners, pointer transforms, and transient classes on `pagehide`; re-measure after `pageshow`. Never leave a transformed rail or an inaccessible hidden element after navigation. |
| Visibility / preference | Pause video and RAF when hidden; reconnect only if still eligible and in view. Listen to `prefers-reduced-motion` and Save-Data changes where available; switching either on immediately cancels enhancement and exposes static state without reload. |

## Responsive transformation matrix

| Viewport | Composition | Scroll and scene behaviour | Product bridge / touch requirements |
|---|---|---|---|
| **390×844 (compact)** | One-column reading order. Hero uses poster/static art; navigation is a dialog-like menu with safe-area padding. Collection cards are full-width or a native horizontal rail with visible next affordance; no cropped text overlay that requires hover. | No pinned world, page-level wheel interception, parallax, cursor depth, automatic video, or WebGL. Scenes are still image + labelled hotspot list + product list. Use `100svh`/`100dvh` carefully, never fixed `100vh` content traps. | CTAs and scene controls ≥44×44 CSS px; product card contains visible name/price/availability/PDP link without horizontal scrolling. Keep one primary add action per screen region; sticky mobile buy bar never covers variation/error text or browser controls. |
| **768×1024 (medium)** | Two-up product/cards only when each card stays ≥min-content width; collection copy can sit beside media only where its container permits. Keep logical DOM order media → story → bridge. | Native rail and explicit previous/next controls; sticky image is permitted only if all item copy remains ordinary document flow. No page pinning, wheel hijack, or hover-dependent scene controls. | Hotspots and cards support touch and keyboard equally. Scene/product list stays visible without opening an overlay. Images reserve intrinsic ratio/width/height. |
| **1440×900 (expanded)** | Wide scenic frame can support deliberate negative space, a side chapter rail, and collection-specific card rhythm. Card grid may expand but product data remains in each card. | Fine-pointer, non-reduced users may receive one sticky Scroll World stage whose progress derives from native `scrollY`; Back/forward, Home/End, PageDown, scrollbar, trackpad, and normal wheel retain expected behaviour. Depth is one layer only. | Chapter controls and direct collection/product links remain visible. Pinned portion is bounded by its own content distance, exits at last chapter, and has a static/native mode switch in capability failure. |

Use fluid values and container queries for reusable cards. Page transformations may use no more than the compact/medium/expanded breakpoints above. Every image/video area declares dimensions or aspect ratio before load; `object-fit` requires an approved focal-point/crop rule per collection.

## Reduced-motion and no-WebGL equivalence

| Enhanced feature | Reduced-motion / no-JS / media-failure equivalent |
|---|---|
| Hero video fade/loop | Poster/still shown immediately; heading, lede, and CTAs unchanged; video remains paused/not requested when possible. |
| Scroll World sticky translation | Ordinary vertical chapter sections or native horizontal rail with previous/next; all chapter links and product cards visible. |
| Scene crossfade, depth, WebGL, hotspot treatment | Static SOT-approved scene image, text description, labelled native links, and same product list. Feature detection failure must not show a loader forever. |
| Card tilt/hover animation | Visible focus/selected state and a simple 120ms press response; neither price nor garment image disappears. |
| Scene/map sequence | All stops/chapters are shown as completed/static information; status text is present rather than progressively announced. |
| Menu/route visual transition | Direct state change with focus management intact. |

Reduced motion is not a degraded purchase path: URL reachability, heading order, collection recognition, product facts, selection, add-to-cart, cart, checkout entry, errors, and support remain equivalent. A static poster or scene must not be labelled as a failed experience.

## Performance budgets and measurable gates

Budgets apply to the **incremental cinematic layer**, measured on a representative real product/scene route with cold cache, 4× CPU throttle, and a mobile network profile; they do not certify the unmeasured baseline. A failure is a gate failure, not a reason to weaken the fallback.

| Area | Budget | Required measurement / pass condition |
|---|---:|---|
| Critical CSS | ≤14 KiB gzip incremental | No cinematic CSS blocks initial text/CTA paint; source/min output audit identifies bytes. |
| Critical JS | ≤18 KiB gzip incremental; ≤35 KiB gzip total V2 interaction bootstrap | No Three/WebGL/scene bundle on initial page. Script parse/eval does not create a long task >50ms in lab trace. |
| Optional scene/WebGL | ≤120 KiB gzip loader/runtime before model; model ≤700 KiB gzip, requested only after explicit scene entry/idle + capability | No-WebGL capability, Save-Data, and reduced-motion paths request neither runtime nor model. If budget cannot be met, ship static scene only. |
| Fonts | ≤3 WOFF2 faces / ≤180 KiB initial route | `font-display: swap`; no layout-changing late font swap for hero/price/CTA. Collection display face loads only on its collection route. |
| Images | Hero poster ≤180 KiB at 390, ≤320 KiB at 768, ≤450 KiB at 1440; one LCP image only | Correct `srcset`/`sizes`, explicit dimensions/aspect ratio, LCP image `fetchpriority=high`; below-fold scenes/cards lazy-load. |
| Video | Hero ≤900 KiB desktop / ≤450 KiB medium; **0 bytes requested on compact, reduced motion, or Save-Data** | Muted, `playsinline`, metadata only; poster is LCP candidate rather than a blocked video frame. No audio track playback/autoplay. |
| CLS | ≤0.05 page route; 0.00 attributable to scenes/rails/card selection | Capture layout-shift entries; no script-written height after first interaction without reserved geometry. `setupPinnedWorld`-style height changes must complete before entering the pinned interval and remeasure only after resize settles. |
| LCP | ≤2.5s p75 target, ≤3.0s lab under stated profile | Report element, URL, transfer size, render time, and whether video/3D was requested before LCP. Cinematic enhancement must not be LCP dependency. |
| INP | ≤200ms p75 target; ≤100ms lab interaction processing for rail controls, hotspots, variation selection, add action | Trace tap/click/key from event to next paint; no blocking scene decode, `scrollTo({behavior:'smooth'})`, or long animation gates an actionable result. |
| Motion | ≤1 active RAF loop/page; 0 when hidden/out of view/reduced; continuous compositor work ≤4ms/frame at 60Hz | Performance trace verifies cancellation on `visibilitychange`, preference change, route exit, and `pointercancel`. |
| Touch/overflow | 0 horizontal document overflow at 390/768/1440; target ≥44×44 CSS px; no prevented vertical scroll | Automated `document.documentElement.scrollWidth <= innerWidth`; wheel/touch test proves a vertical gesture exits rail/stage immediately at boundaries. |
| Fallback | 100% of scenic product CTAs resolve without JS, video, WebGL, or reduced motion | Disable each dependency separately. Product name, current price/availability source, PDP route, and cart/checkout entry remain reachable. |

## Exact remediation candidates from current source (not applied)

| Priority | Observed candidate | Required remediation | Classification |
|---|---|---|---|
| P0 | `assets/js/theme.js` `setupRail()` calls `event.preventDefault()` for vertical wheel movement while a horizontal rail can move. | Remove page-scroll interception. Let native wheel/trackpad page scroll prevail; provide explicit previous/next and native horizontal rail scroll. If a desktop-only optional translation remains, it must not cancel wheel/touch input and must end naturally. | `RECOMMENDED` — scroll-jacking risk |
| P0 | `setupPinnedWorld()` mutates world height and translates rail from global `scrollY`. | Keep only behind fine-pointer + expanded + no-preference gates; add bounded lifecycle cleanup, ResizeObserver/debounced measure, bfcache/pagehide handling, static/native fallback, and no focus movement by passive scroll. Never pin at 390/768. | `RECOMMENDED` — bounded enhancement |
| P1 | V2 applies a broad shared IntersectionObserver reveal to section heads, products, intro, manifesto, steps, contact, service links, and image reveals. | Delete broad uniform reveal assignment. Replace with the route/collection-specific purpose matrix above; content must be visible before JS and only one named editorial arrival may occur per chapter. | `RECOMMENDED` — reject uniform reveals |
| P1 | Hero video loads/plays a repeated fade loop after `loadeddata`; header brand video is also loop-capable. | Make poster/static the primary experience. Do not request autoplay video on compact, Save-Data, or reduced motion; pause/cancel on visibility/error; never attach audio autoplay. Keep all copy/links visually complete independently of playback. | `RECOMMENDED` — media fallback |
| P1 | Scene hotspots are synchronised by `mouseenter`/`focus` in `theme.js`; source inspection alone does not prove semantic click/touch/product mapping. | Use native labelled links/buttons with click/Enter/Space activation, 44px targets, selected-state announcement, and an adjacent server-rendered product list. Bind each to a verified canonical product/PDP; no hover-only discovery. | `RECOMMENDED` — touch/keyboard/product bridge |
| P1 | The hero headline is rewritten with `innerHTML` to generate word spans. | Render non-JS text first; if word animation remains, construct safe spans without replacing semantic content and cancel/disable for reduced motion. Do not use this device-wide pattern as a reveal template. | `RECOMMENDED` — content stability |
| P2 | Existing V1 techniques include Lenis/GSAP/ScrollTrigger and scene loops; their existence is not permission to copy their input model. | Do not import Lenis or any global smooth-scroll/scroll-normalisation layer into V2. Use CSS/IntersectionObserver/native scroll where enhancement is necessary; load optional scene code only by route/capability. | `RECOMMENDED` — reject scroll-jacking |

## Verification matrix — all must be evidenced before approval

| Check | Exact proof | Pass criterion |
|---|---|---|
| Overflow | Browser screenshots plus `scrollWidth <= innerWidth` at 390×844, 768×1024, 1440×900 across home, each collection, PDP, pre-order scene, cart. | No horizontal document scroll, clipped CTA, inaccessible fixed control, or safe-area collision. |
| CLS | Web Vitals/layout-shift trace per route and interaction. | Route CLS ≤0.05; no shift attributable to media, rail pinning, font, sticky buy, header, or product selection. |
| LCP | Lighthouse/field candidate report with LCP element and request waterfall. | Meets stated target; video/WebGL/model absent from LCP critical path. |
| INP | Trace rail controls, hotspot touch/key, variation resolution, failed add, successful add. | Meets target; native feedback/announcement occurs without waiting for animation. |
| Motion | Record normal, reduced, Save-Data, hidden tab, pagehide/bfcache, `pointercancel`, and escape cases. | One or fewer active RAF loops; all timers/observers/media cancel; reduced path is readable and functionally equivalent. |
| Touch / keyboard | Real touch emulation/device and keyboard-only run. | Vertical scroll remains native; all interactive targets reach/operate with Tab/Enter/Space; Escape closes/restores focus; no hover-only product bridge. |
| Fallback | JS disabled, video 404/blocked, WebGL unsupported/blocked, image 404, slow network, media decode failure. | Static context, semantic links, product list/PDP/cart path remain available; no permanent loader, blank scenic region, or invented product state. |
| Commerce reachability | For every collection and active pre-order SKU, a manifest-backed mapping: scene/chapter/card → Woo product → selectable variation → approved media/rights. Exercise simple, variable, unavailable, pre-order, error, and stock-change cases on staging. | Every editorial entry reaches an eligible canonical PDP without scenic media; variation/add/cart responses are WooCommerce-authoritative; unresolved mapping fails closed. |
| Independent review | Founder/brand review of collection grammar and independent visual-QA red-team logo-off/browser assessment. | No self-approval. A PASS requires evidence for the full matrix and a separate visual-QA verdict. |

## Handoff sequence

1. Create the SKU/variation/media/rights and scene-hotspot mapping manifest; do not animate an unbound card or scene.
2. Establish initial DOM, static media, product rail/list, native links, and WooCommerce-authoritative PDP/cart states.
3. Implement responsive layout and no-JS/reduced/no-WebGL variants first; prove reachability and no overflow.
4. Add per-collection choreography within tokens and budgets, then the optional expanded desktop Scroll World only.
5. Measure the gates above and obtain independent visual QA and founder creative approval. This contract makes no approval claim.

