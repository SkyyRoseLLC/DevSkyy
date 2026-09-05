# SkyyRose V2 Phase 3A — Design System and Global Shell Contract

**Mode:** local implementation and isolated verification only.  
**Accepted runtime baseline:** `5de8e2f3eb40827a052996f72bfb290a95bd600a`.  
**Task ledger:** `tasks/v2-phase3-20260905/task-ledger.jsonl`.  
**Contract status:** foundation and global-shell source implemented; final browser/build verification in progress.  
**Pixel approval:** UNVERIFIED; independent reviewer required.

This contract covers tokens, global controls, header, house navigation, search shell,
bag shell, overlay coordination, footer, and responsive/accessibility foundations.
Do not begin theme implementation until the accepted baseline passes its existing
build/verification suite and local baseline screenshots are recorded at their actual
viewport dimensions. Documentation preparation does not satisfy that gate.

Whole-package staging promotion, production deployment, launch, payment completion,
integration changes, new media generation, and Phase 3B page art direction are outside
this authorization. Preserve the certified baseline history without amendments.

## Authority and source census

| Concern | Current source | Application |
| --- | --- | --- |
| Runtime token primitives | `wordpress-theme/skyyrose-flagship-2/assets/css/design-tokens.css` | Extend this existing `--sr2-*` graph; do not introduce a parallel palette or namespace. |
| Editor projection | `tools/v2-runtime/sync-token-contract.cjs` and V2 `theme.json` | Extend the existing build-time mapping; verify deterministic output. |
| Brand direction | `docs/brand/visual-references.md`, `CLAUDE.md` brand canon | Oakland-rooted American independent fashion; the Five are reference principles, not templates to copy. |
| Typography provenance | V1 `data/brand/typography.json`; V2 `assets/sot/fonts/` | Use existing local fonts and valid weights. No external dependency. |
| Collection identity | V1 `data/collections/<slug>/identity.json` | Preserve collection identity and one accent. The certified V2 accessible crimson remains its runtime value. |
| Component implementations | V2 `functions.php`, `assets/css/theme.css`, `assets/js/theme.js`, `template-parts/commerce/` | PHP and vanilla JS; preserve existing native Woo contracts. |
| Media and catalog authority | Certified Phase 2 media resolvers, generated presentation registry, catalog/SOT evidence | No image-state promotion or catalog reclassification. |
| Governance | `docs/theme-team-charter.md`, `docs/design/fashion-design-system-team.md` | Source evidence and independent rendered review have separate owners. |

The historical `docs/design/v2-remodel/reports/design-system-census.md` is useful
design context, but its old build/source/runtime findings are not current certification
facts. Phase 2 established the V2 CSS-to-editor authority. Do not replace that working
contract with the historical recommendation to rebase its tokens on V1.

`.wolf/memory.md` was absent when this contract was prepared. No substitute memory
file was created. This contract draws on repository sources rather than a new external
framework recommendation. Official API documentation lookups, if required by an
implementation decision, must record source URL and lookup date in the task evidence;
no such lookup or runtime claim is implied by this design specification.

## Brand thesis and recognition devices

The global shell is an Oakland fashion house organized like an exhibition index:
athletic typography, precise ordinal markers, framed but largely unboxed information,
restrained rose-metal details, and direct commerce access. Expressiveness comes from
proportion and hierarchy. As purchase intent increases, visual noise decreases.

Five recognition devices must work without the logo or identifying copy:

1. **Ordinal gutter:** two-digit indices occupy a narrow, consistent column beside
   substantial route names and a quieter secondary directory.
2. **Athletic/archival contrast:** Archivo display, readable Hanken text, restrained
   Anton accents, and system-monospace indices create different information roles.
3. **Rose-metal rule:** one thin house accent identifies entry, active navigation,
   and footer transition; no field of gold decoration.
4. **Asymmetric house directory:** clear primary routes oppose a smaller collection
   or service index, avoiding equal-size card rows.
5. **Architectural colophon:** a deliberate identity block and the existing Oakland
   origin line close the house without a generic giant brand wall.

Oakland identity must emerge through the existing brand and verified content. Do not
invent coordinates, civic symbols, landmarks, source stories, statistics, or claims.

## Canonical token graph

Preserve primitive aliases that existing pages consume. Add semantic aliases in the
same file and namespace, then use them in new shell components. Extend the existing
sync script for editor-appropriate values. Components must not branch on raw hex.

| Category | Primitive / proposed semantic mapping | Constraint |
| --- | --- | --- |
| Main text | `--sr2-color-text: var(--sr2-ink)` | Existing ink `#f5f5f0`. |
| Secondary text | `--sr2-color-text-muted: var(--sr2-muted)` | Existing `#b3b3b3`; measure context contrast. |
| Inverse text | `--sr2-color-text-inverse: var(--sr2-void)` | For light surfaces only. |
| Page surface | `--sr2-surface-page: var(--sr2-void)` | Existing `#0a0a0a`. |
| Raised / dialog | `--sr2-surface-raised`, `--sr2-surface-dialog` → card/card-hover primitives | Opaque surfaces, no glass dependence. |
| Commerce panel | `--sr2-surface-commerce: var(--sr2-card)` | A container only where information grouping needs one. |
| Editorial light panel | `--sr2-surface-editorial: var(--sr2-paper)` | Pair with inverse text; existing paper `#f7f3e9`. |
| Borders | `--sr2-border-subtle`, `--sr2-border-control` | Subtle rules are decorative; functional controls require measured contrast. |
| Action | `--sr2-color-action`, `--sr2-color-on-action` | Dark text on accent fill where verified; no unverified light-context accent text. |
| Focus | `--sr2-focus-inner`, `--sr2-focus-outer`, `--sr2-focus-width`, `--sr2-focus-offset` | Two contrasting rings; visible on imagery, light and dark surfaces. |
| Widths | content `1200px`, wide `1500px`, reading `60ch`, commerce form `42rem` | Synchronize mapped editor layout sizes; use available width on small screens. |
| Spacing | Existing quarter-rem scale; page gutter and section fluid primitives | Add role aliases, not repeated arbitrary measurements. |
| Controls | target `44px` minimum; input text `1rem`; existing primary height `48px` | Intrinsic growth for wrapping/localization. |
| Motion | micro → fast `180ms`; interface → normal `420ms`; editorial → slow `900ms` | Scene alias is a future grammar; no new scene implementation. |
| Easing | existing house ease `.16,1,.3,1`; existing dramatic ease for bounded expressive transitions | Exit may have a documented alias; no spring/bounce default. |
| Layers | content < floating < sticky commerce < header < scrim/drawer/nav < modal < toast < critical | Name every new layer; native top-layer dialogs remain authoritative. |

Current editor disagreements to resolve deliberately: V2 `theme.json` wide size is
1440px while runtime wide is 1500px; neutral editor surface/text/border entries are not
currently projected from runtime. Extend `sync-token-contract.cjs` to reconcile named
equivalents without erasing unrelated editor settings. Existing mapped accent, spacing,
font, motion and layer values must continue passing verification.

Collection contexts remain Signature gold, Black Rose silver, Love Hurts certified
crimson `#eb4666`, and Kids rose. House chrome keeps the house rose context; collection
accents belong to their relevant collection labels or content. Do not display several
competing accent treatments on a single shell surface. No collection palette redesign.

## Typography roles and font discipline

| Role | Family / valid weight | Intended scale | Other rules |
| --- | --- | --- | --- |
| Monument | Archivo 850 | `clamp(3rem, 7vw, 7rem)` | Line-height .94, tracking -.045em; one directory feature, not every heading. |
| Display | Archivo 750 | `clamp(2rem, 4vw, 4rem)` | Line-height 1, tracking -.035em. |
| Editorial | Archivo 650 | `clamp(1.5rem, 2.4vw, 2.5rem)` | Line-height 1.12; measure 24ch. |
| Commerce | Hanken Grotesk 600 | `clamp(1rem, 1vw + .75rem, 1.25rem)` | Line-height 1.35; product/price identity, normal case. |
| Body | Hanken Grotesk 400 | 1rem | Line-height 1.6; measure 60ch. |
| Utility | Anton 400 for short labels; Hanken 500 for values/forms | Labels .8125–.875rem; input 1rem | Tracking .08em on short uppercase labels only. |
| Index | `ui-monospace, SFMono-Regular, Menlo, monospace` | .75rem | Tabular numerals, tracking .04em; no font download. |

Introduce semantic role variables/classes without globally rebinding legacy page
heading selectors. These scales are implementation targets, subject to rendered fitting
and reflow evidence rather than a requirement to clip text.

Registered V2 fonts are Archivo, Hanken Grotesk, Anton, Cinzel and Inter: 211,100 bytes
of local WOFF2 files at baseline. This is a source-file census, not measured network
transfer. Archivo/Hanken declare weights 100–900; Anton declares400 only. Do not request
synthetic Anton bold. New shell uses no Cinzel display headings or script fallback;
leave existing page consumers intact until separately authorized page work.

Prohibited: Playfair Display, Cormorant Garamond, Bebas Neue, Yellowtail, external font
imports, type-rendered collection lockups, and European-maison serif direction. Keep
collection script artwork as artwork. No new font is necessary for this phase.

## Composition, grid and spacing

Use modern CSS Grid with a shared ordinal gutter and explicit primary/secondary
hierarchy. Header, directory and footer should align to the existing fluid page gutter
`clamp(1rem,4vw,4rem)`. A desktop navigation composition can allocate approximately
60–66% to primary routes and the remainder to the secondary directory; use minmax
tracks so intrinsic content controls shrinkage. Index column target: 2.5rem.

At small widths, primary routes precede secondary collection/service links in document
order. Reading and commerce widths constrain content only when space permits. Full-bleed
media slots remain available without forcing all content into a card. Use subgrid only
where a verified supported alignment problem justifies it; it is not a dependency.

Do not change homepage, collection, PDP or PLP composition beyond shell compatibility.
Do not apply new monument styling to all existing page headings.

## Global components and state contract

| Primitive | Required states | Behavioral owner / invariant |
| --- | --- | --- |
| Primary / secondary / quiet button | Default, hover, focus, pressed, disabled, loading | Native button/link semantics; no opacity-only disabled distinction. |
| Link / icon / close control | Default, hover, focus, active/current | Destination or accessible name visible/available;44px target. |
| Input / select / textarea | Empty, filled, focus, invalid, disabled, required | Native labels and error associations; 1rem text; no placeholder-only label. |
| Checkbox / radio | Checked, unchecked, focus, disabled, invalid | Keep native semantics and keyboard behavior. |
| Quantity / variation | No selection, valid, unavailable, loading, rejected | Native Woo values/IDs remain authoritative; do not replace controls. |
| Notice | Information, success, warning/error, recoverable failure | Real server state; no invented success; appropriate live-region behavior. |
| Chip / filter / pagination | Default, current/selected, focus, disabled | Preserve existing URLs/GET state; no new filtering backend. |
| Overlay | Closed, opening, open, closing, interrupted, nested attempt | One active owner; focus/scroll/cleanup lifecycle shared. |
| Bag shell | Empty, populated, updating, update failure | Server totals, identity and variation; native fallbacks remain available. |
| Navigation | Closed, open, current route, pointer/touch/keyboard, resize | Shop remains obvious; no animation-dependent links. |
| Search | Empty input, entered query, submit, close, invalid empty submission | Native GET `s`; repaired grouped results unchanged. |
| Mascot/guide | Available, overlay suppressed, commerce quiet, reduced motion | Never obstruct purchase, account or checkout controls. |

### Header and house navigation

Desktop header: Menu and direct Shop at the start; existing verified brand image as
middle anchor; Search, Account and Bag at the end. Use an opaque house surface and
stable dimensions. Do not hide important commerce access on scroll.

When this composition no longer fits, transform to Menu / brand / Bag; Search and
Account remain immediately available in the opened menu. Mobile is a separate
composition, not scaled-down desktop. No overlap at 320px or zoomed text sizes.

Keep the established 01–08 directory: Collections, Shop, Pre-Order, Journal, About,
Contact, Search, Account. Shop is visually equal to Collections. The secondary area
may retain verified existing collection references; no new asset substitution or
automatic cycling is required. Text links are a complete fallback.

Use practical WordPress menu configuration with existing route helpers as defaults.
Current/active state must be semantic as well as visual. Native links remain functional
without advanced motion, and a no-JS navigation path must be available.

### Living Archive search shell

Keep the existing Living Archive copy and visible, labeled input. The form remains
`method=get`, submits `name=s` to the native route, and preserves result-type semantics.
Use an index marker, compact collection shortcuts and a clear submit button. No new
predictive backend. The close control remains reachable with the virtual keyboard.

### Bag shell

Progressively present native Woo mini-cart content in the shared dialog/drawer only
where safe. Required: image, name, variation, quantity, removal, server subtotal,
View Bag and Checkout. Never calculate transactional totals independently in JS.

Keep the existing real Bag URL for no-JS/failure fallback. Quantity display must remain
truthful; native View Bag is an acceptable route for editing if in-drawer mutation would
require unnecessary new transactional endpoints. If a quantity control is implemented,
its update and error state must be server-confirmed with native identity preserved.

### Footer

Use two successive architectural spaces: a prominent house-route directory, followed
by a quiet client-services ledger and legal colophon. Retain the verified brand asset
and existing Oakland line in a compact identity area. Avoid equal multi-column SaaS
cards or a giant decorative wordmark.

Required accessible destinations: Shop, Collections, Journal, Client Services, FAQ,
Shipping + Returns, Size Guide, Contact, Account, Privacy, Terms and Accessibility.
Use WordPress-owned menus where practical with truthful helper fallbacks. Social and
newsletter appear only when configured; never invent accounts or subscription success.

## Overlay, focus and motion architecture

One coordinator owns navigation, search, bag, quick view, size guide and guide/chat
interaction boundaries. Record the opener; close an existing overlay before another
opens; contain focus; restore focus to a visible connected opener or documented safe
fallback; handle Escape once. Background main/footer/guide cannot remain interactive
under an active modal. Preserve previous inert state when restoring it.

Use one shared scroll-lock strategy with scrollbar compensation, safe-area padding,
orientation/resize handling, rapid transition cancellation and route cleanup. Never
leave body fixed/locked after closing. Native dialog top-layer behavior must not be
simulated with arbitrarily large z-index. Respect no-JS links and native forms.

Focus must remain clearly visible on light, dark, imagery and commerce surfaces.
Use a two-tone ring; test forced colors and do not remove the outline without an
effective replacement. Hit targets are at least 44px and grow with wrapping text.

Micro/interface/editorial motion aliases extend existing 180/420/900ms primitives and
house ease. Global transitions should use a small meaningful panel translation and
opacity, not generic fade-up choreography. Reduced motion presents the final stable
state immediately, pauses decorative video and uses static brand imagery. No essential
content begins permanently hidden. No added animation library or WebGL in this phase.

## Media rules

Do not change the approved/stale/rejected/missing state classifications or the Phase 2
resolver hierarchy. Preserve explicit rejection blocking fallback, schema and variation
images. Keep native extension hooks and empty-media handling intact.

Reuse verified brand assets and already bound collection references only. Preserve
intrinsic dimensions, aspect ratios, meaningful alt semantics, responsive loading and
object-position intent. Product images use contain behavior where product identity
requires full visibility; editorial imagery may crop only within its established role.
Do not copy an image merely because its filename sounds appropriate. No generation,
upload, promotion, catalog edits or registry hand editing is authorized.

## Responsive and verification matrix

Widths are test points, not a request for ten bespoke breakpoint patches. Use fluid
type/spacing and intrinsic layout; choose the actual transformation breakpoint from
content fitting and record it. Capture actual CSS viewport metrics in evidence.

| Width | Required shell behavior / verification |
| --- | --- |
| 320 | Minimum reflow; Menu/brand/Bag fit; one-column directory; full-width dialogs; visible close/submit. |
| 360 | Small mobile touch targets, input/keyboard and safe-area behavior. |
| 375 | Mobile cart/account controls and long labels. |
| 390 | Primary mobile screenshots and full interaction/commerce regression. |
| 414 | Wider mobile: intrinsic layout, no unnecessary desktop transformation. |
| 768 | Tablet portrait: route order, menu open resize, footer wrapping, dialogs. |
| 1024 | Tablet/compact desktop: header fit and coarse-pointer behavior. |
| 1280 | Desktop primary/secondary directory hierarchy and scrollbar compensation. |
| 1440 | Primary desktop screenshots, keyboard sequence, typography/material review. |
| 1728+ | Bounded content widths, no stretched controls or oversized whitespace; verify at an actual recorded width. |

| Route / state | 390 | 768 | 1440 | Evidence |
| --- | --- | --- | --- | --- |
| Home shell closed | Required | Required | Required | Baseline comparison; page art direction unchanged. |
| Navigation open | Required | Required | Required | Keyboard, touch, Escape, opener restoration, resize/orientation. |
| Search open | Required | Required | Required | Labeled native GET, close, submit, result grouping regression. |
| Bag empty/populated | Required | Required | Required | Correct server identity/variation/quantity/subtotal, native links. |
| PDP with shell | Required | Required | Required | Native size selection/add; authority state intact. |
| Checkout shell | Required | Required | Required | Quiet controls, no obstruction; no payment submission. |
| Account logged out | Required | Required | Required | Native fields, labels, focus and error state. |
| Footer | Required | Required | Required | Links/ownership, hierarchy, reflow. |
| Reduced motion / overlay error/loading | Required | Required | Required | Stable visibility, semantic state, no stuck locks. |

Additional interaction checks: pointer, keyboard, touch/coarse pointer, rapid open/close,
nested overlay attempts, resize while open, orientation change, Back navigation, 200%
zoom/reflow, focus on imagery, forced colors, and long/translated labels. Timeouts,
unavailable browsers or missing evidence are UNVERIFIED, never PASS.

## Phase 2 preservation and regression acceptance

Keep native purchase controls, 185 variation identities and real size semantics, cart
identity, prices, stock truth, catalog/SOT and generated-registry contracts, repaired
account shortcode behavior, checkout hooks/status messaging, search URLs/type grouping,
media authority and reduced-motion behavior. No presentation flag authorizes payment,
reservation, allocation or fulfillment claims.

After shell changes, execute local PDP → valid size → Add to Bag → cart → checkout
shell. Record exact product, variation, quantity, unit price and subtotal. Verify no
duplicate addition on rapid interaction and no state loss on refresh. Verify logged-out
account, search grouping, and representative approved/stale/rejected/missing media
semantics. Test data is isolated fixture state, not evidence about staging inventory.

For every substantial component batch: build, verify V2, run relevant regressions,
inspect generated drift, render desktop/mobile, inspect captures, refine, check keyboard,
reduced motion and console, then commit. Do not build the whole shell without inspection.

## Performance, distinctiveness and release gates

Record before/after source and served CSS/JS bytes, font transfer, critical image
behavior and observed long tasks using the same local routes/viewports. No added font,
animation library or WebGL payload is expected. Record all growth and its cause; no
claimed performance pass without measured comparable evidence. Existing page media
cost is a baseline limitation, not authorization to redesign those pages here.

Hard failures: generic centered headline/paragraph/two-pill gradient heroes, equal
icon-card rows, monotonous new product-grid design, default SaaS typography, arbitrary
glass, gradient text, universal rounded cards, hierarchy-free bento, decorative blobs,
fake metrics/scarcity, uniform scroll reveals, European-maison serif direction, retired
fonts, multiple accents, type-rendered hero scripts, unverified garments, urgency timers
or pressure copy. Scan changed shell scope and separately record untouched page debt;
do not silently expand into page redesign to clear unrelated findings.

Independent visual review scores brand recognition without logo/copy 20, composition 20,
typography 15, garment protagonism 15, token/material discipline 10, state coherence 10,
motion/responsive translation 10. Approval requires at least 85/100, every category at
least 70% of available points, zero hard failures and zero unverified claims. Garment
protagonism is assessed on the representative PDP/collection pages with the new shell,
not by forcing new garment imagery into utility chrome. The author cannot approve pixels.

Current score and logo-off verdict: **UNVERIFIED in this contract**. The lead owns
independent-review dispatch and final evidence integration; no final reviewer verdict
is asserted here. Baseline and candidate captures belong to
`.artifacts/v2-phase3-20260905/`; the final certification must list the actual files,
viewport metrics and measured accessibility/performance results. A capture directory
is not itself verification. No threshold may be lowered to obtain acceptance.

Carry forward Phase 2 launch blockers: known full-package/staging parity boundary,
Klaviyo blocked-script/integration uncertainty, authenticated account flows, completed
payment/provider flows, inventory/shipping unknowns and existing media approval deficits.
The legacy forced product-sync overwrite risk remains outside this design task.

**Builder handoff: APPROVED for local source implementation.** The lead recorded that
the certified baseline suite passes and home/navigation/search baseline snapshots were
captured at actual 1440/390 viewports before authorizing implementation. Their detailed
runtime/capture receipts belong to the task evidence. This is an implementation gate,
not pixel or release approval. Final design approval remains **BLOCKED** until independent
rendered review and the required regression evidence pass.

### Foundation implementation record

The semantic graph and native-control layer are implemented in `design-tokens.css`,
new `assets/css/controls.css`, the existing `sync-token-contract.cjs`, and generated
`theme.json`. Existing primitive values, font assets and collection accents are unchanged.
Editor neutral/layout projections and semantic role settings now derive from runtime.
Editor accent buttons use dark text to match the verified storefront contrast contract.

Seven tests in `tools/v2-runtime/test-token-system.cjs` verify idempotence/unrelated
extension preservation, runtime-change propagation, missing/cyclic alias failure,
normal-size action/text contrast plus control-boundary contrast, focus/layer ordering,
collection-local action/focus inheritance, and light-context text/disabled-state contrast.
All seven passed after the native-control compatibility fixes. These source checks do not replace browser state/contrast
verification. CleanCSS parsing of both changed CSS sources produced zero errors/warnings;
the coordinator owns actual generated assets, enqueue order and final byte measurement.

The control stylesheet is designed to load after existing theme CSS; the lead owns
enqueue/build verification. It styles native Woo fields,
buttons, quantities, notices, pagination and form labels without changing purchase
semantics or page compositions. Shared classes include `sr2-control` with primary,
secondary and quiet variants, `sr2-icon-button`, `sr2-field`, `sr2-notice`, `sr2-chip`,
`sr2-pagination`, `sr2-index-label`, `sr2-product-meta`, `sr2-reading` and
`sr2-editorial-grid`. A `sr2-surface--light` context reverses text/focus tokens without
changing the house accent. Native controls receive visible focus, meaningful disabled
and invalid states, forced-colors treatment and reduced-motion fallbacks.

### Native-control compatibility record

The existing Woo password visibility button is retained and enlarged to the canonical
44px square target, centered within its native wrapper. Input padding reserves room
for the button, and its existing state-dependent icon is made visible on dark surfaces.
No new password visibility handler or data flow was introduced.

Woo Select2/SelectWoo selected text, placeholder, caret in both open and closed states,
dropdown, options, search field, disabled treatment and focus now consume semantic
control tokens. Dropdown scope includes the Woo page because Select2 mounts dropdowns
outside the form. Forced-colors overrides are included. This repairs observed native
compatibility gaps without replacing country/state selection behavior. Seven token tests
and CleanCSS parsing passed; final rendered account/checkout verification remains the
lead's responsibility.

### Global-shell implementation record

Source census at this update confirms:

- `inc/global-shell.php` owns reusable header, menu fallback, bag shell and footer
  rendering. `assets/css/global-shell.css` owns their layouts; `theme.js` owns the
  shared overlay lifecycle. The existing PHP/vanilla architecture remains in use.
- Header has desktop direct Shop/Search/Account/Bag access, an existing static brand
  asset, and a mobile Menu/brand/Bag transformation at 48rem. The directory uses
  numbered primary routes, smaller utility links, and existing collection preview
  references. Source references alone do not grant new media approval.
- Primary and footer menus use `wp_nav_menu` when configured with working route-helper
  fallbacks. Client-service links and legal destinations remain explicit and translated.
  No newsletter or social service was fabricated.
- Search retains a labeled native GET form and collection shortcuts. The native Woo
  mini-cart supplies bag contents, variations, quantity display, removal, subtotal,
  View Bag and Checkout. Quantity editing remains on the native Bag page; no new
  in-drawer mutation endpoint or client-side totals calculator was introduced.
- The bag dialog is omitted on checkout, preserving the real Bag link. Checkout also
  suppresses the expressive footer directory and desktop direct-navigation links to
  reduce noise. The service/legal footer remains available.
- No-JS directory fallback remains in document flow. The overlay coordinator has
  navigation/dialog ownership, opener tracking, scroll lock, inert restoration and
  pagehide cleanup. Source presence is not proof of all interaction states; the lead
  is running the final browser matrix and regression suite.

### Contract reconciliation items for final certification

Two source observations from the documentation census were resolved before certification:

1. The header now clears legacy `is-hidden` state and retains only scroll styling. Shop/Bag access remains stable.
2. Fallback links use WordPress queried-object identity (and Woo shop identity) for `aria-current="page"`; shared links underline the current route. Configured menus retain WordPress ownership.

The source review approved the local implementation with 27 scoped Node tests passing.
Final executed browser, payload and independent visual results live in the Phase 3A
report and task evidence. This contract confers no founder approval, deployment
permission or Phase 3B authorization.

Stop after the user-requested Phase 3A certification report. Phase 3B homepage,
collection-world, PLP/PDP or journal art direction requires the next authorization.
