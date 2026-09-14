# SkyyRose V2 Phase 3A Design System & Global Shell Report

## Status

**Phase 3A implementation is complete as a reproducible local candidate. Overall
readiness is CONDITIONAL.** Functional, build and scoped accessibility checks pass;
independent visual review passes at **88/100**. Mobile loading performance remains
poor, and the accepted Phase 2 launch boundaries remain open.

The isolated preview is `http://127.0.0.1:18303/`, using WordPress 7.1, WooCommerce
11.1.0 and Docker MariaDB 11.4. It contains synthetic products and variations, with
mail, outbound requests, webhooks and payment gateways disabled. No staging or
production files were deployed. No order or payment was submitted. Phase 3B has not
started. The fixture is for local verification, not a representation of live inventory.

## Source

| Identity | Value |
| --- | --- |
| Certified Phase 2 baseline | `5de8e2f3eb40827a052996f72bfb290a95bd600a` |
| Initial checkout, evidence-only descendant | `42ccb4b557c7e580956eb347470233dfc0bd4f6a` |
| Dedicated branch | `codex/v2-phase3-design-system-shell-20260905` |
| Shell implementation | `69249828bc77d37d77223f57fdef1e0a0c36cd0b` |
| Final runtime candidate | `75ce80b90380d0e5915be79dab33484eda6d3fca` |

The accepted baseline reproduced before implementation and its history was preserved.
This report is committed separately as evidence; the package is bound to the final
runtime commit above. Both builds from a clean detached checkout left it clean.

## Design System

Extended the existing CSS custom-property system into semantic color, action, focus,
surface, control, spacing, grid, layer and motion contracts. Seven type roles—monument,
display, editorial, commerce, body, utility and index—use existing project fonts and
intentional fallbacks. Fluid scales and intrinsic grids support reading, editorial and
commerce layouts without creating a second frontend framework.

The existing CSS-to-`theme.json` adapter now checks type, layout, spacing and controls
as well as colors. It fails on missing or cyclic token aliases. Four collection
contexts retain their identity underneath one house system; light-context controls
have separately verified contrast. Existing font binaries remain unchanged.

The enforceable contract is `docs/design/skyyrose-v2-phase3a-design-system.md`.

## Global Components

Added focused PHP shell composition in `inc/global-shell.php`, new global-shell and
control stylesheets, and one overlay coordinator in the existing vanilla `theme.js`.
Buttons, icon/close controls, fields, selects, textareas, checkboxes, radios, quantity,
variation controls, chips, pagination and native Woo notices share semantic states.
Native form and transaction semantics remain authoritative. Source/minified assets,
editor configuration and translation output are synchronized.

## Header

The stable opaque header provides direct desktop Shop, Search, Account and Bag access,
plus Menu and the house mark. Mobile has its own Menu/mark/Bag composition, with
Search and Account in the menu. Legacy downward auto-hide was removed to keep purchase
and navigation access predictable. Existing WordPress menu ownership is retained.

## Navigation

The indexed house menu exposes Collections, Shop, Pre-Order, Journal, About, Contact,
Search and Account. Desktop pairs routes with four existing collection-image previews;
mobile stacks the composition. Existing collection route helpers are used. Fallback
routes derive current-page semantics from WordPress identity, including the native shop.

Pointer, keyboard, touch/coarse pointer, rapid cycles, resize, orientation, Escape,
focus return and history were exercised. The delayed-script regression proves the
menu stays hidden before initialization without causing a layout flash. With JavaScript
disabled, the navigation is visible in document flow and native links remain usable.

## Search

The Living Archive shell uses a native accessible dialog around the preserved GET
search form. Input focus and Escape return are verified, including Menu-to-Search
handoff. Native search URLs and exclusive Products/Stories grouping remain intact.
The search backend was not replaced.

## Cart

The Bag drawer progressively enhances the real cart link using the native WooCommerce
mini-cart and fragment events. Product identity, image, variation, quantity, removal,
subtotal, View Bag and Checkout come from WooCommerce. Quantity editing remains on
the native Bag page; JavaScript does not calculate totals. Checkout uses the quieter
native page surface rather than a second cart drawer.

Fragment replacement repairs focus only when the focused control is lost or hidden.
Confirmed native removal messages are relayed into an in-dialog status region. A
delayed removal preserves the item and avoids premature success; an injected HTTP 503
uses WooCommerce's own nonce-bearing GET recovery, then verifies an empty unlocked cart.

## Footer

An indexed house section, client services and Oakland colophon create the closing
composition. Shop, Collections, Journal, FAQ, shipping/returns, size guide, Contact,
Account, privacy, terms and accessibility remain discoverable. WordPress menus keep
links maintainable; one footer-house menu location was added. No unconfigured social
or newsletter integration was invented.

## Overlay System

Navigation, Search, Bag, quick view, size guide and Ask Skyy share one lifecycle:
active state, background inertness, scroll lock, focus containment/return, Escape,
stacking and cleanup. Existing native dialogs are integrated without monkey-patching
their methods. Body styles, scrollbar compensation and scroll position are restored;
the measured orientation test returned from 600px to 600px.

Independent supplementary checks passed ten desktop/mobile overlay states, three
valid repeated Ask Skyy focus-return cycles and an actual pointer hit on the size guide.
The guide retains its identity and is suppressed behind active overlays. A reduced-motion
visibility transition that interfered with focus restoration was fixed at its CSS source.

## Motion

Defined durations, easing, distance and stagger tokens for subsequent work. Global
motion is restrained and reduced-motion behavior is explicit, including instant
scroll behavior and existing paused media. No new animation library, framework,
WebGL scene or signature collection animation was introduced.

## Responsive

Verified 40 shell/overlay observations across **320, 360, 375, 390, 414, 768, 1024,
1280, 1440 and 1728px**, plus 21 representative route observations at 390, 768 and
1440px. There was no document-width overflow in the checked states. At 390px and
200% root text enlargement, menu scroll width equals its 390px client width.
Safe areas, native touch targets and orientation changes are included in the shell.

## Accessibility

**18 Axe scans: zero violations.** Manual/browser probes cover keyboard containment,
focus return, labels, landmarks, heading structure, required fields, native password
visibility, dynamic cart status, forced colors and reduced motion. Password reveal
and size-guide targets are 44px. Native SelectWoo country/state controls received
compatible text, caret, dropdown and focus styling.

Lighthouse accessibility scored 100 in all compared cases. These are scoped automated
and interaction results, not a blanket WCAG conformance certification or a completed
assistive-technology/device matrix. Real-device Safari and human screen-reader testing
remain outside the observed evidence.

## Performance

**Performance is not PASS.** Measurements use the same isolated fixture and blocked
external services; they are lab observations, not field Core Web Vitals.

| Theme asset measure | Phase 2 | Phase 3A | Change |
| --- | ---: | ---: | ---: |
| Minified CSS bytes | 237,165 | 267,962 | +30,797 (+13.0%) |
| Per-file gzip CSS estimate | 43,367 | 49,493 | +6,126 |
| Minified JS bytes | 58,809 | 62,287 | +3,478 (+5.9%) |
| Per-file gzip JS estimate | 21,373 | 22,481 | +1,108 |
| Observed font bytes | 211,100 | 211,100 | 0 |

Observed served JavaScript increases by 6,417 bytes including the native Woo cart
fragment dependency. No new font or animation-library payload was added. Existing
large page styles and image/media bytes remain unchanged.

| Lighthouse 13.4.1 case | Performance before → after | LCP before → after | CLS before → after |
| --- | --- | --- | --- |
| Home mobile | 58 → 56 | 11.41s → 12.01s | 0.158 → 0.157 |
| Home desktop | 84 → 84 | 2.65s → 2.57s | 0.0108 → 0.0079 |
| PDP mobile | 62 → 61 | 12.08s → 12.39s | 0.0327 → 0.0323 |

The initial navigation startup flash produced a serious layout shift; it was repaired
and specifically regression-tested. The remaining mobile home shift is present in the
unchanged Phase 2 hero. Mobile LCP remains poor in both versions, with a small candidate
regression that must remain visible in the release assessment. TBT is 0–3ms in these runs.

Three cold samples per route/width (24 total) gave median local load observations:
home 390px 462→468ms, PDP 390px 437→445ms, home 1440px 448→462ms, PDP 1440px
416→468ms. Recorded long tasks were 11→12, maximum 86→106ms. This small localhost
sample cannot establish field impact or attribute all timing differences to the shell.
Critical viewport image readiness was asserted by the browser capture suite; the
performance sampler also lists horizontally offscreen lazy filmstrip images, which
must not be misreported as broken critical media.

## Commerce Regression

The real local Woo flow passed: SG-005 → size M → quantity 1 → Add to Bag → native
cart → checkout shell, preserving variation 182 and unit/subtotal 25.00 in the fixture.
Empty cart has one primary heading. Pending removal and native error recovery passed.
The seven order-state PHP sentinel cases still execute native gateway hooks once,
without duplicate details or unsupported payment assurances.

The complete clean-path browser gate reports zero console/page errors and zero HTTP
errors. Three navigation-aborted requests and one intentionally blocked external embed
are recorded separately. Local PHP observation found no fatal errors or warnings;
fixture provisioning had 477 PHP deprecations and two early Woo translation notices.
This does not certify the remote PHP/plugin runtime or completed payment lifecycle.

## Account/Search Regression

My Account login renders with native required fields and working password reveal at
the checked widths. Native search GET/history and result-type separation pass. Logged-in
dashboard, order details, address changes, saved payment methods and logout remain
unverified with authorized customer test credentials. No account mutation or live
customer session was used for this phase.

## Media Integrity

Protected V2 data, SOT assets, WooCommerce override templates and the canonical catalog
are unchanged from the accepted Phase 2 commit. Integrity verification preserves 33
front hashes/dimensions, 18 motion variations and existing casts/posters/registry pins.
Opening-media state remains **16 stale / 9 missing-front / 5 rejected / 3 approved**.
Representative approved, stale, rejected and missing PDP behavior was exercised.
No media was generated, promoted, repinned to bypass a gate or assigned new approval.

## Screenshots

All local captures are in `.artifacts/v2-phase3-20260905/`. `screenshots.json` records
70 baseline/final/supplementary files with dimensions and SHA-256 hashes. Pixel dimensions
may exceed viewport height for full-page captures; browser receipts preserve actual
viewport metrics. Representative comparisons:

| State | Baseline | Final |
| --- | --- | --- |
| Home, desktop/mobile | `baseline-home-{1440,390}.png` | `final-home-{1440,390}.png` |
| Navigation | `baseline-nav-{1440,390}.png` | `final-nav-{1440,390}.png` |
| Search | `baseline-search-{1440,390}.png` | `final-search-{1440,390}.png` |
| Native cart / Bag drawer | `baseline-cart-{1440,390}.png` | `final-bag-{1440,390}.png` |
| PDP | `baseline-pdp-{1440,390}.png` | `final-pdp-{1440,390}.png` |
| Checkout mobile | `baseline-checkout-390.png` | `final-checkout-390.png` |
| Footer | `baseline-footer-{1440,390}.png` | `final-footer-{1440,390}.png` |

Final 768px shell/checkout captures, all three checkout widths, empty/loading/recovery
Bag states, forced colors, enlarged text, quick view, size guide and Ask Skyy are also
retained. Independent review is recorded in `independent-review.md`.

## Build

**PASS:** certified build, V2 verification, 27 Node regressions, seven Python integrity/
i18n tests, PHP syntax and preserved media/checkout/search contract tests. Eight CSS and
eight JS generated assets, POT, token/editor parity and runtime manifests verify.

Two full package builds from a clean detached checkout at the final runtime commit
produced identical manifests and byte-identical ZIPs, with **379 runtime entries**.
No generated source drift remained. Toolchain: Node 22.23.2, npm 10.9.8, Python 3.12.12,
PHP 8.5.6; QA Playwright 1.58.2, Axe 4.11.1, Lighthouse 13.4.1.

Archive: `.artifacts/v2-phase3-20260905/skyyrose-v2-phase3a-75ce80b90.zip`.

SHA-256: `48aa6e3084c5544eae79daaf662d940d95e41808c9123ad1758818dc94f5838f`.

`build-certification.json` records clean-state proof and log hashes;
`release-manifest.json` binds all package entries, source and toolchain. Its
`deployment_authorized` value remains false. The build emits a harmless Node
NO_COLOR/FORCE_COLOR warning. No unrelated monorepo-wide certification is asserted.

## Remaining Launch Blockers

- Mobile LCP and remaining hero/font-swap layout shift require performance work and
  representative-device validation before launch readiness can be claimed.
- Accepted Phase 2 whole-package/staging parity was 362/374 matches with 12 known
  untouched differences. This phase does not re-audit or authorize that boundary.
- Klaviyo checkout ORB/browser blocking and unresolved integration behavior remain open.
- Authenticated account workflows and provider redirect, capture, settlement, refund
  and completed-payment behavior are not certified.
- Physical inventory/size truth, shipping and fulfillment configuration need separate
  verification; synthetic local stock is not business authority.
- Existing media approval deficits and the legacy forced product-sync simple-payload
  overwrite hazard remain. Database-authored copy was not comprehensively audited.
- Local noindex/isolation affects SEO results (Lighthouse 58); this is not a production
  SEO or third-party integration assessment. The external About embed was blocked.

## Phase 3B Readiness

**Conditional technical readiness for a separately authorized next phase.** The
reusable system and shell are implemented, reproducible and independently reviewed.
Performance and carried-forward launch gates must stay explicit. Nonblocking page
refinements include narrow tablet checkout summary, account spacing, legacy guide
typography and extreme text-enlargement wrapping.

The stop boundary is active: no flagship homepage, collection-world, PLP/PDP redesign,
Town Line, Black Rose Salon, The Heir, Journal art direction or advanced transitions
have been started. The next action requires the next scoped authorization; this report
does not grant staging promotion, production deployment, launch or payment approval.
