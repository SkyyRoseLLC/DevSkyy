# SkyyRose V2 Phase 2 Runtime Certification Report

## Status

**CONDITIONAL PASS — Phase 2 runtime and commerce repairs are verified on staging.** The clean canonical source builds reproducibly. This is not a production release or full-package deployment certification. No Phase 3 redesign began, no production files changed, and no payment was submitted.

Conditions: 12 untouched, fully identified package/staging differences remain from the accepted source-normalization phase; Klaviyo's checkout script is browser-blocked; authenticated account and completed payment flows were not exercised. These boundaries must remain visible through subsequent release work.

## Baseline

- Accepted engineering baseline: `ddb3dce6db82d69010f8d15fee4fbb6b99661066`.
- Certified repair source: `5de8e2f3eb40827a052996f72bfb290a95bd600a`.
- Branch: `codex/v2-phase2-runtime-20260905`.
- Theme: `skyyrose-flagship-2`, version `2.4.4`.
- Preserved recovery `bb0e6eecf0af54b3f0c14e5eadb7b138530ce09b` and normalized candidate `53ae72dc339d1130b2ce8c2dd2d18108569e1efe` were not changed.

All staging writes checked the exact staging home URL, active theme and known file preimage, saved rollback bytes, and verified the resulting hash. No bulk theme replacement was performed.

## Runtime

The homepage requested deferred jQuery core while the optimizer emitted blocking jQuery Migrate and a blocking Woo/blockUI bundle. WordPress's calculated dependency eligibility and the optimizer's emitted markup disagreed. Removing the theme-owned homepage defer request was the smallest controlled correction; other safe optimization remained. Cached canonical HTML required a guarded staging cache flush.

Repeated navigation and the final 36 route/viewport checks showed one registered, blocking jQuery core and no captured recurring project-owned JS exceptions. The native variation path exposed an additional CSP defect: the installed Woo/Underscore template path attempted dynamic compilation. A narrow cached renderer for Woo's two standard variation templates avoids eval without relaxing CSP, replacing Woo variation IDs, or modifying vendor files. Purchase readiness now waits for a real native variation ID.

Evidence: gate-2.1.md, variation-runtime-repair.md, final-browser-matrix.json, final-console.json.

## Account

The actual My Account page contained placeholder content. A separate guarded, idempotent content migration installed `[woocommerce_my_account]` through the native Woo page assignment; no environment-specific page ID entered theme logic. Existing registration-disabled and guest-checkout settings remain.

Logged-out login, invalid-login rejection, lost-password form, anonymous endpoint routing and mobile/desktop layout were verified. No authorized test credentials were available; authenticated dashboard, order details, addresses, payment methods and logout remain unverified. No password-reset email was sent. Exact preimage and rollback are recorded in the account migration evidence.

## Product Model

All **33 canonical SKUs** were reconciled against canonical garment sizes, current attributes, product type, stock state and intended purchase model:

- **31 SIZE REQUIRED** products now use native variable-product semantics.
- **2 NO SIZE REQUIRED** products remain simple: LH-005 and SG-007.
- **0 UNKNOWN** classifications remain.
- Two noncanonical draft products remain untouched.

See product-model-reconciliation.md/.json. The presentation registry remains separate from live Woo purchasing authority.

## Variation Migration

The migration preserved existing parent IDs, SKUs, URLs, aggregate prices, media/gallery relationships, categories and stock truth. It created **185 real Woo variations**, each carrying its native size attribute and existing price. Parent stock had no managed physical quantity; no per-size quantities were invented. LH-006's incorrect One Size attribute was replaced with its canonical S–3XL range.

Before mutation, product configuration, meta/terms, stock and order references were exported. Dry-run plans, transactional table checks, per-product journals, expected-state hashes, idempotency and rollback dry-run were verified. Rollback removes only migration-owned children and restores the saved simple-product configuration, provided no state drift or order references have appeared. Actual rollback was not executed.

Product webhook delivery was suppressed within the migration process; immediate outbound HTTP was blocked there. Persistent gateway/webhook configuration was not disabled. General third-party asynchronous side effects were not comprehensively audited.

All 185 variations passed native WC_Cart checks for parent ID, variation ID, size, quantity and price. Full baseline exports remain under `.artifacts/v2-phase2-20260905`; tracked backup hashes and migration receipts identify them.

## Media

The opening-media states remain **16 stale / 9 missing-front / 5 rejected-authenticity / 3 approved**. No image was generated, uploaded or promoted.

PDP resolution prefers valid approved editorial attachments, then valid assigned Woo commerce imagery. Explicit authenticity rejection blocks the fallback, schema image and variation-image payload. Genuine missing media uses a compact status. All before-summary extension callbacks still run; only the native gallery callback is temporarily withheld when no permitted media exists and then restored.

Approved BR-006, stale SG-005, missing-front BR-002 and rejected BR-003 were exercised. Loaded responsive images and schema selection were checked; sentinel hook callbacks fired once and scoped image filters/hooks were restored. This is PDP media certification, not blanket approval of existing card/cart thumbnails or image authenticity.

## Pre-Order

No transactional pre-order engine was verified. Registry/product flags and static edition metadata do not establish reservations, allocation, delayed capture or shipping dates. Stripe is configured with testmode=no and capture=yes; gateway settings were preserved.

Removed unsupported secured-place/reservation/date assurances from theme-owned home, PDP, pre-order and service copy. The PDP explicitly explains standard checkout, no stock reservation or deferred payment from the label, and contacting Client Services for shipping estimates. No new pre-order engine was built. Legacy database-authored content outside these templates has not received a complete copy audit.

## Checkout Truth

Removed “Nothing was charged.” Failed, pending, on-hold, processing, completed, cancelled and refunded orders now receive distinct conservative messages. Native retry URLs remain; before-thankyou, gateway-specific and generic thankyou hooks run once. Removed a manual order-details include that duplicated Woo's registered order-details callback.

Seven unsaved WC_Order states (ID 0) and local template regression tests passed. No persisted order or gateway transaction was created. The cart's separate quantity defect was also fixed: KSES had stripped native quantity inputs; native Woo controls now render and quantity 1→2 correctly produced a $50 subtotal during verification.

## Search

Native GET search and pagination remain. One bounded main query supplies exclusive Products / Collections / Stories–Journal / Pages groups; a single exact SKU lookup can supplement page one, deduplicated by ID and restricted to visible, published parent products. Native queries are capped at 24 entries, plus at most one exact-SKU result.

`rose`, `sg-005`, `Love Hurts`, `journal` and a nonsense query were exercised. Products no longer appear in generic story/page results. Builder-generated page excerpts leaked CSS; page/collection excerpts are omitted while journal excerpts remain. Long queries wrap and the fixed header has clearance.

## Mobile

At 390px, navigation now isolates background focus, wraps keyboard traversal and restores Escape focus. Native dialogs coordinate with navigation and one another, own scroll lock, and respect viewport/safe-area limits. Search, quick view and size guide retain native modal behavior and visible focus return.

Skyy's mobile controls are in the page flow before the footer, with no proactive walk-on or translating entrance. They cannot cover headings or commerce CTAs. Guide controls become hidden/inert during navigation; delayed minimize focus does not steal focus after the user moves elsewhere. The mascot artwork and collection design were preserved.

Native SG-005 purchases at actual 390px verified size, quantity and price, followed by removal. All final route widths matched document scroll widths at 390/768/1440. See gate-2.8.md and mobile-overlay-browser.json.

## Tokens

Existing `assets/css/design-tokens.css` is the explicit primitive authority. The build derives mapped theme.json settings and verification rejects drift. Love Hurts editor crimson now matches existing frontend `#EB4666`; the frontend palette was not redesigned.

Mapped spacing, Archivo/Hanken Grotesk/Anton/Cinzel identifiers, 180/420/900ms timings, house easing, and guide/header/skip layers are explicit. WordPress's theme JSON resolver and browser computed values were verified. Native dialogs retain browser top-layer ownership.

## Build

A new detached clean worktree at the certified repair SHA passed:

1. Pinned dependency install: Node 22.23.2, npm 10.9.8, Python 3.12.12 with the certified environment. npm reported zero vulnerabilities for the theme's 13 audited packages.
2. Production registry/assets/i18n/token build.
3. Full V2 verification: source integrity, protected scenes/media, marketplace checks, PHP behavior/lint, 10 Node regressions, and token parity.
4. Deterministic package validation: **374 runtime files**, **208,157,493 bytes**.
5. A second build/package with byte-identical ZIP and zero Git drift.

ZIP SHA-256: `72901ec0708258dc14596245f7ec2cbf458ffe46f27c76af93508300ea06a50c`.

Artifact: `.artifacts/v2-phase2-20260905/skyyrose-flagship-2-phase2.zip`. Release manifest and clean build logs sit beside it. `final-build-certification.json` identifies the tested source and exact results. The report/evidence commit after this SHA changes no runtime source.

## Commerce E2E

- **22 recorded browser flows:** two sizes for each of 11 representative garment categories (crewneck, joggers, jersey, hoodie, jacket, shorts, bomber, sweatpants, set, kids hoodie set and shirt).
- **185 native WC_Cart variation validations**.
- Missing size, invalid size/variation, rapid repeated click, refresh persistence, quantity update and mobile purchase checks.
- Final flow: Signature collection → SG-005 → M → native variation **10484** → Add to Bag once → cart quantity **1**, subtotal **$25** → checkout correct M variant, quantity1, subtotal/observed total **$25**. The displayed shipping result applied to that session; it is not a universal shipping/tax quotation.
- Native checkout order-review AJAX returned **HTTP 200**. No payment submission. Final test item removed; cart confirmed empty.

## Browser

**36 route/viewport observations:** Home, Shop, Signature, Black Rose, Love Hurts, Kids Capsule, SG-005 PDP, rejected BR-003 PDP, Cart, populated Checkout, My Account and Search, each at **390 / 768 / 1440px**. Menu and search interactions were tested separately. Screenshots are in `.artifacts/v2-phase2-20260905/final-*.png`; additional reduced-motion and repair screenshots are retained there.

These are runtime/responsive baseline checks, not Phase 3 visual approval, cross-browser certification or a performance score.

## Accessibility

Verified keyboard menu containment/Escape, search and dialog focus lifecycle, native size selection (S → variation10483) and traversal to Add to cart, cart controls, account username→password, and checkout email→first-name focus. Visible focus styling remained.

Reduced-motion checks on Home, Signature and PDP confirmed the preference, 1ms motion primitives, paused/non-autoplay video and static brand imagery. No screen-reader audit or complete WCAG conformance assessment is claimed.

## Console/Network

No captured project-owned console exceptions across the final route matrix; every route emitted one registered jQuery core. No broken images were observed in those load checks.

The initial network observer became inactive, so its empty results were explicitly superseded. A separate observer-verified pass captured **647 requests / 634 responses across 12 routes**, with **zero observed HTTP error statuses**. Six video requests were intentionally cancelled/aborted; they are distinguished from failed asset responses. Native checkout review AJAX returned 200 without an observed failure loop.

One unresolved third-party failure remains: `static.klaviyo.com/onsite/js//klaviyo.js?...` is blocked with `net::ERR_BLOCKED_BY_ORB` on checkout. Its configuration/root cause and marketing-event delivery are not certified. Network evidence is a bounded load/interaction sample, not a prolonged soak test.

## Known Limitations

- **Whole-package staging byte parity is not claimed.** All **22 Phase 2 runtime files** match staging. **362/374** packaged files match staging; **12 untouched pre-existing differences** all match the preserved recovery commit. They comprise nine minified assets, the approved hero-font fallback cleanup, semantically identical registry JSON key ordering, and the POT. See preexisting-package-differences.json. No unexplained new runtime drift was found, and the full ZIP was not deployed.
- Source-only package.json and the performance test were not uploaded; these are build/test artifacts, not storefront runtime repairs.
- Authenticated account behavior, provider redirects, actual capture/refunds, emails and completed orders remain unverified. No sandbox completion was authorized; Stripe remains live-configured.
- Physical per-size inventory and shipping promises are unknown. Migration preserved existing untracked-stock availability and did not establish real allocations.
- Legacy `WordPressProductSync._build_woo_product` still constructs simple-product payloads. Its normal API path skips existing products, but explicit `force_update=True` must not overwrite the migrated catalog; integration modernization is outside this theme repair. Historical certification snapshots are not live product import instructions.
- Existing media approval deficits, the Klaviyo failure, historical database-authored content and third-party asynchronous side effects remain bounded limitations.

## Commits

| Commit | Repair |
|---|---|
| `3eb1b74b0` | jQuery execution order |
| `9963a6d4e` | Native account migration |
| `0bcc54074` | Native cart quantities |
| `ffacb2f80` | Size reconciliation and variation migration |
| `f9ccaef11` | CSP-safe native variation rendering/readiness |
| `f51f31409` | PDP commerce media hierarchy/hooks |
| `f45f8458a` | Truthful pre-order presentation |
| `22b0ce4db` | Status-aware checkout acknowledgement/hooks |
| `b0858722a` | Exclusive native search groups |
| `7bc31b2b9` | Mobile overlay/focus repair |
| `5de8e2f3e` | Runtime/editor token contract |

Commits are local to the named branch. Evidence is committed separately after the tested runtime SHA.

## Phase 3 Readiness

**Conditionally ready for local design-system/global-shell implementation from the certified repair SHA.** Preserve the repaired Woo controls, media gates, source contract and staging boundaries. This is not authorization to launch, submit payment, overwrite catalog data, or represent the entire ZIP as deployed/tested on staging. Whole-package promotion requires resolving the documented parity boundary and retaining the outstanding account/payment/integration checks.

Phase 3 has not started. Work stops at this report.
