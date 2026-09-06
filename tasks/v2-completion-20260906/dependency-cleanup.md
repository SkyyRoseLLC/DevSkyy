# V2 dependency cleanup review

Read-only audit of `415dc4368b44efc06d0d28928362901f9624a20f`, branch `codex/v2-visual-recovery-founder-20260905`, on 2026-09-06. This report authorizes no deployment and makes no runtime edits. Current work by the manager and Skyy engineer is deliberately preserved. The latest founder directive requires completion and optimization, with future Town Line and Skyy authoring retained.

## Recommended bounded changes

| Priority | Concrete change | Measured opportunity | Conditions |
|---|---|---:|---|
| 1 | Extract the 104 exclusively owned About, Reserve, Contact and Commerce Preview rules from `assets/css/theme.css` to conditional content CSS. | 14,244 source bytes removed from the global stylesheet; compressed delivery must be measured after build. | Use the existing conservative `legacy-world-components` route gate. Preserve every declaration, rule order and enclosing at-rule. Keep mixed selectors global. Enqueue after theme and before controls/global-shell. |
| 2 | Make the two BR-006 MP4s below source-only in the release boundary. | 38,020,556 package file bytes, about 18.3% of the current unpacked runtime. | Preserve both files at their exact paths and hashes, especially the founder footage's SOT authority. Change package classification only. |
| 3 | Archive and remove the two literal `if ( false )` draft spans in `page.php`. | 8,907 runtime source bytes; no current network or rendered-markup reduction. | Preserve the exact original spans in an authoring artifact or recoverable source record, with hashes. Keep live partial calls, route branches and every referenced media asset. |
| 4 | Remove the impossible House motion enqueue branch and its two unused filename locals in `functions.php`. | Zero current request savings; removes misleading dead orchestration. | Keep `house-of-roses-motion.js`, its built output, film/helper and future Town Line assets. Black Rose is unconditionally an enabled editorial collection, so the branch requiring the opposite cannot execute. |
| 5 | Scope classic Woo CSS only after proving all native overlay states on exact modern Home/four collection routes. | Root's profile identifies a substantial opportunity; this audit does not independently measure it. | Layout/smallscreen are the narrower first step. General CSS also styles notices, mini-cart, forms and buttons. Preserve transactional/archive/content/immersive routes and provide an extension opt-in; require browser evidence before certifying full removal. |

The exact CSS selector, source-line and ancestor inventory is in `dependency-css-rules.json`. The groups are About 56 rules/7,804 bytes, Reserve 30/3,527, Contact 9/1,422, and Commerce Preview 9/1,491. Prefix-only ownership is necessary but is not permission to delete declarations. The preview partial has no current PHP caller found; its unknown authoring status is why its styles should remain available on general routes.

The safe extraction gate is: not front page, not an enabled collection world, not single product, and not Shop/product taxonomy. This matches the already accepted legacy stylesheet gate and keeps About, Contact, Pre-Order, Collections, general content and legacy/unknown templates covered. Do not narrow it to only the three named pages during this pass. Native mini-cart and Quick View remain on the governed routes; shared `.woocommerce`, card, notice, button, overlay, grid and form rules are outside this extraction.

## Exact package exclusions

Paths below are relative to `wordpress-theme/skyyrose-flagship-2/`.

| File | Bytes | SHA-256 |
|---|---:|---|
| `assets/scroll-world/video/black-rose/br-006/source-footage/black-rose-sherpa-founder-footage-2024.mp4` | 20,973,139 | `8b0953c5824f07f0a8f483aeac34305ec032d453c93da17a37cfa4ecd6918127` |
| `assets/scroll-world/video/black-rose/br-006/motion-plates/br-006-hero-origin-backdrop-v1.mp4` | 17,047,417 | `cc5f9ff9ea27eaa2ca517da6986f3787bab5953bcc9d93e87c4416c9bea9cba2` |

The founder footage remains a physical-product authority reference: `tools/v2-source-certification/inputs/product-sot.json:1262` and `inputs/woocommerce-product-sync.json:2274` bind it as BR-006 `garment_reference_3`. It is not a current delivered scene video. Neither file has a literal theme runtime PHP/JS/CSS/JSON caller in the scan, belongs to the approved runtime scene set, nor appears in the matching local post-content query. The backdrop file appears in package/census/history evidence. Both should remain available for production tooling and future authoring; source deletion is rejected.

The current boundary has 507 release entries totaling 207,747,719 file bytes. Excluding these two produces 169,727,163 file bytes before ZIP overhead and any other changes. This is a distribution reduction, not a claim of page-load improvement. The previous stored ZIP's size includes overhead. Rebuild the package and check its manifest rather than substituting this arithmetic for a packaging receipt.

## Exact draft boundaries

`page.php:27–62` is the Pre-Order draft, 5,855 bytes, immediately after the live `template-parts/v2-preorder` include. `page.php:65–80` is the About draft, 3,052 bytes, immediately after `template-parts/v2-about`. Both have literal false guards. The machine report records the exact span hashes. Removing those spans cannot execute or remove a current hook, query or user-facing element. Preserve the draft source for reference, especially its older salon/hotspot composition; its product wording and SKU assumptions must not be copied into future live Town Line without current authority validation.

## Active dependency map and preservation decisions

- The local database uses `skyyrose-flagship-2` as stylesheet and template. Home is page 10; Shop 5, Cart 6, Checkout 7, Account 8. All 22 local page template metadata values are default. This does **not** mean custom collection templates are unused: the theme resolves the canonical four collection slugs itself. `dependency-local-db.json` records the read-only SQL snapshot; the manager separately exported `.artifacts/v2-completion-20260906/page-assignments.json`.
- Home uses `front-page.php`, its current Home partials, collection-world/home-page CSS, shared card and visual recovery systems. The four collection slugs take the early governed branch in `template-collection.php`. Shop/PDP use their native Woo overrides and dedicated CSS. General pages use `page.php`; About and Pre-Order delegate to live partials. WordPress fallback files remain required even without an explicit template assignment.
- **Preserve `legacy-world-components.css` and its build output.** `skyyrose2_render_collection_rail()` has live callers at `page.php:22` and `template-parts/v2-preorder.php:49`. The dead draft's third call is not the only caller. Collections and Pre-Order still consume the classic rail. The fallback tail of `template-collection.php` remains reachable for an explicitly assigned unknown slug, and local absence of dedicated immersive pages does not invalidate their public template contract.
- **Preserve the Town Line system.** Keep `skyyrose2_render_black_rose_jersey_series()`, its registered product/source relationships, House motion JS and built output, film/posters, and Town Line art. Their final composition is deferred. Removing an impossible current enqueue does not establish that these assets are disposable.
- **Preserve Skyy completely.** Keep canonical GLB, local pinned Three module graph, Draco decoder files, current renderer and derived gait, loader, guide, dialog/Home lifecycle, fallback portrait and authoring records. A source-only classification requires distinguishing authoring assets from the actual dependency graph. No Skyy deletion is proposed.
- **Preserve all seven Woo overrides:** archive-product, content-product, single-product, single-product/product-image, cart/cart, checkout/form-checkout and checkout/thankyou. The gallery wrapper governs initial, embedded default/reset and variation/AJAX media permission; the checkout and cart overrides are repaired commerce boundaries. They are not orphan templates.
- **Preserve the nine distinct font files and licenses.** No release-file SHA duplicate group exists. Archivo/Cinzel/Inter and collection script fonts have token/theme/provenance or authoring references. `design-tokens.css`, `theme.json`, `data/font-provenance.json` and the native-scene authoring script are consumers. Request/subset optimization may still help, but naming similarity or unused initial coverage cannot establish duplicate files.
- **Hold unknowns.** The two alpha spin videos, unassigned immersive templates and `v2-commerce-preview.php` are candidates for further source-only assessment, not authorized deletions. Static absence and one local database do not prove absence from remote content, future authoring or dynamically named calls.

## Build, package and verification boundaries

The V2 build uses pinned PostCSS/CleanCSS/Terser, scene guards, tokens, registry, same-source card renditions and i18n. Source CSS/JS and tracked minified outputs must remain reproducible. `tools/v2-source-certification/package-boundary.json` determines release inclusion; canonical input and runtime hash records must distinguish intentional runtime changes from immutable authoring authority. The file census records every classified path, size, hash and release category; reference hits record inspected dependency trails.

After implementation, the manager must rebuild/check assets and integrity, run the existing relevant regressions, check packaging parity, and verify Home/four collections/Shop/PDP plus Collections/Pre-Order/About/Contact. CSS comparison must include mobile and desktop, no-JS, empty/filled bag, native notices, Quick View and keyboard focus. The current root changes to WooCSS delivery are not independently approved here: the extension opt-in is useful, but does not itself prove inherited general CSS is unused in all states.

No runtime file, database state, asset, build output or browser state was changed by this audit. Direct local database reads used a read-only transaction without bootstrapping WordPress. No remote database inventory was performed. Existing broad baselines were reused as context rather than rerun. The recommendations are bounded by the current source and local fixture; unknown consumers stay preserved.

Machine evidence: `dependency-cleanup.json`, `dependency-file-census.json`, `dependency-reference-hits.json`, `dependency-local-db.json`, and `dependency-css-rules.json`.

## Implementation source recheck

The manager's actual partition is `content-page.css`, 105 rules and 13,332 bytes removed from global source. This supersedes the preliminary 104-rule/14,244-byte opportunity count: mixed selectors are split and the narrower actual prefix expression leaves Commerce Preview BEM selectors global. This is safe but is a smaller optimization; do not claim all preview styles moved. Independent PostCSS comparison confirms each moved selector retains its ordered declarations, importance and ancestor conditions against base415. `content-css-source-review.json` records this PASS and the remaining preview selectors. Concurrent cart/account declarations are explicitly outside that partition comparison.

The enqueue follows the conservative legacy gate and remains before controls/global-shell. The impossible House enqueue and only its unused path locals were removed; the source module remains. No actionable source defect was found in these two deltas. Browser checks of About/Contact/Pre-Order/Collections remain necessary for overlapping-selector cascade and visible parity; per-selector equality alone is not pixel equivalence.
