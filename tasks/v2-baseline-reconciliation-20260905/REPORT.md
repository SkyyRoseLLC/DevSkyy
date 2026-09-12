# SkyyRose V2 Baseline Reconciliation Report

**Certification: FAIL — the recovery is verified, but the proposed source baseline is not yet self-contained or fully verifiable.**

Scope: recovery and reconciliation only, September 5, 2026. No deployment,
redesign, commerce-data change, WordPress-content change, plugin setting change,
order, payment, or customer operation was performed. Phase 2 has not begun.

## Evidence Snapshot

The original archive remains unchanged at
`.artifacts/v2-commerce-baseline-20260905/staging-v2-before.tar.gz`.

SHA-256: `3850af0ae0caa20a49e8ec4f029ef1543be4ac5ef65202b2a1a0336293f09f4a`.

All 447 regular files were imported without content edits into evidence commit
`bb0e6eecf0af54b3f0c14e5eadb7b138530ce09b`. Every path and SHA-256 in the Git tree
matches the captured server manifest. A separate streamed Git-archive recovery
rehearsal also verified all 447 paths/hashes. There were no symlinks or special
files in the captured archive.

See [the full evidence manifest](evidence-manifest.json) and
[local recovery rehearsal](rollback-rehearsal.json). This is a theme backup;
it does not contain WordPress database, media-library uploads, plugins, or orders.

## Git Baseline

- Audited source: `6d3e87bab9f401856e9d122b2f2ccbfa4a129b2a`.
- Prior investigation checkpoint: `56691ca771b342ac55cabf80de6ca14a7000d834` on
  `codex/v2-commerce-safe-baseline-20260905`.
- Recovery branch: `codex/v2-recovered-staging-baseline-20260905`.
- Exact evidence import: `bb0e6eecf0af54b3f0c14e5eadb7b138530ce09b`.
- Normalized review candidate: `53ae72dc339d1130b2ce8c2dd2d18108569e1efe`.

Existing history and the audited branch were preserved. Nothing was pushed.

## Recovered Staging State

The active staging theme is `skyyrose-flagship-2`, under
`/srv/htdocs/wp-content/themes/skyyrose-flagship-2`. Staging reported WordPress
7.1 and PHP 8.4.25. The recovered theme includes 48 PHP files, six first-level
CSS sources, eight first-level JavaScript sources, generated bundles, runtime
JSON contracts, creative tooling, and media/QA artifacts.

It is a classic PHP WooCommerce theme with theme-local presentation registries,
native Woo product and purchase hooks, vanilla JavaScript progressive
enhancement, collection scenes, and separately loaded mascot/3D features.

## Difference Matrix

Compared with the audited checkout: 192 common files are byte-identical,
39 common files changed, 216 files exist only in staging, and three local media
files are absent from staging. Four shared shell scripts also lost executable
mode bits during deployment; those metadata differences are separate from
content counts.

| Classification | Modified shared | Staging-only | Total staging deltas |
|---|---:|---:|---:|
| KEEP | 0 | 48 | 48 |
| MERGE | 19 | 157 | 176 |
| REGENERATE | 12 | 3 | 15 |
| COMMERCE RISK | 8 | 6 | 14 |
| UNKNOWN | 0 | 2 | 2 |
| OBSOLETE | 0 | 0 | 0 |
| DEPLOYMENT RESIDUE | 0 | 0 | 0 |
| SECURITY RISK | 0 | 0 | 0 |
| **Total** | **39** | **216** | **255** |

The three local-only BR-004 preview files are separately classified MERGE and
remain recoverable on the audited branch. Zero files in a category means no
sufficiently supported classification, not proof that the theme is secure or
free of obsolete material.

Every file has a hash, disposition, rationale, historical-match evidence,
working-tree corroboration where available, and source references in the
[readable matrix](difference-matrix.md) and [machine-readable matrix](difference-matrix.json).
KEEP proposes source retention; it does not grant visual approval or deployment
eligibility. MERGE files have been preserved, not automatically reconciled into
an approved architecture.

## Important Modified Files

| Surface | Finding and disposition |
|---|---|
| `functions.php` | Scene/media evolution plus the approved-card loader and exact-SKU fallback. Exact match to the verified 46-file deployment artifact. COMMERCE RISK until media and purchase presentation are reconciled. |
| `assets/css/theme.css` | Existing scene/PDP CSS with a 65-line editorial-card/frame/nameplate block appended. That whole block matches the card-work checkout; full deployed file matches its release artifact. MERGE, without redesign. |
| `assets/js/theme.js` | Exact historical match to `41a3142def89375782208ac4bb3e5d022d96719c`. Its minified output reproduces modulo one trailing newline. Existing menu, search, quick-view and cart-fragment behavior retained. |
| `inc/performance.php` | Exact match to `803fb08b5ea2aedc027693bcd17248c18a04bb11`; adds homepage-only `jquery-core` defer at enqueue priority 120. MERGE; known runtime risk remains. |
| `inc/security.php` and `inc/seo-indexing.php` | Byte-identical to audited source. No recovered security/CSP or SEO changes silently substituted. |
| `woocommerce/` overrides | All byte-identical to audited source, including cart, checkout, thank-you, archive, and single-product wrappers. |
| `search.php` | Byte-identical; the earlier search-type overlap is not fixed by recovery. |
| `template-parts/commerce/product-hero.php` | Changes native-gallery behavior to require approved on-model media; conditionally skips `woocommerce_before_single_product_summary` when missing. COMMERCE RISK for media visibility and third-party hook compatibility. |
| Product/card registries | Runtime manifest hashes agree with historical product SOT, but 33 garment-type fields are additional deployed data absent from the generator. COMMERCE RISK and generated/manual drift. |
| Mascot/chat | Most sources unchanged; loader/minified differences reproduce as newline normalization. Recovered mascot template and main CSS differ from audited source. Mobile obstruction remains a baseline defect, not a recovery fix. |

## Staging-Only Source

Content inspection and historical comparison establish meaningful development:

- Approved-card helper, exact-SKU image manifest, 33 card images, card renderer,
  and Kids guardian-proof routing. All are tied to the verified card release.
- Scene-composition helper and template, collection-motion controller, scene
  CSS, and contracts. Several match the current `DevSkyy-product-card-approved`
  working tree exactly, but this is content corroboration, not a historical
  deployment receipt for the entire scene layer.
- Collection motion code includes viewport arbitration, reduced-motion and
  Save-Data handling, manual play/pause, visibility suspension, and still-image
  fallback. It does not introduce a new cart or payment implementation.
- Media authoring/validation scripts have recognizable project purposes. Some
  are nonportable, one-shot tools with hardcoded workstation paths; some can
  invoke paid external providers. Those tools were inspected, not executed.
  They should be reconciled into internal tooling, outside a runtime package.

Of the 255 staging deltas, 150 match historical file contents exactly. Among
the 105 without an exact historical match, 89 match another current worktree.
These counts overlap with the 46-file release-artifact evidence. A current
worktree match does not establish an immutable source commit.

## Generated / Deployment Residue

The pinned build uses CleanCSS 5.3.3 and Terser 5.36.0. It emits sibling `.min`
files; the V2 repository intentionally tracks those outputs. Do not simply
delete all minified files. `build-pot.py` generates translations and the
registry generator consumes the root product SOT.

All 14 minified outputs can be explained: `theme.min.css` is byte-identical to
the recovered-source rebuild, and the other 13 match after removing one
trailing newline. The translation catalog was stale and regenerates to 557
messages. The review candidate normalizes those 13 bundles, regenerates the
POT, and restores four shell executable bits in a separate commit. It leaves
all runtime source and registry bytes unchanged from the evidence import.

No log, backup, ZIP/tar, swap file, environment file, or OS metadata filename
was found in the 447-file snapshot. A content-hash scan found no exact duplicate
files. Candidate/source/QA imagery remains MERGE, since absence of a direct
runtime reference is not proof of obsolescence.

The earlier 46-file card release tar contained an additional AppleDouble entry
`assets/css/._theme.min.css`. That entry is absent from the recovery snapshot
and was not imported. Its presence in the earlier tar is deployment packaging
residue, not a 448th captured theme file.

## jQuery Failure Analysis

**Observed execution-order failure: high confidence. Exclusive root-cause
attribution to one optimizer: not established.**

Three canonical-homepage reloads each produced two `jQuery is not defined`
exceptions: jQuery Migrate and `/_jb_static/??8405d66706`. The latter bundle's
failing expression is the jQuery blockUI initialization `e(jQuery)`; the bundle
also contains WooCommerce-related code. This is not merely an ornamental
theme-script warning.

Rendered order:

1. `jquery-core-js`: `defer`, `data-wp-strategy="defer"`, and
   `data-jetpack-boost="ignore"`.
2. jQuery Migrate: ordinary blocking script, no defer/async.
3. The affected optimized bundle: ordinary blocking script, no defer/async.
4. Deferred jQuery executes after parsing, too late for the preceding consumers.

Layer evidence:

- Theme: explicitly requests defer for `jquery-core` on the homepage.
- WordPress: `jquery` aliases `jquery-core` and `jquery-migrate`; the Migrate
  registration has no direct dependency on core. Core loading-strategy logic
  evaluates enqueued dependents, so dependency metadata and optimizer handling
  both matter.
- Page Optimize 0.6.3: extends `WP_Scripts`, builds a separate concatenation
  output list, excludes requested delayed scripts from concatenation, and emits
  concatenated tags according to its own load mode. The stored load-mode option
  is absent; the inspected default is blocking. It excludes jquery/core by
  default.
- Jetpack Boost 4.7.0: minification and render-blocking modules are enabled.
  Its code owns the `_jb_static` path and can buffer/move scripts to the body
  end while retaining excluded tags. The observed core tag is excluded from
  that movement. Minify exclusions include jquery and jquery-core.
- LiteSpeed: inspected JS minify/combine flags are empty and JS defer is `0`;
  there is no evidence that it introduced this observed defer attribute.
- Elementor/Pro are active. Inspected experiment options do not establish
  ownership of the core defer tag. No Elementor-specific cause was demonstrated.
- WordPress.com caching remains a possible confounder. The documented
  request-only `?concat-js=0` diagnostic produced the same optimized tags and
  errors; because it did not yield an isolated unoptimized response, it cannot
  prove Page Optimize is solely responsible or uninvolved.

A vendor-checksum command verified one of the two inspected optimizer plugins;
Page Optimize was reported with added localization files. It did not report
changed executable plugin files. This is not evidence of tampering.

No optimizer was disabled or reconfigured. A valid future causal test must
control the generated response and dependency graph, not merely delete an
attribute or disable plugins until the error disappears.

## Commerce Risk Review

Preserved structural contracts include cart-item keys, `woocommerce-cart`
nonce, quantity fields keyed by cart-item identity, Woo item filters, checkout
form and order-review hooks, Woo-native product summary/add-to-cart dispatch,
and native product permalink/price/stock access. No new payment gateway,
authentication, inventory, or order-writing code was found in the recovered
runtime delta.

Specific concerns remain:

- The recovered PDP gates native gallery output on an on-model-front role and
  matches attachment basenames. It does not establish attachment pixel identity
  from the media hash at runtime. It also conditionally omits the whole
  before-summary hook, which can suppress third-party contributions.
- SG-005's opening-media record has no views and is `STALE_PRODUCT_HASH`, while
  the separate approved-front manifest supplies its card image. Thus the card
  and PDP resolve different authority paths. Across 33 SKUs, the opening-media
  manifest has 16 stale, 9 missing-front, 5 rejected-authenticity, and only 3
  approved records. Recovery must not convert blocked media into approved media.
- The approved-front helper validates path containment and file existence, but
  does not compare the manifest's SHA-256 in its runtime resolver. Release
  evidence validates the captured image bytes; future replacement drift needs
  its own validation boundary.
- Scene links resolve real Woo products and use product permalinks, but their
  pre-order label depends on presentation metadata. No transactional preorder
  engine is introduced by these changes.
- Cart/checkout overrides being unchanged is not proof of complete gateway
  compatibility. No order/payment scenario was executed. The existing failed
  thank-you copy still says “Nothing was charged”; its correction remains in
  the later authorized phase, not this recovery.
- My Account, real size/variation semantics, and search grouping remain earlier
  audit issues. No data migrations or account actions were performed.

## Build Reproducibility

| Validation | Proposed candidate in this checkout | Historical-input experiment |
|---|---|---|
| Asset build/check | PASS; 6 CSS + 8 JS outputs explained | PASS |
| Translation build/check | PASS; 557 messages | PASS |
| PHP syntax | PASS; 48 files on PHP 8.5.6 | PASS |
| Provisioning structural checks | PASS | PASS |
| Performance contract | PASS | PASS |
| SEO/indexing contract | PASS | PASS |
| Registry generation/check | BLOCKED: missing root product-SOT module/manifest | PASS with hash-matching historical inputs; removes 33 extra garment fields |
| Full production `npm run build` | BLOCKED at registry dependency | PASS in the isolated experiment |
| Opening-media validator | BLOCKED without root SOT/input assets | PASS: 33 records, 3 approved asset sets |
| Collection-hero motion validator | PASS on recovered assets | PASS |
| Full `npm run verify` | FAIL: missing founder-scene manifest | FAIL: after recovering that historical manifest, BR-COMMERCE-1 scene asset is missing |
| Marketplace script | BLOCKED by root dependency | FAIL: retired font in `scripts/build-hero-commerce-c1.cjs:131` |

The historical experiment used the matching product-SOT module and manifest
from `803fb08b5ea2aedc027693bcd17248c18a04bb11`, with the existing
`DevSkyy-product-card-approved` checkout providing read-only source inputs.
Its SOT hash is `4ccfbe18aba2846c8406f1f0d34853158e563601051527a47f8e401a71beea12`,
matching both recovered manifests. Those external inputs have not silently
become dependencies of the proposed Git candidate. No canonical CSV/dossier
was replaced. Source inputs must be explicitly reconciled before a clean
checkout can reproduce the full build.

Node 22.23.2/npm 10.9.8 and the existing pinned dependency installation were
used. Python validation used the existing Python 3.12.12 environment with
Pillow 12.3.0 after system Python lacked Pillow. No provider-generation or
judge commands were run.

## Deployment Provenance

**Known:** a scoped deployment was found under the `3f2d` card-work checkout in
`.artifacts/v2-staging-card-deploy-20260905/`. Its receipt records 46 files,
33 cards, and verification at `2026-09-05T11:33:39.433382+00:00`.

The candidate archive SHA-256 is
`12c1d69eea3bf804b0afc7455f540eb3d236dafdb0cfae57e0c8e61cfbe84aad`.
That hash matches the receipt, and all 46 declared files match this recovered
snapshot. The script checks staging siteurl/theme and before/after hashes,
then copies listed assets, data, helpers and consumers into place. The receipt
explicitly describes merging local approved-card work onto existing staging
source to preserve newer hero/film/scene work.

This explains why complete deployed `functions.php`, `theme.css`, and the
registry had no exact historical Git match. They were assembled for that release,
not built from one source commit. See [verified release evidence](deployment-receipt-verification.json).

**Corroborated but not fully release-attributed:** the pre-existing scene layer
matches historical files and the `DevSkyy-product-card-approved` worktree in
many places. There is no verified single original Git release accounting for
all 447 files. The recovery commit now provides exact content provenance from
capture forward; it does not invent the missing earlier deployment history.

The generic `scripts/deploy-theme.sh` was successfully inspected read-only in
this phase. It defaults to V1/production, checks V1 version/asset conventions,
and excludes data outside a V1 allowlist. It is unsuitable as an unmodified V2
release command. Existing CI/scripts do not supply a verified receipt linking
the complete captured V2 tree to one CI artifact.

Specifically, `.github/workflows/ci.yml:419` runs its WordPress PHP, build and
minification-drift job against `skyyrose-flagship` and the parent build package,
not V2. A green result there is not a V2 gate. V2's own `package-theme.sh` runs
its build and verification, normalizes timestamps and sorts ZIP entries, but
its broad copy currently retains internal creative scripts and review assets.
It was inspected, not run to publish a release package.

## Candidate Canonical Architecture

Retain the classic V2 PHP theme and WooCommerce authority. Keep native GET
search, endpoints, cart/checkout hooks and server-rendered purchase controls.
Preserve the captured creative/runtime work while deciding its source
disposition; do not replace it with the older audited theme.

The next certifiable source should contain:

1. One explicitly reconciled root catalog/SOT input contract, including the
   source of `garment_type`; no hand-maintained additions to generated output.
2. A deterministic registry, asset and translation build from a clean checkout.
3. Separate runtime manifests/assets and internal creative scripts/QA evidence.
   Motion approval and media-fidelity evidence remain independent gates.
4. V2 verification against the intended current scene contract, with stale
   legacy verifier dependencies reconciled rather than bypassed.
5. A V2-specific immutable release manifest binding source revision, toolchain,
   artifact hashes and staging destination. Generic V1 deployment remains out
   of scope until adapted and independently tested.

## Candidate Canonical Commit

**Proposed review candidate, not approved canonical source:**
`53ae72dc339d1130b2ce8c2dd2d18108569e1efe`.

Relative to the exact recovered evidence, this commit changes only 13 generated
bundle trailing newlines, the generated POT catalog, and four executable bits.
It retains all 447 files and all unexplained/flagged source. The registry was
not regenerated into this commit because doing so would remove runtime-used
garment metadata. This candidate cannot yet be certified from a clean checkout.

## Remaining Unknowns

- `data/collection-hero-motion.json` and `data/collection-scene-motion.json`:
  current working-tree content is corroborated and some asset hashes validate,
  but the complete original approval/release chain is not independently bound
  to a historical Git commit. They remain explicitly UNKNOWN, not KEEP.
- Original deployment ownership for the entire pre-card scene/runtime layer.
- Which intended SOT generation/release should be canonical for this current
  checkout, including garment-type generation and all required physical inputs.
- Intended treatment of the founder-scene verifier's missing legacy assets.
- Exact causal contribution of each optimizer and cache layer to the observed
  jQuery order; browser sequence is proven, isolated optimizer attribution is not.
- Real WordPress/WooCommerce runtime equivalence. The local renderer substitutes
  fixture data and service pages, including $128 fixture prices versus real
  staging prices; it cannot certify product, account, checkout or payment behavior.

## Baseline Certification

**FAIL.** Evidence preservation, content recovery, classification, safe asset
normalization and the local recovery rehearsal pass. Full clean-checkout build
provenance and V2 verification do not. A conditional pass would conceal those
substantive gates.

Browser evidence: home, shop, Signature collection, PDP, cart, checkout shell,
and account fixture routes rendered without PHP fatal output; menu and search
dialog opened. Local console collection returned no errors for that limited
fixture sample. Screenshots and DOM records are in the ignored evidence folder.
Actual 390px captures were verified through browser metrics, with no horizontal
overflow on sampled home/PDP. Earlier viewport-capability captures retained
1280px and are not used as mobile evidence; use filenames ending `-390.png`.
The PDP still presents unverified-media empty space. Search-result grouping and
native service-page functionality are not represented faithfully by this fixture.

Staging runtime errors still reproduce. No visual equivalence, commerce health,
or production readiness claim follows from fixture rendering.

## Restoration Procedure

For a future separately authorized recovery, the exact-byte source is evidence
commit `bb0e6eecf0af54b3f0c14e5eadb7b138530ce09b`, not a rebuild of the normalized
candidate. Use the original verified archive or export that exact theme tree:

```sh
git archive --format=tar \
  bb0e6eecf0af54b3f0c14e5eadb7b138530ce09b:wordpress-theme/skyyrose-flagship-2 \
  > v2-recovered-evidence.tar
```

The stream/export path has been rehearsed locally and all 447 member hashes
match. Archive-container hashes can differ from the original gzip archive;
the path-bound file manifest is the restore comparison authority.

Before any future server write: confirm the staging home and active V2 slug;
verify every proposed file against the manifest; back up the then-current
theme; unpack to a separate sibling; verify hashes and PHP compatibility; only
then switch the exact V2 theme directory under an approved maintenance/rollback
procedure. Preserve the replaced directory for reversal. Validate public routes,
assets and PHP errors afterward. No such remote restoration was executed here.

Build command for future reconciled releases is `npm run build` inside the V2
theme with pinned Node dependencies and explicitly recorded Python/SOT inputs.
That full clean-checkout build is currently blocked; do not use a partial build
as a release artifact. Exact snapshot recovery requires no build or regeneration.

Database dependencies are separate: active theme selection, page/template
assignments, product/media relationships, Woo options, plugins and optimizer
settings are not captured by the theme archive. A future theme rollback cannot
reverse any database migration.

## Next Recommended Action

Reconcile the build-input and verification contracts first: recover or adapt
the historically proven SOT adapter without replacing current product truth,
generate garment-type metadata from its approved source, separate one-shot
creative tooling from runtime packaging, and align scene verification with the
intended scene records. Re-run the full V2 gate in a clean, isolated checkout.

Only after that source baseline is accepted should Phase 2 diagnose and repair
the jQuery dependency order. Do not deploy this review candidate or start
runtime remediation automatically. This recovery/reconciliation pass stops here.
