# V2 handoff reconciliation — 2026-08-15

This note resolves the stale promotion payload that referenced
`/private/tmp/skyyrose-v2-promotion-worktree` and integration head
`299a9f252a5375b17a4bddfecc44d8b225207564`.

## Authoritative workspace

| Field | Current truth |
|---|---|
| Branch | `codex/skyyrose-v2-marketplace` |
| Worktree | `/Users/theceo/DevSkyy` |
| Integration ref | `refs/heads/codex/skyyrose-v2-marketplace` |
| Last verified V2 source snapshot | `a620435d646d89d3f99840350d3151a25dcbec47` |
| Main base | `f97383524206825e9f0bd24042e7990be43aee27` |
| Payload commit | `d249a687c4dd1f9f053aa8dd5f66e8717f72f8a4` |
| Attestation commit | `8b4be49d0a2642212b088974e4eca66560006f29` |
| Prototype capture | `1982efe66` (archival; no longer an active worktree) |
| Release candidate source | `47b711dac45232e3be958e8722bdb25f72ed81fa` |
| Scoped V2 paths | Tracked and clean at the source snapshot |
| Whole worktree | Dirty only because the unrelated root `CLAUDE.md` is modified |
| Promotion | `BLOCKED / FAIL_CLOSED` |

The temporary promotion worktree and the old `/Users/theceo/DevSkyy-v2-marketplace`
path are not valid sources. New work must start from `/Users/theceo/DevSkyy` on
the branch above and must preserve the unrelated `CLAUDE.md` edit. The two
documentation-only reconciliation commits after the source snapshot do not
change the V2 payload; the branch ref is the authority for future work.

## Post-reconciliation source sync

After this report was first written, two focused corrections were committed on
the active branch:

- `ba346e596` scopes the global WooCommerce product while editorial V2 cards
  render, preventing an add-to-cart action from inheriting a neighboring loop
  product; it also regenerates the current 490-message POT at that historical
  source snapshot.
- `50060893b` makes the local PDP renderer SKU-aware, so Signature, Black Rose,
  Love Hurts, Kids Capsule, and Jersey Series can each be inspected with their
  own fixture product and collection binding. This remains preview-only.

That historical package was verified with SHA-256
`49e1cdd84918caed83229fcaef1bd06725a04d7433002cb352bdc722dbbca4dd`. It is
not a promotion approval and does not replace the immutable payload/attestation
pair below.

## Black Rose Jersey Series ownership correction

The founder correction is now synchronized in source, registry, preview, and
design contracts: Jersey Series remains a dedicated `jersey-series`
presentation and Town Line reveal, but every Jersey SKU is owned by the
`black-rose` collection. Its canonical discovery route is
`/collections/black-rose/#jersey-series`; core Black Rose product cards still
exclude the jerseys so the release remains a distinct chapter inside the
parent collection story. The local PDP breadcrumb and `data-collection`
binding now return to Black Rose, and the Black Rose collection template owns
the anchored release chapter.

This correction is committed at `80f068ffd4a349a956f43770aa1bbe252df4f68c`.
The fresh build/verification reports 492 translation messages, 33 current
presentation records, 4 CSS and 7 JS assets, and a deterministic package SHA
of `e46f33de753f153d377c1f00aca5f247a10f39c9040583ab687e8e5d4abf1bb7`.
`unzip -t` passes. These are source/package checks only; the promotion gate
remains fail-closed pending the independent visual, rights, WooCommerce
staging, and founder authorization evidence listed below.

## Evidence alignment

- `.fashion-theme/promotion-manifest.json` remains the release gate and stays
  fail-closed. Its payload package is explicit rather than self-referential.
- `.fashion-theme/workspace-ledger.json` is the mutable workspace observation;
  it records `worktree_dirty: true` and `owned_paths_dirty: false`.
- `.fashion-theme/codex-desktop-handoff.json` is the review snapshot for the
  payload and attestation, not a claim that the worktree is deployable.
- `docs/design/v2-remodel/CODEX-DESKTOP-MANIFEST.json` now labels the prototype
  as archival and points verification commands at the checked-out branch.
- The V1 benchmark, catalog binding audit, design-system census, and commerce
  gap map now distinguish their historical prototype observations from the
  active tracked candidate.

## Verification performed

```text
cd wordpress-theme/skyyrose-flagship-2 && npm run verify
Product presentation registry is current (33 SKUs).
Translation catalog is current (492 messages).
Marketplace registry structure passed.
Verified 4 CSS and 7 JS assets.
SkyyRose Flagship 2 marketplace verification passed.

npm run package:theme
ZIP: f4e70896b79d02cd31fc28c76c91c4aacc4a3a25568defe8790d81e1d4d65ba2
unzip -t: no errors detected
```

The local preview listener is available at
`http://127.0.0.1:8099/tools/v2-theme-preview.php?route=home`. It is a fixture
renderer only: it does not connect to staging, mutate WooCommerce, or prove
live price, stock, variation, checkout, rights, or performance behavior.

## Still blocked by design

The reconciliation does not promote the theme. Media-rights records, complete
SKU → WooCommerce variation bindings, independent same-candidate visual QA,
fresh staging transaction evidence, and founder release authorization remain
required. The Kids Capsule full-color throne correction and the three-world
homepage opening remain preserved in the active source; historical four-card
captures must not be used as current evidence.

## Journal and commerce-state preview completion

The active candidate now includes a source-backed Journal fallback and a
local-only state matrix for the routes that must be inspectable before staging:

- `journal` renders the four founder-approved V1 press records (Maxim, San
  Francisco Post, Best of Best Review, and CEO Weekly) when no live WordPress
  posts are available. Live posts remain authoritative after import.
- `wishlist`, `faq`, `shipping-returns`, `size-guide`, `privacy-policy`,
  `terms-of-service`, and `accessibility` resolve through the real V2 page
  template and expose reviewable fixture copy in the local harness.
- `cart`, `checkout`, `account`, and `order-tracking` render explicit branded
  state shells. They are not WooCommerce transaction evidence and do not
  fabricate payment, order, stock, or customer data.
- `404` is routed through the real V2 error template, so navigation and
  fallback states can be inspected without staging writes.

This synchronization is committed at
`862cbafe278f380b4cbbddfd5b7b788992749010`. The fresh build reports 495
translation messages, 33 presentation records, and 4 CSS plus 7 JS generated
assets. The deterministic local package is
`93876ddf4d5d15cc0e91e26e8b6d50e134b0c515a7c25eb8a7c8726b24069cf3` and
`unzip -t` passes. This remains source/package verification only; the preview
listener is fixture-only and does not prove live WooCommerce routing,
variation resolution, checkout, rights, accessibility, or performance.
