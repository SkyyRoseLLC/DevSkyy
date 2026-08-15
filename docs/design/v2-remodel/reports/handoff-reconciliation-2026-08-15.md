# V2 handoff reconciliation — 2026-08-15

This note resolves the stale promotion payload that referenced
`/private/tmp/skyyrose-v2-promotion-worktree` and integration head
`299a9f252a5375b17a4bddfecc44d8b225207564`.

## Authoritative workspace

| Field | Current truth |
|---|---|
| Branch | `codex/skyyrose-v2-marketplace` |
| Worktree | `/Users/theceo/DevSkyy` |
| Integration head | `9b4bb5acb1e61f55c8a116d6937a316e16fe9595` |
| Main base | `f97383524206825e9f0bd24042e7990be43aee27` |
| Payload commit | `d249a687c4dd1f9f053aa8dd5f66e8717f72f8a4` |
| Attestation commit | `8b4be49d0a2642212b088974e4eca66560006f29` |
| Prototype capture | `1982efe66` (archival; no longer an active worktree) |
| Release candidate source | `47b711dac45232e3be958e8722bdb25f72ed81fa` |
| Scoped V2 paths | Tracked and clean |
| Whole worktree | Dirty only because the unrelated root `CLAUDE.md` is modified |
| Promotion | `BLOCKED / FAIL_CLOSED` |

The temporary promotion worktree and the old `/Users/theceo/DevSkyy-v2-marketplace`
path are not valid sources. New work must start from `/Users/theceo/DevSkyy` on
the branch above and must preserve the unrelated `CLAUDE.md` edit.

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
Translation catalog is current (490 messages).
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
