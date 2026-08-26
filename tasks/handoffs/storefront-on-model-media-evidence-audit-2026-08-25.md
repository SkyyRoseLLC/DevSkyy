# Storefront On-Model Media Evidence Audit — 2026-08-25

## Scope and boundary

This is a read-only evidence handoff for the V2 product-card media projection.
It neither approves new media nor authorizes storefront wiring, a WooCommerce
write, a commit, deployment, campaign use, or rights/promotion.  Product
fidelity, native-scene integration, and founder promotion decisions remain
separate gates.

The review source is repository revision `1c7e413241f7d3605aca3bee47f523f75017709a`
(`feat(sot): add catalog-wide support gate`).  Its product SOT digest and its
opening-media manifest binding are both
`4ccfbe18aba2846c8406f1f0d34853158e563601051527a47f8e401a71beea12`.
The manifest's 33 SKU keys and per-SKU product hashes exactly match that SOT.

## Hash-bound current card-media support

| SKU | Source | SHA-256 | Bytes | Evidence state |
| --- | --- | --- | ---: | --- |
| `br-006` | `wordpress-theme/skyyrose-flagship/assets/images/products/br-006-onmodel.webp` | `d474d97d9fac60eff864a14f30b6e07845ea71696ae0483a2c5f7108f22c815b` | 240482 | Registered current on-model front |
| `sg-009` | `wordpress-theme/skyyrose-flagship/assets/images/products/sg-009-onmodel.webp` | `8f833dc10024745bcc4d8378028124d62f1609472d909d61b8f504dcbc641e33` | 222756 | Registered current on-model front |
| `sg-013` | `wordpress-theme/skyyrose-flagship/assets/images/products/sg-013-onmodel.webp` | `cb58e694639ff02b4dd7c1d009516df237f0acb39a8c6f6948930dad5d3c157b` | 269208 | Registered current on-model front |

All three files are present, git-tracked, and byte/hash-match their records in
this worktree.  They were inspected as human on-model photos, not ghost or
mannequin placeholders.  That observation does not establish source rights,
model release, campaign use, or new founder approval.

## Per-SKU closure matrix

| Status | SKUs | Required closure, before any promotion |
| --- | --- | --- |
| `SUPPORTED` (3) | `br-006`, `sg-009`, `sg-013` | Preserve the exact bound source and product hash; re-review only if either changes. |
| `MISSING_APPROVED_ON_MODEL_FRONT` (9) | `br-002`, `br-005`, `br-009`, `br-010`, `br-012`, `kids-002`, `sg-001`, `sg-002`, `sg-007` | Register an authentic same-SKU front on-model asset with the intake fields below. Existing flat, ghost, or pending-hub candidates do not satisfy it. |
| `REJECTED_AUTHENTICITY` (5) | `br-001`, `br-003`, `br-004`, `br-007`, `br-011` | Supply a new authentic source that resolves the recorded fidelity defect; rejected material stays quarantined and cannot be a reference or fallback. |
| `STALE_PRODUCT_HASH` (16) | `br-008`, `br-014`, `br-015`, `kids-001`, `lh-002`, `lh-003`, `lh-004`, `lh-005`, `lh-006`, `sg-003`, `sg-005`, `sg-006`, `sg-011`, `sg-012`, `sg-014`, `sg-015` | Rebuild the candidate/review package against the current product SOT hash, then register a fresh approval. A metadata rebind or a filename match is insufficient. |

The rejected-SKU reasons are: `br-001` has an incompatible contrasting
embroidered graphic; `br-003` suppresses the canonical patch color; `br-004`
does not match the corrected hoodie; `br-007` has incorrect/missing required
logo placements; and `br-011` invents a prohibited NHL shield.

## Required registration record for a human-supplied source

One record is required for one SKU and one view.  Do not infer any of these
values from a filename or copy them from a different SKU.

```json
{
  "sku": "<canonical SKU>",
  "role": "on_model_front",
  "source": "<repo-relative asset path>",
  "sha256": "<exact source SHA-256>",
  "bytes": 0,
  "product_sot_sha256": "<exact current data/product-sot.json SHA-256>",
  "product_hash": "<exact current product hash for this SKU>",
  "reviewed_at": "<ISO-8601 timestamp>",
  "approval_reference": "<founder or authorized-review record ID>",
  "fidelity_verdict": "APPROVED_CURRENT_STOREFRONT_ON_MODEL_FRONT",
  "source_rights_record": "<owner/capture/release/usage record ID or explicit separate-gate reference>"
}
```

The registry validator must reject the record unless all of the following are
true: the SKU exists in the current product SOT; its `product_sot_sha256` and
`product_hash` match exactly; the source resolves inside the repository; the
source hash and byte count match; the source path equals that SKU's SOT
`on_model_front` path; the role is exactly `on_model_front`; and both the
fidelity decision and approval reference are present.  A product card may use
the asset only after that validation passes.  Rights/promotion must remain an
independent authorization check.

## Local-original and generation-receipt rule

Every generation candidate must retain its originals locally.  A provider URL,
temporary upload ID, optimized reference pack, contact sheet, or derived crop
is never an original and cannot be the only receipt input.  Before a paid call,
write a receipt containing at least:

```json
{
  "sku": "<canonical SKU>",
  "view": "front|left|right|back|top|bottom",
  "originals": [
    {
      "role": "physical_product_authority|approved_on_model_source|logo_authority",
      "local_path": "<repo-relative path>",
      "sha256": "<exact SHA-256 of the retained local original>",
      "bytes": 0
    }
  ],
  "product_sot_sha256": "<current SOT SHA-256>",
  "product_hash": "<current SKU product hash>",
  "model": "gpt-image-2",
  "operation": "candidate_only",
  "output": {
    "local_path": "<quarantined local candidate path>",
    "sha256": "<candidate SHA-256>",
    "bytes": 0
  }
}
```

The receipt validator must recompute every original and output digest from the
local path.  It must reject URLs, paths outside the repository, missing files,
hash or byte drift, an empty originals list, stale product bindings, and any
attempt to make a derived asset the sole product authority.  Originals remain
immutable; optimization may create a separate derived pack only after the
original receipt is written.

## Registry hardening required before enforcement

The executable intake gate is
`scripts/validate-on-model-media-intake.py`.  It rejects a media record whose
source does not resolve to the SKU's SOT front asset, whose SHA-256 or byte
count drifts, or whose product SOT/product hash is stale:

```bash
python3 scripts/validate-on-model-media-intake.py \
  --product-sot data/product-sot.json \
  --registry wordpress-theme/skyyrose-flagship-2/data/opening-product-media.json
```

Add `--require-explicit-approval` for promotion, campaign, or any other use
that needs a per-view approval reference.  The legacy three-card record is
accepted only in ordinary card-support mode because it has a dated review but
no per-view approval ID; strict mode deliberately blocks it until a human
registers that missing reference.

Read-only validation against the matching V2 worktree on 2026-08-25 reported
`SUPPORTED=3 BLOCKED=30` in ordinary card-support mode and
`SUPPORTED=0 BLOCKED=33` with `--require-explicit-approval`.

The existing support helper recognizes an approved front solely from the
presence of a view with role `on_model_front`; the active V2 checkout must call
this validator before it treats that role as supported.  This worktree is
intentionally older than the V2 product-SOT/registry revision cited above, so
the command is a handoff gate rather than an executable storefront change here.
