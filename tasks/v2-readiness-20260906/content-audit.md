# V2 readiness: all 33 SKU content audit

Status: **PARTIAL CONTENT — editorial/founder input required**. This audit changed reports only. It did not change catalog, dossiers, theme, media bindings, or live/staging data.

## Authority and method

Product identity/structured values come from SOT.md → flagship catalog CSV; garment facts come from each named founder dossier. Existing V2 opening-product-media rejection is preserved. Hub/card approval is recorded with its actual scope and hashes, never promoted to PDP. Current native fixture was queried with SELECT statements in a MySQL READ ONLY transaction, without loading WordPress or querying customer/order tables. The old SQLite snapshot was excluded after confirming active configuration uses MySQL.

**COMPLETE** means authoritative storefront-ready content for that field; **PARTIAL** means a seed/specification exists but a conflict, missing detail, measurement, multi-view approval or commerce-authority boundary remains; **MISSING** means no usable content exists or explicit rejection prohibits use. Overall COMPLETE requires every field COMPLETE. Garment type is distinguished from synthetic simple/variable setup. Size options are distinguished from measured guidance.

All 33 named dossiers and all 33 canonical CSV descriptions exist. The live local synthetic fixture currently has **33 empty short descriptions and 33 empty full descriptions**, 31 variable products and 2 simple products. These raw observations supersede any inference from six representative route screenshots. No fixture text is treated as founder content.

Overall: {'PARTIAL': 33}. Canonical pre-order SKUs: 15.

## Per-SKU matrix

| SKU | Short | Material | Fit | Care | Story | Gallery | Size | Type | Preorder | Overall |
|---|---|---|---|---|---|---|---|---|---|---|
| BR-001 | PARTIAL | PARTIAL | COMPLETE | MISSING | PARTIAL | MISSING | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-002 | PARTIAL | PARTIAL | COMPLETE | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-003 | PARTIAL | PARTIAL | PARTIAL | MISSING | PARTIAL | MISSING | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-004 | PARTIAL | PARTIAL | PARTIAL | MISSING | PARTIAL | MISSING | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-005 | PARTIAL | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-006 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-007 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | MISSING | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-008 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-009 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-010 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-011 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | MISSING | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-012 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| LH-002 | COMPLETE | PARTIAL | COMPLETE | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| LH-003 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| LH-004 | PARTIAL | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| LH-005 | COMPLETE | COMPLETE | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| LH-006 | COMPLETE | PARTIAL | COMPLETE | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-001 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-002 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-003 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-005 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-006 | PARTIAL | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-007 | PARTIAL | PARTIAL | COMPLETE | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-009 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-011 | COMPLETE | COMPLETE | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-012 | COMPLETE | COMPLETE | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-013 | PARTIAL | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-014 | COMPLETE | PARTIAL | COMPLETE | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| SG-015 | PARTIAL | COMPLETE | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| KIDS-001 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| KIDS-002 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-014 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |
| BR-015 | COMPLETE | PARTIAL | PARTIAL | MISSING | PARTIAL | PARTIAL | PARTIAL | PARTIAL | COMPLETE | PARTIAL |

## Field totals

| Field | COMPLETE | PARTIAL | MISSING |
|---|---:|---:|---:|
| short_description | 23 | 10 | 0 |
| material | 4 | 29 | 0 |
| fit | 6 | 27 | 0 |
| care | 0 | 0 | 33 |
| story | 0 | 33 | 0 |
| gallery | 0 | 28 | 5 |
| size | 0 | 33 | 0 |
| product_type | 0 | 33 | 0 |
| preorder | 33 | 0 | 0 |

## Founder/content queue

**CREATIVE ASSET BLOCKER — BR-003 remains.** No PDP-specific approval overriding any of the five current rejection records was found in the audited authorities. Every candidate record, usage scope, existing file hash and approval-record hash is in content-review-queue.json. Missing local blob hashes remain null rather than invented.

- **BR-001**: Founder correction requires an embossed BLACK Rose crewneck treatment, but every current br-001 product and on-model source depicts a contrasting embroidered rose/cloud graphic. No current source is replica-eligible. Preserve rejected/unavailable PDP state; card/hub verification is insufficient to lift this gate.
- **BR-003**: Current on-model front suppresses canonical patch color. Preserve rejected/unavailable PDP state; card/hub verification is insufficient to lift this gate.
- **BR-004**: Current on-model front does not match the corrected hoodie: no circular upper-arm patch is permitted; the small wearer-left chest mark, longitudinal wearer-left forearm artwork, light floral hood lining, and white drawstrings must remain exact. Preserve rejected/unavailable PDP state; card/hub verification is insufficient to lift this gate.
- **BR-007**: Current front has incorrect logo color and the back omits required Love Hurts placements. Preserve rejected/unavailable PDP state; card/hub verification is insufficient to lift this gate.
- **BR-011**: Current front invents a prohibited NHL shield at the collar. Preserve rejected/unavailable PDP state; card/hub verification is insufficient to lift this gate.

## Canonical conflicts requiring founder resolution

- **BR-001**: CSV short copy says embroidered; founder dossier explicitly requires embossed front chest, not embroidered.
- **BR-002**: CSV short copy says embroidered roses; founder dossier specifies silicone thigh patch.
- **BR-003**: CSV short copy lists four colorways together; dossier assigns Oakland/Giants/White to separate SKUs. Clarify product versus series copy.
- **BR-004**: Current media-rejection reason specifies wearer-left chest/forearm/floral lining/white strings; named dossier describes a different chest treatment/clean sleeves. Reconcile authority without lifting rejection.
- **BR-005**: CSV promises numbered tag; named dossier contains no numbered-tag specification. Founder confirmation required before enrichment.
- **LH-004**: CSV says satin bomber; named dossier explicitly says NOT a satin bomber.
- **SG-006**: CSV says mint/lavender colorblock; dossier requires a solid mint body and no rainbow chevrons.
- **SG-007**: CSV rose variants red/grey/black/purple do not map cleanly to dossier purple, white-outline, single-red, red-cluster variants.
- **SG-013**: CSV says colorblock; dossier explicitly prohibits color-blocking. Dossier cross-reference says sg-006 has rainbow chevrons, conflicting with the current sg-006 dossier.
- **SG-015**: CSV color is Black; dossier explicitly requires white body, pink hood and multicolor chevrons.

## Editorial next actions

1. Resolve the listed contradictions before drafting richer copy. Preserve the named-dossier authority and rejection truth rather than reconciling by guessing.
2. Supply care instructions for all 33 products. Confirm composition/component specifications where only fabric texture/weight is known.
3. Supply garment/accessory measurements and customer fit guidance; existing size options alone are not a measured size guide.
4. Approve SKU-specific story and gallery plans, especially the five rejected PDPs. Native fixture primary attachments are test data, not new media approval.
5. After founder content approval, perform a separately authorized sync into native Woo fields; this audit performs no synchronization.

## Evidence

- `.artifacts/v2-readiness-20260906/content/all-33-sku-content-audit.json`: 33 rows × nine field classifications, exact source excerpts/paths/hashes, canonical-versus-native data, and conflicts.
- `tasks/v2-readiness-20260906/content-review-queue.json`: five-SKU authority queue plus editorial worklist.
- `.artifacts/v2-readiness-20260906/content/synthetic-native-wp-products.json`: active SELECT-only synthetic product export; no credentials/customer/order content.
- `.artifacts/v2-readiness-20260906/content/read-fixture.php` and `build-audit.py`: reproducible audit scripts; reports only.

Boundary: `data/product-sot.json` referenced by opening-product-media.json is absent in this worktree. Embedded historical product hashes were not called current verified authority. Production WooCommerce content was not read; conclusions about empty native fields apply to this local synthetic fixture only.
