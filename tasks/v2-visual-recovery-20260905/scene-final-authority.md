# Nine approved Scroll World scenes: final source authority

Status: **APPROVED SOURCE SELECTION VERIFIED; INTEGRATED PIXEL/COMMERCE/PERFORMANCE QA REMAINS REQUIRED.**

The final founder directive `3b686012-fe90-4073-ae81-e1287755a211` is implemented as nine exact scene records: three Signature, three Black Rose and three Love Hurts. No Kids chapter was invented. The four historical collection-entry portraits are HOLD for founder review and are not integrated. Home reuses three members of the exact nine-scene set.

## Evidence chain

- `tasks/v2-source-certification-20260905/provenance/collection-motion-wiring-k1--approval.json` records explicit local authorization for the existing approved motion scenes.
- The adjacent `collection-motion-wiring-k1--receipt.json` names all nine IDs, three per collection, original source URLs, and 18 imported delivery hashes. Its status is explicitly `LOCAL_WIRING_IMPLEMENTED_NOT_RUNTIME_VERIFIED`; it grants no deployment or visual-QA claim.
- Immutable recovered commit `bb0e6eecf0af54b3f0c14e5eadb7b138530ce09b` contains `data/collection-scene-motion.json` with the exact nine records, each explicitly `founder_approved_visual: true` and `local_wiring_authorized: true`. The repaired B foundation preserves this selection.
- `.artifacts/v2-visual-recovery-20260905/selected-runtime-scenes.json` is actual local guarded WordPress resolver output: exactly these nine chapters, no model overlays, no candidate srcsets, no active placeholders. It is a source-selection attestation, not final browser fidelity evidence.
- The new canonical manifest binds each approval/receipt/selection document by SHA-256, then each exact required poster and desktop/mobile delivery file by SHA-256 and byte count.

Approval is therefore grounded in explicit approved-visual/local-wiring records plus the latest founder contract, not in archive presence, a filename, or a historical package. Historical candidate naming in an exact selected poster path does not promote its other variants.

## Exact final scene structure

| Scene | World | Existing label | Native SKU bindings | Poster bytes | Desktop film bytes | Mobile film bytes |
| --- | --- | --- | --- | ---: | ---: | ---: |
| SIG-COMMERCE-1 | signature | The Golden Gate Overlook | sg-009, sg-007 | 255858 | 3033789 | 1222252 |
| SIG-COMMERCE-2 | signature | The Lateral Terrace | sg-013, sg-014, sg-006 | 310334 | 3529477 | 1548209 |
| SIG-COMMERCE-3 | signature | The Departure Terrace | sg-001, sg-005, sg-003, sg-002, sg-015 | 263964 | 3476074 | 1589393 |
| BR-COMMERCE-1 | black-rose | The Type Foundry | br-001, br-002 | 133642 | 1209097 | 567028 |
| BR-COMMERCE-2 | black-rose | The Moonlit Waterfront | br-005, br-007, br-004 | 289988 | 3354874 | 1605812 |
| BR-COMMERCE-3 | black-rose | The Town Line | br-008, br-009, br-010, br-011, br-012 | 2482505 | 2997756 | 1456483 |
| LH-COMMERCE-1 | love-hurts | The Vow Aisle | lh-004, lh-002, lh-006 | 113872 | 3606213 | 1798799 |
| LH-COMMERCE-2 | love-hurts | The Rose Side Chapel | lh-003 | 340456 | 3473611 | 1671544 |
| LH-COMMERCE-3 | love-hurts | The Rose Vitrine | lh-005 | 87158 | 1387412 | 619697 |

Total retained scene media: **42,425,297 bytes across 27 assets**. This is the entire nine-scene delivery library, not first-page transfer. The 18 videos are **38,147,520 bytes**; the nine selected posters are **4,277,777 bytes**. Per-scene initial/deferred browser transfer, decode, frame, layout and memory cost must still be profiled by the runtime owner.

Every record preserves the approved baked composition with `contain`, portrait/landscape source dimensions, and distinct mobile delivery. Story and real WooCommerce links sit below media; animation completion is never required. Mobile visual approval and runtime behavior remain to be verified; writing this contract does not certify those outcomes.

**Town Line boundary:** BR-COMMERCE-3 already carries the label “The Town Line” and an approved five-jersey lounge composition. It remains the third established Black Rose scene. This is not permission to build the separate final Town Line Pre-Order film/story/layout, which remains preserved for a later founder specification.

## Packaging review: safe, specific candidates

The current approved motion/generated-composition directories have no additional released poster/video selections beyond these 27. The four selected composition PNG masters were already excluded from the verified B package. Do not remove selected WebPs merely because their parent directory is named `generated-candidates`.

These five historical placeholder PNGs are released in B but superseded in the resolved nine-scene chain. Recommend marking them source-only in package policy after the runtime canonical filter is installed and validated; retain original source bytes and history. No exclusions or deletions were made by this subtask.

| Superseded placeholder path | Bytes |
| --- | ---: |
| `assets/scroll-world/placeholders/founder-selected-v1/black-rose-east-oakland-crewneck-v1.png` | 2106007 |
| `assets/scroll-world/placeholders/founder-selected-v1/love-hurts-graphic-heart-backdrop-v1.png` | 1835160 |
| `assets/scroll-world/placeholders/founder-selected-v1/love-hurts-star-heart-statue-backdrop-v1.png` | 1924573 |
| `assets/scroll-world/placeholders/founder-selected-v1/signature-bay-bridge-shorts-v1.png` | 2273134 |
| `assets/scroll-world/placeholders/founder-selected-v1/signature-graphic-monogram-backdrop-v1.png` | 1950716 |

Total candidate package saving: **10,089,590 bytes**. This is release size, not measured network transfer savings. The legacy placeholder manifest must not re-enable these as runtime fallbacks.

Other scroll-world files are not classified as debris solely because they are outside this nine-scene set. Preserve the four entry portraits as source only; latest authority does not select them for runtime. `scene-5-finale.webp` has no exact runtime literal found, but dynamic usage and approval are not established: **UNKNOWN**, no automatic integration or deletion. BR-006 source-footage, backdrop and web-loop assets belong to a separate product/story system and require its owner’s current-use check before any package exclusion.

## Guard and verification

`tools/v2-runtime/check-approved-scroll-world-scenes.cjs` validates schema, exact IDs/order/world, explicit approval flags and hash-bound evidence, exactly three selected media roles, asset/receipt hashes, containment, controller, native CTA contracts and motion manifest parity. `--resolved FILE` additionally rejects missing/extra runtime chapters, foreign poster/video paths, legacy overlays/srcsets/placeholders and SKU binding drift.

The root runtime owner must consume the canonical manifest and fail closed before rendering unknown scenes. The guard alone cannot stop a hard-coded unrelated image in a template; browser request/DOM evidence remains necessary. Guard output says whether a resolver snapshot was checked and explicitly disclaims browser visual QA.

Verification executed: `node --test tools/v2-runtime/test-approved-scroll-world-scenes.cjs` **14/14 PASS**; `node tools/v2-runtime/check-approved-scroll-world-scenes.cjs --resolved .artifacts/v2-visual-recovery-20260905/selected-runtime-scenes.json` **PASS**, nine resolved chapters / 27 asset hashes. This subtask changed only its three owned files plus this evidence document; it did not alter PHP, package policy, build pins, deployment or source media.

## Scoped BR3 poster delivery optimization

The root explicitly allowlisted both output paths before generation. Only the existing approved BR-COMMERCE-3 PNG was resized. Parent bytes remain 2482505 and SHA-256 `90aa66aa432bc9176b393dfbb7ce15d507860e47c970650337f78c4b1a480683`. No crop, paint, generation, compositing, source replacement or new scene selection occurred.

| Delivery | Dimensions | Bytes | Reduction versus original | SHA-256 |
| --- | --- | ---: | ---: | --- |
| assets/scroll-world/derived/approved-scenes/br-commerce-3-poster-640w.webp | 640 × 427 | 69660 | 97.19% | fe92f8838cb20f911e95afc2f1ab75ad1bc94601d3a00b0db7bf9c9165abd1a4 |
| assets/scroll-world/derived/approved-scenes/br-commerce-3-poster-1024w.webp | 1024 × 683 | 156866 | 93.68% | 2cbf726ea73c1c0a3eead460228b302d07f6ea51ebd3e5a70c9380515e25b9d8 |

Two supporting derivatives add **226,526 bytes** to the retained source delivery library; the canonical nine-scene/27-media selection remains unchanged. Network savings depend on root wiring and browser source selection. These byte reductions are not measured LCP improvements.

The full parent image and 640px derivative were visually inspected. The five garments, Bay Bridge skyline, furniture, lighting and frame boundaries remain in the same composition. Lower-resolution fabric/lettering detail is expected; final integrated desktop/mobile browser fidelity review remains independent. Aspect ratios differ only by the unavoidable integer output-height rounding from cwebp (640×427 and 1024×683).

Reproduce with `node tools/v2-runtime/build-scene-posters.cjs --check` (read-only verification) or `--write` (regenerate exactly the two allowlisted files). Toolchain pin: cwebp **1.6.0**, libsharpyuv **0.4.2**, `-q 90 -m 6 -resize WIDTH 0`. The builder encodes to temporary files, checks committed hashes, dimensions and source provenance, and refuses a toolchain mismatch or unauthorized destination. The original is never rewritten.

The updated guard validates only these two source-bound supporting records and their output dimensions/hashes. It still rejects extra historical srcsets and extra scene IDs; canonical responsive delivery is a separately declared field. Extended suite: **19/19 PASS**. Reproducible encoding check: **PASS**. Existing captured PHP source selection: **PASS**, nine scenes / 27 canonical media / two supporting derivatives. Root owns adding the script to build/verification and connecting the srcset; this document does not claim that work is complete.
