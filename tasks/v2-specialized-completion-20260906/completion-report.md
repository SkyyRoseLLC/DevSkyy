# Specialized completion — four independent ownership tracks

Accepted starting point: V2-READINESS-20260906. Phase: V2-SPECIALIZED-COMPLETION-20260906. Overall V2 remains NEEDS_MORE_WORK, but the four deliverables have independent outcomes.

| Track | Accountable next owner | Status | What this means |
|---|---|---|---|
| FRONTEND PERFORMANCE | Frontend performance engineer, with independent visual/native-commerce review | **NEEDS_MORE_WORK** | Priority routes still fail LCP. Scratch CSS delivery experiment rejected; no accepted-theme change. |
| PLATFORM READINESS | Hosting / WordPress.com platform engineer | **READY_TO_VERIFY_ON_STAGING** | Exact checklist and evidence contract ready. All 12 remote checks NOT_RUN; staging untouched. |
| ASK SKYY ASSET | Blender character optimization artist, technical animator and independent visual reviewer | **READY_FOR_BLENDER** | Exact canonical input/runtime backups and dedicated phase contract verified. No import or model optimization performed. |
| PRODUCT CONTENT | Editorial/product-truth owner; founder decides media clearance | **READY_FOR_FOUNDER_CONTENT_REVIEW** | All 33 SKUs × 11 fields and five-SKU visual decision queue ready. Content remains PARTIAL; no media approval or promotion. |

## A — application-owned performance

The bounded experiment moved exact theme-owned stylesheet bytes into the document on Home, Shop, Collection and PDP, preserving cascade and relative asset URLs. Native Woo resources, media and accepted theme files were untouched. This is an isolated local router experiment, not a theme implementation or platform change. It tests how much the theme stylesheet discovery chain contributes without substituting server compression or native dependency deferral.

The fresh Home sample moved LCP from 5,436ms to 5,118ms and FCP from 4,225ms to 3,460ms, while document bytes rose from 97,753 to 244,030. That is an unfavorable cross-page cache tradeoff and does not meet the 2,500ms LCP gate. Priority Shop/Collection and secondary PDP also remain over target in the paired experiment. See [Track A measurements and visual evidence](track-a-performance.md) for all exact route results, profile, excluded samples and capture limits.

**No optimization was adopted.** A timing improvement alone cannot waive visual equivalence, native commerce or route budgets. The accepted source remains the readiness candidate; its previous six-route plain/gzip results remain the current application baseline, not superseded by the scratch variant. No broad cleanup, creative removal, Cart/Checkout creative change or speculative native-script deferral occurred.

Next work remains a narrow critical subset or single confirmed resource-discovery intervention on Home, Shop and Collection, with measured ownership, paired runs and EQUIVALENT/BETTER visual gates before adoption. PDP remains secondary. Server/CDN transport belongs to Track B. No sitewide frontend PASS is inferred from the prepared other tracks.

## B — exact staging verification procedure

[Platform checklist](track-b-platform-checklist.md) and [machine-readable check contract](track-b-contract.json) cover Brotli/gzip/identity negotiation, static decoded hashes, Cache-Control, conditional responses, immutable version keys, Vary, content type, byte ranges, actual CDN URLs, HTML/session cache exclusions, Cart/Checkout personalization and actual TTFB.

The procedure requires an authorized target and deployed source identity, preserves privacy/noindex, uses existing authorized synthetic sessions, and forbids changes or payment/order actions. It records route/network/protocol/encoding/cache/session context separately. There are twelve NOT_RUN checks and a 262-row local resource selection aid; local URLs are not represented as verified staging URLs. The documented WordPress.com compression expectation is a platform expectation, not an observed response. Official current references and precise command recipes are included.

READY_TO_VERIFY_ON_STAGING means the procedure is ready for a separately authorized staging verification. The target origin and session inputs must be bound at execution. No staging origin was queried, configuration changed, cache purged or asset deployed during this phase.

## C — dedicated Blender phase

[Production checklist](track-c-blender-phase.md), [contract](track-c-contract.json) and [preservation evidence](track-c-evidence.json) establish ten immutable-by-policy backups. Nineteen checksum checks passed across those backups and nine unique originals. Separate canonical-input and runtime role copies both match the accepted GLB; these roles currently point to the same source file. No upstream `.blend` has been identified, and no historic substitute was selected.

Current asset: 1,930,256 triangles, 1,030,595 vertices, 6,058,568 bytes, 18 bones, one material/draw call and approximately 89.2MiB decoded lower bound. Target: **80k–120k triangles where fidelity permits and ≤2,621,440 bytes**. Preserve face, hair silhouette, hands, clothing silhouette and major folds in that priority order, plus proportions, rig identity and practical current animation compatibility. Reduction is aggressive only in visually safe regions, not a blind global ratio.

Blender, glTF import support and Draco bridge availability were checked. Actual import is the first gated operation of the next phase. Work starts from a new working copy; export goes to a separate candidate path. Import parity, topology, deformation, export size, runtime cadence, poster continuity and founder disposition remain unperformed gates. Source, runtime software and model are unchanged.

## D — editorial and media authority

[33-SKU matrix](track-d-content-matrix.md) and [full evidence JSON](track-d-content-matrix.json) classify short description, material, construction, fit, care, story, supported styling note, gallery, approved detail imagery, size and preorder state. Every field is COMPLETE, PARTIAL or MISSING with source evidence. All 33 products remain PARTIAL overall.

| Newly explicit field | COMPLETE | PARTIAL | MISSING |
|---|---:|---:|---:|
| Construction | 23 | 10 | 0 |
| Supported styling note | 2 | 5 | 26 |
| Approved detail imagery | 0 | 0 | 33 |

Care is also MISSING for all 33; ten source conflicts remain. Missing claims were not inferred from image appearance, garment category or scene poses. Existing synthetic Woo empty descriptions remain a local-fixture observation, not a production content assertion. Field-level COMPLETE does not mean the product is complete or approved for publication.

The [separate five-SKU queue](track-d-media-review-queue.json) covers BR-001, BR-003, BR-004, BR-007 and BR-011. The visual sheet exposes ten local image references, including duplicate-byte source/card copies, with provenance, hashes, current usage scope and rejection reasons. These are candidate evidence for founder disposition, not ten unique newly approved choices. No PDP clearance is inferred from card/ad/web authority. Where candidates still visibly conflict with the specification, that conflict remains explicit. **CREATIVE ASSET BLOCKER — BR-003** remains.

Ten original media links were converted to byte-identical local review-copy links for HTTP usability. This is review delivery only; canonical media, PDP routing and approval manifests were not changed. Founder decisions remain pending and are not automatically submitted by the static review page.

## Independent review and protected state

[Independent B/C/D review](independent-review.md) approved the bounded deliverables after the HTTP image-link correction. It independently verified 33×11 totals, 455 file-hash references, 429 line-anchor bounds, all ten embedded/review-copy image hashes, and all 19 Skyy preservation checksum entries. Editorial visual review was representative and explicitly bounded; it did not approve media authenticity or publication.

Protected state includes five animated heroes, nine scenes, card identity, native Quick View, scene-to-commerce handoffs, current Skyy software and Town Line source. Final source/binary recheck and the new evidence index bind this phase to the accepted source. With no accepted application code change, a redundant full build/test rerun would not establish a new result; prior accepted test results remain labeled prior evidence. The scratch experiment has its own new measurements and failure records.

No deployment, staging alteration, destructive model work, catalog/media promotion, Town Line authoring or cinematic redesign occurred. Each track now has its own owner, evidence, next operation and approval boundary.

## Final review delivery verification

The combined local review, platform overview, Blender handoff and media sheet passed eight desktop/mobile browser checks at390/1440px: zero page exceptions, broken images or document overflow. A mobile long-text/table wrapping issue in the served Blender review was fixed in the review-only integration; its initial failure receipt remains. No theme styling was changed. Track A supplies24 fresh diagnostic captures: six settled baseline/inline pairs are byte-identical; one early Collection heading raster difference converges after settling. Root independently viewed the mobile Home settled pair and found no visible composition change in that bounded state; this does not certify the rejected experiment for adoption.
