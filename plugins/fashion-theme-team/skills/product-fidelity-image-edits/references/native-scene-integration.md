# Native collection-scene integration

This contract governs cinematic commerce scenes in which exact garments must look photographed inside the collection world. It complements Fashion Theme Team direction and handoff rules; it does not replace them.

## Required chain

1. **Discover** — visually classify every source; bind SKU, view, role, path, hash, rights/review state, and exact product features. Reject stale, mislabeled, or previously rejected inputs.
2. **Direct** — define the collection story, scene role, exact cast, CTA ownership, garment-first focal order, and how the scene differs from the other collection chapters.
3. **Analyze the source** — record output geometry, subject box, horizon, foot/contact points, lens class, camera height, key-light direction, color temperature, shadow direction/softness, floor response, edge quality, and occlusion opportunities.
4. **Contract** — choose one integration strategy and write the optical contract before a prompt exists.
5. **Prompt** — describe one camera and one light field. Build the environment around the source geometry. State exact protected product features verbatim and forbid unrelated product invention.
6. **Preflight again** — recompute every input hash after prompt authoring. Any drift invalidates the prompt and reference pack.
7. **Generate** — produce a review candidate only. Generation does not authorize wiring.
8. **Inspect** — review at full size for product features and native integration as separate dimensions.
9. **Founder review** — only the founder can approve promotion. Wiring, commit, staging, deployment, and commercial approval remain separate actions.

## Integration strategies

### `environment_rebuild_around_source`

Preferred when an approved on-model source already has the required pose. Keep the approved subject geometry fixed and build the world from its camera, horizon, floor contacts, and light. Do not generate an arbitrary plate first and then search for a place to paste the model.

### `protected_garment_reconstruction`

Use only when a genuine garment matte can remain protected while non-garment pixels are reconstructed to share the scene. The garment, logos, trim, embroidery, patches, materials, and silhouette must pass product review. The body and environment must pass anatomy and native-integration review. Pose changes remain blocked unless a same-pose authority or verified garment/character 3D route exists.

### `full_native_redraw_candidate`

Use only with explicit founder authorization to create a non-exact review candidate. It can explore composition but cannot claim pixel-exact product fidelity or enter final wiring without a later exact-product correction and complete re-review.

## Optical contract

The JSON contract records:

- output width/height and the intended subject bounding box;
- horizon Y and at least two foot/contact points in output pixels;
- lens class: `wide`, `normal`, or `telephoto`;
- camera height: `low`, `eye_level`, or `high`;
- key-light direction, color temperature, and softness;
- shadow direction and softness;
- floor response: `matte`, `satin`, `polished`, `wet`, or `textured`;
- reflection mode: `none`, `diffuse`, `subtle`, or `mirror`;
- explicit requirements for contact shadow, edge spill, depth occlusion, atmospheric depth, and consistent scene grain;
- exact product priority, collection motif, and anti-generic exclusions.

Coordinates must be integers inside the declared output geometry. Subject and contact geometry is a prompt constraint and a review target, not proof that a model obeyed it.

The contract target must be the same hash-bound approved on-model file declared as a `physical_product_authority`. Every SKU in the scene cast requires a `physical_product_authority` or `approved_techflat`; a logo, patch crop, garment matte, dossier, filename, or contact sheet cannot satisfy product truth by itself.

## Independent review receipt

The candidate review JSON uses schema `product-fidelity-native-review.v1` and binds the candidate SHA-256. It contains a reviewer distinct from the author, five 0–100 scores, hard-fail findings, and a founder status.

Required score dimensions:

- `product_fidelity`
- `optical_integration`
- `anatomy_pose`
- `collection_story`
- `commerce_readiness`

Default minimums are 95, 90, 90, 90, and 90. Any hard fail blocks the candidate regardless of average. Founder status remains `PENDING`, `APPROVED`, or `REJECTED`; a technically passing candidate with `PENDING` is only `PASS_REVIEW_CANDIDATE` and is not wiring-approved.

## Native-integration hard fails

- wrong SKU, view, wording, logo, embroidery, print, patch, trim, material, color, or construction;
- floating feet, duplicated limbs, broken anatomy, unsupported pose change, or mismatched body scale;
- camera/horizon/perspective mismatch;
- missing or impossible contact shadow;
- key light, spill, reflection, grain, depth, or edge treatment inconsistent with the scene;
- subject placed in a decorative pocket with no physical relationship to the environment;
- rejected or stale input reused as authority;
- a scene that repeats another chapter instead of advancing the collection story;
- a CTA/cast mismatch or invented product.
