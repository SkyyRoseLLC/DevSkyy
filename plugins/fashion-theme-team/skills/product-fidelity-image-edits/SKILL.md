---
name: product-fidelity-image-edits
description: Prepare, route, and verify exact-product image, native collection-scene, video, and 3D work when logos, embroidery, prints, patches, construction, protected pixels, photographic integration, temporal identity, geometry, or PBR materials must not drift. Use for bounded product patches, model-in-scene work, collection campaigns, image-to-video, product animation, reference packs, and image-to-3D. Do not use for unrestricted concept art where product fidelity is not required.
---

# Product Fidelity Media

Use this skill when a visually plausible rewrite is not sufficient. It routes image, video, and 3D work through explicit capability and fidelity contracts, then fails closed when the selected tool cannot enforce the required invariants.

Before choosing a provider or model, read [model selection](references/model-selection.md) and query the machine-readable [model capability registry](references/model-capability-registry.json). Model names and APIs are volatile; an unknown or stale model is not an implied match.

## Route first

Choose exactly one image mode:

1. **Localized product patch** — wording, logo, embroidery, patch, print, panel, trim, or construction changes inside an existing image.
   - Requires an explicit editable-region mask aligned to the target.
   - Requires a mask-capable editor or deterministic compositor.
   - Requires zero changed pixels outside the authorized mask and zero per-channel comparison tolerance. Contracts cannot weaken this policy.
   - Built-in Imagegen is not allowed for this mode because it cannot accept an explicit mask or prove outside-mask preservation.

2. **Native collection scene** — rebuild a world around an approved on-model source while preserving product truth and proving that the result reads as one photograph.
   - Requires a collection/story contract, exact SKU cast, current source hashes, and one source view per product authority.
   - Requires a measured camera, horizon, subject-scale, floor/contact, light, shadow, reflection, occlusion, and edge-integration plan before prompting.
   - Prefer environment reconstruction around the approved source. A pose change is blocked unless the approved product already exists in that pose or a verified 3D garment route can preserve construction.
   - Requires separate product-fidelity and native-integration review receipts. A technically preserved garment that reads as a pasted cutout is `BLOCKED`.
   - Read [native collection scenes](references/native-scene-integration.md) before authoring the contract.

3. **Protected scene composite** — replace or regenerate the environment while the exact approved product/model layer remains unchanged.
   - Generate the environment without the protected product.
   - Composite the real-alpha protected layer afterward.
   - Verify opaque protected pixels remain byte-identical.
   - A pose change is blocked unless an approved product/model source already exists in that pose. A generative pose change redraws the garment and is not exact-product preservation.
   - This route is a layout/proof method by default. It is not a final commerce scene until the native-integration gate also passes; byte-identical RGB alone is insufficient.

If neither mode fits, stop and report that the request needs founder-approved redraw authorization or new source photography.

For motion, use [video fidelity](references/video-fidelity.md). For a true 3D asset, use [3D fidelity](references/3d-fidelity.md). Never treat an animated still, turntable video, depth warp, Gaussian splat, NeRF, or generated multi-view sheet as proof of a correct mesh.

## Required workflow

1. Inspect the target and every reference visually. Filenames are never product truth.
2. Create a v2 JSON contract using [the contract schema](references/contract-schema.md). Record explicit `sku`, `view`, `role`, path, SHA-256, `verbatim_text` declaration, and hash-bound source-authority receipt for each product reference. The authority receipt must bind the exact source bytes to the current catalog, imagery SOT manifest, governing dossier, and resolver version. For collection scenes, also bind collection, scene, exact SKU/CTA cast, source analysis, optical contract, promotion boundary, and independent-review rubric.
3. Resolve the declared provider/model against the registry. Run `model_registry.py validate` before any paid call. If a provider exposes runtime model discovery, record that receipt too; the registry is a policy floor, not proof of live availability.
   - For a one-shot OpenAI localized patch, use the Image API `/v1/images/edits` and pin the reviewed GPT Image 2 snapshot. Do not substitute the Responses image tool or a wrapper-only route.
   - Omit `input_fidelity` for GPT Image 2. Bind exact target/output geometry, prompt hash, allowlisted request parameters, the ordered target-plus-product-authority request images, API-ready alpha mask, output, request ID, and SDK/client version in a credential-free generation receipt.
4. Normalize misleading filenames into canonical pack names. Never allow a file named `back` to enter a `side` job without canonical renaming and explicit `view: side` metadata.
5. Run:

   ```bash
   python3 scripts/fidelity_gate.py preflight --contract /absolute/path/contract.json --workspace /absolute/path/workspace
   python3 scripts/fidelity_gate.py optimize --contract /absolute/path/contract.json --workspace /absolute/path/workspace --out-dir /absolute/path/reference-pack
   ```

6. For a localized patch, inspect the generated mask overlay and provider mask before any paid or remote generation. For OpenAI, the target and provider mask must both be PNG and below 50MB; the same-size mask uses alpha zero only inside the editable region and alpha 255 outside it. The requested output size must exactly match the valid target geometry. Do not continue if the mask covers the face, body, other garment regions, scene, or product features that should remain locked.
7. Generate only through the contract-declared route. Do not silently fall back from masked editing to a full-frame semantic edit.
8. Verify the result. Localized patches require a v2 contract- and candidate-bound independent semantic-review receipt with the exact contracted check set and verbatim-text result. Native collection scenes require a v2 hash-bound independent visual-review receipt:

   ```bash
   python3 scripts/fidelity_gate.py verify --contract /absolute/path/localized-contract.json --workspace /absolute/path/workspace --output /absolute/path/candidate.png --generation-receipt /absolute/path/generation-receipt.json --review /absolute/path/semantic-review.json
   python3 scripts/fidelity_gate.py verify --contract /absolute/path/native-scene-contract.json --workspace /absolute/path/workspace --output /absolute/path/candidate.png --review /absolute/path/native-review.json
   ```

9. Quarantine any result with dimension drift, outside-mask drift, protected-pixel drift, text/logo mismatch, source-hash drift, temporal identity drift, view inconsistency, geometry error, texture/PBR mismatch, camera mismatch, floating feet, missing contact shadow, inconsistent reflections, light/spill mismatch, implausible occlusion, pasted edges, or weak collection-story continuity. Never use a rejected result as the next edit target.
10. Keep founder approval, wiring, commit, deployment, and commercial approval separate.

## Reference-pack rules

- One physical SKU and one declared view per authority image.
- Founder-shot physical product first; approved techflat second; exact logo/patch registry and dossier only as declared fallbacks.
- Multi-SKU proof boards may be `context_only`; they cannot be the sole `physical_product_authority` for an exact patch.
- Preserve original source files and hashes. Optimized copies are derived references, never replacements for source truth.
- Protected layers require genuine alpha. An opaque checkerboard or baked background is not a protected layer.
- Exact in-image wording must be included verbatim in the contract.
- A full-model alpha is not a native-scene guarantee. Preserve an independently extracted garment matte when possible, and design the environment from the approved source camera rather than generating an unrelated plate.
- A rejected scene, layout proof, contact sheet, or filename-only candidate is `context_only`; it can never become product or scene authority.

## Stop conditions

Stop without generating when:

- a localized edit has no reviewed mask;
- the selected generator cannot consume a mask;
- the reference view is unknown or conflicts with the prompt;
- a protected layer lacks real alpha;
- a requested pose change has no approved same-pose product source;
- a native scene lacks a measured optical contract, exact SKU/CTA cast, distinct collection-story role, or independent native-integration review plan;
- a final-commerce scene relies only on full-model alpha placement against an independently generated plate;
- any source hash changed after prompt authoring;
- any product reference lacks a current hash-bound catalog/SOT/dossier authority receipt;
- the candidate cannot be compared at the same geometry as the target;
- the provider/model is absent, deprecated, stale, unavailable, or lacks the required capability;
- video generation has no locked first frame or per-frame SKU verification plan;
- a 3D request has only one view when unseen regions contain construction or artwork;
- a generated mesh is called an exact replica without front/side/back render, geometry, UV, and material proof.

For the failure pattern this skill prevents, read [root cause and prevention](references/root-cause.md). Fashion storefront story, collection direction, HTML/JSON handoff, and independent commerce QA remain owned by the Fashion Theme Team; this skill owns the media truth and native-scene gate beneath that motherbase.
