# Model selection and capability policy

The registry is a dated capability map, not a marketing leaderboard. A model may make beautiful media and still be disallowed for product truth.

## Selection order

1. Define the invariant: exact pixels, bounded edit, identity across frames, multi-view consistency, or true geometry.
2. Prefer deterministic operations: alpha composite, color-managed relight, transform, tracked overlay, camera move, CAD/CLO/Blender edit, or texture replacement.
3. Use a generative model only inside the smallest authorized region or for an unprotected background.
4. Require an explicit provider feature, not prompt wording. “Change only” is not a mask. “Keep identical” is not an identity lock. “3D” in a prompt is not a mesh.
5. Verify independently. Provider claims do not replace hash, pixel, frame, mesh, UV, or material checks.

## Capability classes

- `deterministic_composite`: eligible for exact protected-pixel work when inputs have real alpha and transforms are recorded.
- `explicit_mask_edit`: eligible for bounded edits only when the actual adapter exposes a mask and output passes outside-mask verification.
- `semantic_edit`: concept/candidate use; never exact-product by itself.
- `reference_generation`: references influence output but do not lock logos, wording, construction, or pixels.
- `image_to_video` and `video_edit`: generative through time; require temporal gates.
- `image_to_3d`: infers hidden geometry and texture; reconstruction candidate only.
- `deterministic_3d`: DCC/CAD/garment-authoring workflow where mesh, UVs, textures, measurements, and revisions are explicit.

## Unknown and external models

The agent covers major current families plus DevSkyy-local routes. It cannot truthfully enumerate every hosted fine-tune or future model. Unknown IDs fail closed. Add a registry entry only after checking first-party documentation, recording `verified_on`, and declaring inputs, outputs, masks, references, geometry, temporal behavior, and fidelity limitations.

## Adapter and API-surface truth beat provider truth

An API can support masks while a wrapper does not, and two APIs from the same provider can have different control and reproducibility properties. The Codex `image_gen.imagegen` wrapper available here exposes references and a prompt only, so it remains blocked for localized product patches.

For a one-shot OpenAI localized patch, use the Image API `/v1/images/edits` with the pinned `gpt-image-2-2026-04-21` snapshot. The contracted target and optimized mask must both be PNG and below 50MB; the same-size mask uses alpha zero in editable pixels, and its hash must be recorded. Bind `parameters.size` to the exact target geometry and satisfy the model's current size constraints. Omit `input_fidelity`, reject parameters outside the registry allowlist, and bind the exact prompt hash. Capture `x-request-id`, SDK/client version, exact request parameters, the ordered target-plus-product-authority input hashes, mask hash, and output hash; keep credentials out of every key or value. Use Responses image generation for conversational or multi-step workflows, not as a silent substitute for the directly model-addressed one-shot gate.

## Current DevSkyy routes

- GPT Image: backgrounds and candidates are allowed. One-shot local corrections require the direct Image API edit path, pinned model snapshot, API-ready alpha mask, generation receipt, and post-verification.
- Gemini/Nano Banana: useful for multi-reference scenes and prompt reasoning, but semantic references do not prove pixel locks.
- FLUX Kontext: semantic edit. Annotation boxes and reference strength are not explicit pixel locks.
- FLUX Fill and Stability Inpaint: possible bounded-edit engines only with an endpoint-specific same-size mask and zero outside-mask drift.
- Adobe Precise Composite: strong external candidate for protected product placement; still verify the downloaded object pixels and placement.
- TRELLIS, Meshy, Tripo, Hunyuan, SPAR3D, and TripoSR: reconstruction candidates governed by `3d-fidelity.md`.

See `sources.md` for the first-party documentation used to build the registry.
