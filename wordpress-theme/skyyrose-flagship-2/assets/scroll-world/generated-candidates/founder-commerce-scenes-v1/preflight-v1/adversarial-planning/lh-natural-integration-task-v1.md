# Adversarial planning task: natural model integration for SkyyRose V2 Scroll World

Plan a reusable prompt and compositing architecture for the V2 Scroll World scenes in:

- `/Users/theceo/DevSkyy-product-card-approved/wordpress-theme/skyyrose-flagship-2`

The immediate proof scene is `LH-COMMERCE-1` (Love Hurts, The Vow Aisle). The founder-approved inputs are:

- Cathedral plate: `assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/love-hurts/lh-commerce-1-bomber-cathedral-plate-v1.png`
- Approved two-model layer: `assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/limited-pro-v1/lh-commerce-1-pro-session-logo-patched-v2.png`
- Current draft prompt: `assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/preflight-v1/vision-authored-prompts/lh-commerce-1-natural-integration-v1.json`
- Current draft compositor: `scripts/build-natural-scene-integration.py`
- Existing runtime wiring: `data/scene-narrative-blueprints.json`, `template-collection.php`, and `assets/css/theme.css`

The current CSS overlay looks patched in. The founder wants the models to look as if the complete scene was generated or photographed with them naturally present—not like transparent cutouts placed afterward.

The plan must solve both sides of the problem:

1. Natural scene synthesis: camera/perspective, scale, floor plane, pose relationship, occlusion, light direction and softness, contact shadows, material-specific light response, reflected color, floor reflections, depth/atmosphere, grain, edge integration, and collection narrative must be authored as one coherent scene.
2. Product fidelity: exact garment construction, embroidery, tackle twill, silicone/embossed marks, sublimation, colors, logos, product identities, and founder-approved model identity may not drift. A generated result that looks natural but invents the products is a failure.

Important constraints:

- Image-generation prompts are stored in JSON or HTML before generation.
- Every input file and SOT binding must be preflighted and hash-bound; stale or untracked references fail closed.
- The generator may be used for scene interaction, but the final workflow must explain how protected product truth survives without creating a sticker/cutout look.
- Do not assume a simple "composite exact pixels last" step is sufficient; that is the failure mode under review. If the plan retains protected pixels, specify how local relighting, occlusion, edge transfer, shadow casting, reflections, and shared image texture can be added without uncontrolled product repainting.
- The enchanted rose must remain an equal focal point in Love Hurts scene 1.
- UI/copy/product-link safe zones must be planned at desktop, tablet, and mobile before generation.
- Batch generation only; founder reviews boards, not one image at a time.
- No V2 wiring until the candidate passes mechanical checks, visual adversarial review, and founder approval.
- No deployment or production write.

Required deliverable:

- A concrete plan with real steps, exact files to add/change, and one falsifiable verification for every step.
- A reusable scene prompt schema that separates immutable product facts, scene-native integration instructions, camera/floor/light/occlusion contracts, forbidden drift, and responsive safe zones.
- A two-pass or multi-pass image strategy that can make models feel native while preserving product truth.
- Automatic rejection rules for "pasted cutout," product drift, blocked hero motif, floating feet, impossible shadows/reflections, inconsistent light, and UI collisions.
- A bounded rollout order: prove Love Hurts scene 1 first, then extend only to scenes whose model/product layers are founder-approved.
- Explicit risks and open questions.

Do not edit files. Draft the initial plan only.
