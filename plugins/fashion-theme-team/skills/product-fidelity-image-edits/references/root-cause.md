# Root cause and prevention

## Repeated failure pattern

A prompt says “change only the product,” but a full-frame generative editor receives a complete scene, small product photos with different geometry, model or proof-board references, and text, logo, pose, lighting, and environment requirements at once.

Without an explicit mask, the model performs semantic reconstruction of the complete frame. Prompt language is not a pixel lock. The result can look coherent while changing the face, scene, aspect ratio, product wording, embroidery, or non-target pixels.

## Why reference optimization alone is insufficient

Canonical names, clear crops, higher-resolution presentation, view labels, and isolated detail crops reduce ambiguity. They do not make a maskless editor deterministic. Localized exact-product correction still requires a mask and an outside-mask diff gate.

## Hardened prevention

- `localized_product_patch` requires an explicit mask and a mask-capable route.
- `protected_scene_composite` generates the background separately and composites a real-alpha product/model layer afterward.
- `native_collection_scene` begins from the approved on-model source and designs camera, floor, lighting, occlusion, and environment around it; it requires an independent optical-integration verdict in addition to product truth.
- Pose changes require an approved source in the requested pose; otherwise exact fidelity is unprovable.
- Source, mask, target, prompt, and output are hash-bound.
- Geometry must remain identical for localized patches.
- Any outside-mask change is a hard failure by default.
- Rejected candidates are evidence only and cannot become edit sources.
- Protected-pixel equality and native photographic integration are separate gates. Neither can substitute for the other.

## Why the protected-composite method can still fail

A full-model alpha may preserve every opaque RGB pixel and still look pasted into an unrelated plate. The failure is structural: the source and plate disagree about lens, camera height, horizon, body scale, floor plane, contact points, key/fill direction, color temperature, shadow azimuth, reflection behavior, atmospheric depth, occlusion, and edge spill.

For final commerce imagery, build outward from the approved source camera. Preserve a garment matte separately from the person when possible. Require eyes-on comparison at full size. If the feet float, the shadow belongs to another light, the edge color does not receive the scene, or the body has no physical relationship to nearby objects, the candidate is blocked even when product pixels are exact.

## Common traps

- A `.png` with a checkerboard baked into RGB is not transparent.
- A file named `back` may visually show a side; visual classification and canonical metadata control.
- A multi-SKU contact sheet is useful for review but weak as the sole exact-product reference.
- Upscaling cannot create embroidery or lettering detail that the source does not contain.
- A natural-pose request conflicts with pixel-exact garment preservation unless that pose already exists in an approved product source.
- A perfect alpha extraction does not establish a shared camera or shared light field.
- A generated plate selected only for beauty will usually fail a source-camera match.
