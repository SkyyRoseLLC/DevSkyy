# Contract schema

The gate consumes JSON. Every path must resolve inside `--workspace`; absolute and parent-traversal paths are rejected.

## Localized product patch

```json
{
  "schema": "product-fidelity-edit.v1",
  "operation": "localized_product_patch",
  "target": {"path": "scene.png", "sha256": "64 lowercase hex characters"},
  "generator": {"route": "masked_edit", "supports_explicit_mask": true},
  "edit_region": {
    "type": "polygon",
    "coordinate_space": "pixels",
    "points": [[430, 440], [650, 440], [680, 700], [420, 700]],
    "feather_px": 2
  },
  "references": [
    {
      "sku": "br-007",
      "view": "side",
      "role": "physical_product_authority",
      "path": "founder-side.jpg",
      "sha256": "64 lowercase hex characters",
      "verbatim_text": ["Love Hurts"]
    }
  ],
  "verification": {
    "max_outside_changed_pixels": 0,
    "min_inside_changed_pixels": 1,
    "per_channel_tolerance": 0
  },
  "semantic_review": {"required": true, "scope": "logo_and_construction"}
}
```

Supported edit-region types are `polygon` and `bbox`. A `bbox` uses `[left, top, right, bottom]`. The internal mask uses white for editable pixels and black for locked pixels. Provider-specific mask polarity must be converted explicitly by the caller and recorded with a derived-mask SHA-256 receipt. A hash-bound semantic review of exact lettering/logo/construction is required; outside-mask equality alone cannot prove that the generated pixels are correct.

## Protected scene composite

```json
{
  "schema": "product-fidelity-edit.v1",
  "operation": "protected_scene_composite",
  "target": {"path": "current-scene.png", "sha256": "64 lowercase hex characters"},
  "generator": {"route": "background_only_then_composite", "supports_explicit_mask": false},
  "pose_change": false,
  "references": [
    {
      "sku": "sg-009",
      "view": "on_model_front",
      "role": "protected_product_layer",
      "path": "sg-009-protected.png",
      "sha256": "64 lowercase hex characters",
      "approved_pose_source": true
    }
  ],
  "verification": {"protected_rgb_must_match": true}
}
```

Protected composites are layout/proof artifacts unless they also carry the native optical contract and independent review required below. Exact opaque RGB preservation does not by itself authorize final commerce use.

## Native collection scene

```json
{
  "schema": "product-fidelity-edit.v1",
  "operation": "native_collection_scene",
  "target": {"path": "approved-on-model.png", "sha256": "64 lowercase hex characters"},
  "generator": {
    "route": "environment_rebuild_around_source",
    "supports_reference_images": true,
    "product_redraw_allowed": false
  },
  "pose_change": false,
  "collection_scene": {
    "collection_id": "love-hurts",
    "scene_id": "LH-COMMERCE-1",
    "story_role": "enchanted rose vow aisle",
    "scene_variation": "cathedral-aisle",
    "products": [
      {"sku": "lh-001", "cta": "Shop the Bomber"}
    ]
  },
  "references": [
    {
      "sku": "lh-001",
      "view": "on_model_front",
      "role": "physical_product_authority",
      "path": "approved-on-model.png",
      "sha256": "64 lowercase hex characters",
      "authority_state": "founder_approved",
      "approved_pose_source": true,
      "verbatim_text": ["Love Hurts"]
    },
    {
      "sku": "lh-001",
      "view": "on_model_front",
      "role": "protected_garment_matte",
      "path": "lh-001-garment-matte.png",
      "sha256": "64 lowercase hex characters",
      "authority_state": "founder_approved",
      "approved_pose_source": true
    },
    {
      "sku": "scene-love-hurts",
      "view": "scene",
      "role": "scene_authority",
      "path": "approved-collection-hero.png",
      "sha256": "64 lowercase hex characters",
      "authority_state": "founder_approved"
    }
  ],
  "protected_placements": [
    {"path": "lh-001-garment-matte.png", "x": 220, "y": 70}
  ],
  "optical_contract": {
    "output_dimensions": [1536, 864],
    "subject_bbox": [220, 70, 760, 840],
    "horizon_y": 430,
    "foot_contact_points": [[390, 820], [610, 820]],
    "lens_class": "normal",
    "camera_height": "eye_level",
    "key_light": {"direction": "camera_left", "temperature_k": 3200, "softness": "soft"},
    "shadow": {"direction": "camera_right", "softness": "soft"},
    "floor_response": "polished",
    "reflection_mode": "subtle",
    "contact_shadow_required": true,
    "edge_spill_required": true,
    "depth_occlusion_required": true,
    "atmospheric_depth_required": true,
    "scene_grain_match_required": true
  },
  "review_gate": {
    "candidate_author": "builder-id",
    "independent_reviewer_required": true,
    "minimum_scores": {
      "product_fidelity": 95,
      "optical_integration": 90,
      "anatomy_pose": 90,
      "collection_story": 90,
      "commerce_readiness": 90
    }
  },
  "promotion": {
    "founder_approval_required": true,
    "wiring_allowed": false,
    "deployment_allowed": false
  }
}
```

Allowed generator routes are `environment_rebuild_around_source`, `protected_garment_reconstruction`, and `full_native_redraw_candidate`. The redraw route requires `product_redraw_allowed: true`, is candidate-only, and cannot claim exact product preservation.

The independent review receipt passed to `verify --review` is:

```json
{
  "schema": "product-fidelity-native-review.v1",
  "candidate_sha256": "64 lowercase hex characters",
  "reviewer": "reviewer-id",
  "scores": {
    "product_fidelity": 97,
    "optical_integration": 94,
    "anatomy_pose": 95,
    "collection_story": 96,
    "commerce_readiness": 94
  },
  "hard_fails": [],
  "founder_status": "PENDING"
}
```

`reviewer` must differ from `candidate_author`. `founder_status: APPROVED` is still not deployment authorization; the theme workflow must separately update the wiring/promotion manifest.

## Reference roles

- `physical_product_authority`
- `approved_techflat`
- `exact_logo_or_patch_authority`
- `protected_product_layer`
- `protected_garment_matte`
- `scene_authority`
- `context_only`

Every reference requires an explicit view. Recommended views include `front`, `side`, `back`, `on_model_front`, `on_model_side`, `on_model_back`, and `scene`.
