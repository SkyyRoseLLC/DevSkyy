# Contract schema

The gate consumes JSON. Every path must resolve inside `--workspace`; absolute paths are allowed only when they remain inside that workspace, and parent traversal cannot escape it.

## Localized product patch

```json
{
  "schema": "product-fidelity-edit.v2",
  "operation": "localized_product_patch",
  "target": {"path": "scene.png", "sha256": "64 lowercase hex characters"},
  "generator": {
    "route": "masked_edit",
    "supports_explicit_mask": true,
    "model_id": "openai-responses-image-mask"
  },
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
      "verbatim_text": ["Love Hurts"],
      "authority_receipt": {
        "path": "br-007-side-authority.json",
        "sha256": "64 lowercase hex characters"
      }
    }
  ],
  "verification": {
    "max_outside_changed_pixels": 0,
    "min_inside_changed_pixels": 1,
    "per_channel_tolerance": 0
  },
  "semantic_review": {
    "required": true,
    "scope": "logo_and_construction",
    "candidate_author": "builder-id",
    "required_checks": [
      "color_and_material",
      "construction",
      "placement",
      "verbatim_text"
    ]
  }
}
```

Supported edit-region types are `polygon` and `bbox`. A `bbox` uses half-open `[left, top, right, bottom]` coordinates: right and bottom are locked. `feather_px` must be a non-negative integer no larger than one quarter of the shorter image dimension. The internal mask uses white for editable pixels and black for locked pixels. Provider-specific mask polarity must be converted explicitly by the caller and recorded with a derived-mask SHA-256 receipt. A hash-bound independent semantic review of exact lettering/logo/construction is required; outside-mask equality alone cannot prove that the generated pixels are correct.

Pass the localized review receipt to verification:

```bash
python3 scripts/fidelity_gate.py verify \
  --contract /absolute/path/contract.json \
  --workspace /absolute/path/workspace \
  --output /absolute/path/candidate.png \
  --review /absolute/path/semantic-review.json
```

The semantic review receipt is bound to both the canonical JSON contract and candidate bytes:

```json
{
  "schema": "product-fidelity-semantic-review.v2",
  "contract_sha256": "SHA-256 of canonical sorted compact JSON contract",
  "candidate_sha256": "SHA-256 of candidate bytes",
  "reviewer": "independent-reviewer-id",
  "scope": "logo_and_construction",
  "verdict": "PASS",
  "findings": [],
  "checks": {
    "color_and_material": "PASS",
    "construction": "PASS",
    "placement": "PASS",
    "verbatim_text": "PASS"
  },
  "expected_verbatim_text": ["Love Hurts"]
}
```

The reviewer identity is normalized for case and surrounding whitespace and must differ from `semantic_review.candidate_author`. The review must contain exactly the contracted check set, every check must be `PASS`, and `expected_verbatim_text` must exactly match the reference declarations. A missing, stale, self-authored, wrong-scope, non-passing, extra-check, missing-check, text-mismatched, or findings-bearing review blocks verification. Reviewer names are self-declared unless a higher-level signed attestation system verifies them, so a passing receipt still leaves wiring and deployment disabled.

## Source-authority receipt

Every product-bearing reference role requires an `authority_receipt` hash binding. This prevents a contract from treating a bare source hash as canonical SKU/view evidence: it must also bind the current catalog, imagery SOT, dossier, and resolver version.

```json
{
  "schema": "product-fidelity-source-authority.v1",
  "status": "VERIFIED",
  "sku": "br-007",
  "view": "side",
  "role": "physical_product_authority",
  "source_sha256": "SHA-256 of the exact reference bytes",
  "catalog": {
    "path": "skyyrose-catalog.csv",
    "sha256": "SHA-256 of the current canonical catalog"
  },
  "sot_manifest": {
    "path": "data/sot-images.json",
    "sha256": "SHA-256 of the current imagery SOT"
  },
  "dossier": {
    "path": "dossiers/black-rose-shorts.md",
    "sha256": "SHA-256 of the governing dossier"
  },
  "resolver": {
    "id": "devskyy-product-authority-resolver",
    "version": "1.0.0"
  }
}
```

All receipt and authority paths must remain inside the declared workspace. CLI `--receipt` destinations must also remain inside the workspace, have an existing parent directory, and be new paths: the gate uses exclusive creation and refuses to overwrite any existing file or collide with the contract, target, reference, candidate, or review inputs. The gate reads each source file once, hashes the exact bytes it judges, and blocks missing, stale, malformed, escaped, or mismatched bindings. The source-authority receipt is a local hash-bound provenance assertion, not a cryptographic identity signature; preflight therefore remains `generation_authorized: false` until the caller separately proves live model availability and obtains the required human authorization.

## Protected scene composite

```json
{
  "schema": "product-fidelity-edit.v2",
  "operation": "protected_scene_composite",
  "target": {"path": "current-scene.png", "sha256": "64 lowercase hex characters"},
  "generator": {
    "route": "background_only_then_composite",
    "supports_explicit_mask": false,
    "model_id": "adobe-precise-composite"
  },
  "pose_change": false,
  "references": [
    {
      "sku": "sg-009",
      "view": "on_model_front",
      "role": "protected_product_layer",
      "path": "sg-009-protected.png",
      "sha256": "64 lowercase hex characters",
      "approved_pose_source": true,
      "verbatim_text": [],
      "authority_receipt": {
        "path": "sg-009-front-authority.json",
        "sha256": "64 lowercase hex characters"
      }
    }
  ],
  "protected_placements": [
    {"path": "sg-009-protected.png", "x": 220, "y": 70}
  ],
  "verification": {"protected_rgb_must_match": true}
}
```

Every placement must resolve to exactly one hash-verified protected reference, use non-negative integer coordinates, and remain inside the target geometry. Protected composites must preserve target dimensions and are layout/proof artifacts unless they also carry the native optical contract and independent review required below. Exact opaque RGB preservation does not by itself authorize final commerce use.

## Native collection scene

```json
{
  "schema": "product-fidelity-edit.v2",
  "operation": "native_collection_scene",
  "target": {"path": "approved-on-model.png", "sha256": "64 lowercase hex characters"},
  "generator": {
    "route": "environment_rebuild_around_source",
    "supports_reference_images": true,
    "product_redraw_allowed": false,
    "model_id": "openai-responses-image-mask"
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
      "verbatim_text": ["Love Hurts"],
      "authority_receipt": {
        "path": "lh-001-front-authority.json",
        "sha256": "64 lowercase hex characters"
      }
    },
    {
      "sku": "lh-001",
      "view": "on_model_front",
      "role": "protected_garment_matte",
      "path": "lh-001-garment-matte.png",
      "sha256": "64 lowercase hex characters",
      "authority_state": "founder_approved",
      "approved_pose_source": true,
      "verbatim_text": ["Love Hurts"],
      "authority_receipt": {
        "path": "lh-001-matte-authority.json",
        "sha256": "64 lowercase hex characters"
      }
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
  "schema": "product-fidelity-native-review.v2",
  "contract_sha256": "SHA-256 of canonical sorted compact JSON contract",
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

`reviewer` must differ from `candidate_author`. The review identity and `founder_status` are self-declared by this receipt. `founder_status: APPROVED` therefore produces `FOUNDER_APPROVAL_ATTESTATION_REQUIRED`, not founder-approved or deployment-authorized state; the theme workflow must verify a separate founder attestation before updating wiring or promotion manifests.

## Version and policy boundaries

`product-fidelity-edit.v1` is intentionally blocked with `SCHEMA_MIGRATION_REQUIRED`; it must not silently inherit v2 guarantees. V2 requires a registry-backed `generator.model_id`, current model capability evidence, product source-authority receipts, explicit `verbatim_text` declarations, zero outside-mask change, zero per-channel tolerance, strict protected placement, byte-snapshot hashing, and the v2 semantic/native review shapes. A model registry match is a policy floor, not proof of current API availability, so preflight never authorizes a paid generation call by itself.

## Reference roles

- `physical_product_authority`
- `approved_techflat`
- `exact_logo_or_patch_authority`
- `protected_product_layer`
- `protected_garment_matte`
- `scene_authority`
- `context_only`

Every reference requires an explicit view. Recommended views include `front`, `side`, `back`, `on_model_front`, `on_model_side`, `on_model_back`, and `scene`.
