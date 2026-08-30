#!/usr/bin/env python3
"""Validate the fail-closed V2 native Scroll World production contract.

This validates an executable source stage with blocked scene/promotion gates.
It rejects any route that can silently use the historical environment-only
pack, a random collection product, an undeclared role, or an unapproved
candidate asset.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "docs/design/v2-remodel/native-scene-regeneration-v2/scroll-world-production/contract.json"
DEFAULT_EVIDENCE = ROOT / "docs/design/v2-remodel/native-scene-regeneration-v2/scroll-world-production/evidence.json"
DEFAULT_MANIFEST = ROOT / "wordpress-theme/skyyrose-flagship-2/data/scroll-world-production-manifest.json"
DEFAULT_SCENE_REFERENCE_INDEX = ROOT / "data/scroll-world-scene-reference-index.json"
EXPECTED = {
    "black-rose": ("BR-COMMERCE-1", "BR-COMMERCE-2", "BR-COMMERCE-3"),
    "love-hurts": ("LH-COMMERCE-1", "LH-COMMERCE-2", "LH-COMMERCE-3"),
    "signature": ("SIG-COMMERCE-1", "SIG-COMMERCE-2", "SIG-COMMERCE-3"),
}
ROLES = ("hero_world_moment_1", "hero_world_moment_2", "collection_preorder")
HERO_ROLES = ("hero_world_moment_1", "hero_world_moment_2")
REQUIRED_SCENE_WEARERS = ("man", "woman")
DECLARED_SCENE_CTA = {
    "black-rose": {
        "BR-COMMERCE-1": {"role": "hero_world_moment_1", "cta": {"label": "Complete the Set", "href": "#shop"}, "product_skus": {"br-001", "br-002"}},
        "BR-COMMERCE-2": {"role": "hero_world_moment_2", "cta": {"label": "Complete the Look", "href": "#shop"}, "product_skus": {"br-005", "br-007", "br-004"}},
        "BR-COMMERCE-3": {"role": "collection_preorder", "cta": {"label": "Pre-Order the Jersey Series", "href": "#jersey-series"}, "product_skus": {"br-003", "br-008", "br-009", "br-010", "br-011", "br-012", "br-014", "br-015"}},
    },
    "love-hurts": {
        "LH-COMMERCE-1": {"role": "hero_world_moment_1", "cta": {"label": "Complete the Look", "href": "#shop"}, "product_skus": {"lh-004", "lh-002", "lh-006"}},
        "LH-COMMERCE-2": {"role": "hero_world_moment_2", "cta": {"label": "Shop the Shorts", "href": "#shop"}, "product_skus": {"lh-003"}},
        "LH-COMMERCE-3": {"role": "collection_preorder", "cta": {"label": "Pre-Order The Fannie", "href": "/pre-order/"}, "product_skus": {"lh-005"}},
    },
    "signature": {
        "SIG-COMMERCE-1": {"role": "hero_world_moment_1", "cta": {"label": "Complete the Look", "href": "#shop"}, "product_skus": {"sg-009", "sg-007"}},
        "SIG-COMMERCE-2": {"role": "hero_world_moment_2", "cta": {"label": "Complete the Look", "href": "#shop"}, "product_skus": {"sg-013", "sg-014", "sg-006"}},
        "SIG-COMMERCE-3": {"role": "collection_preorder", "cta": {"label": "Explore Pre-Orders", "href": "/pre-order/"}, "product_skus": {"sg-005", "sg-001", "sg-002", "sg-003", "sg-015"}},
    },
}
CTA_ANCHOR_KINDS = ("font_statue", "scene_backdrop", "hero_graphic", "scene_backdrop_and_gold_monuments")
CTA_HERO_ANCHORS = {
    "black-rose": {
        "BR-COMMERCE-1": {
            "id": "BR_FONT_STATUE",
            "kind": "font_statue",
            "path": "wordpress-theme/skyyrose-flagship-2/assets/sot/images/lockups/hero-derived/black-rose-font-statue-hero-exact-v1.png",
            "sha256": "a879be465973a5edf2aa8fe6ae2e0eb34d66e11cf3e5081b3f4beea4d967aa3c",
        },
        "BR-COMMERCE-2": {
            "id": "BR_STAR_GRAPHIC",
            "kind": "hero_graphic",
            "path": "wordpress-theme/skyyrose-flagship-2/assets/sot/images/lockups/black-rose-star-graphic.png",
            "sha256": "6af686d798fb6541c3a691492809224a6318d62573d1462c4f3f99af9957ec78",
        },
        "BR-COMMERCE-3": {
            "id": "BR_BAY_BRIDGE_NIGHT_BACKDROP",
            "kind": "scene_backdrop",
            "path": "wordpress-theme/skyyrose-flagship-2/assets/sot/images/preorder/black-rose-salon.webp",
            "sha256": "2ca5326cb30844311f0dd593c7b6141ab7819ba0926913183e65c9bd6207a3eb",
        },
    },
    "signature": {
        "SIG-COMMERCE-1": {
            "id": "SIG_FONT_STATUE",
            "kind": "font_statue",
            "path": "wordpress-theme/skyyrose-flagship-2/assets/sot/images/lockups/founder-supplied/derived/signature-skyyrose-font-sculpture-protected-v1.png",
            "sha256": "e259c246a320b8220044da915c3b6fa89ecbbe7ab4a74eb80bafebe668d0e274",
        },
        "SIG-COMMERCE-2": {
            "id": "SIG_GOLDEN_GATE_MONUMENTS_HERO",
            "kind": "scene_backdrop_and_gold_monuments",
            "path": "wordpress-theme/skyyrose-flagship-2/assets/sot/images/hero/signature-golden-gate-monuments-v2.webp",
            "sha256": "f8bb7d2a1572f575e4f9e4e45edebf6115b4339708e64f95c3bb251813fc0fd8",
        },
        "SIG-COMMERCE-3": {
            "id": "SIG_SR_ROSE_GRAPHIC",
            "kind": "hero_graphic",
            "path": "wordpress-theme/skyyrose-flagship-2/assets/sot/images/lockups/founder-supplied/signature-sr-rose-graphic-founder-supplied-v1.png",
            "sha256": "2667368884d96e5d0689fdb5f6795940b0ea9aa79035d4542859c235b7864bcc",
        },
    },
}
LOVE_HURTS_ROSE_ANCHOR = {
    "id": "LH_ENCHANTED_ROSE_CATHEDRAL",
    "kind": "protected_rose_world",
    "path": "skyyrose/assets/scenes/love-hurts/love-hurts-enchanted-rose-cathedral/love-hurts-enchanted-rose-cathedral-v2.png",
    "sha256": "4f3c937f958f9ee4f6e87ec2696a70ea67903ffe24b52b41f699bb50680c7b02",
}
LOVE_HURTS_ROSE_ASSIGNMENTS = {
    "LH-COMMERCE-1": "ceremonial_vow_aisle_approach_to_the_protected_rose",
    "LH-COMMERCE-2": "fractured_chamber_with_protected_rose_sharp_opposite_the_product",
    "LH-COMMERCE-3": "intimate_vitrine_with_the_fannie_foreground_and_protected_rose_legible",
}
VOW_AISLE_CAST = ["lh-004", "lh-002", "lh-006"]
VOW_AISLE_WEARER_ASSIGNMENTS = [
    {"wearer": "man", "look_id": "lh-bomber-black", "skus": ["lh-004", "lh-002"]},
    {"wearer": "woman", "look_id": "lh-bomber-white", "skus": ["lh-004", "lh-006"]},
]
VOW_AISLE_CANDIDATE = {
    "path": "renders/scroll-world/LH-COMMERCE-1/preserved-candidate/LH-COMMERCE-1-vow-aisle-founder-approved-v1.png",
    "sha256": "1eb217504536634047eb3a682915a4d8ea0218b9968c9d3ff96addb231578f38",
    "receipt": "data/candidates/lh-commerce-1-vow-aisle-preserved-retrieval-receipt-v1.json",
}
ALLOWED_SCENE_STATUSES = {
    "BLOCKED_SOURCE_GATE",
    "SOURCE_AUTHORITY_CANDIDATES_REJECTED__BLOCKED",
    "PRESERVED_NOT_LOCALLY_VERIFIABLE",
    "PRESERVED_LOCAL_HASH_VERIFIED_QUARANTINED",
    "FOUNDER_APPROVED_VISUAL__RUNTIME_PROMOTION_PENDING",
    "FOUNDER_APPROVED_PRESERVE_DO_NOT_REGENERATE__RUNTIME_PROMOTION_PENDING",
}


def load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected object: {path}")
    return value


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def scene_contract_path(value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    candidate = (ROOT / value).resolve()
    try:
        candidate.relative_to(ROOT.resolve())
    except ValueError:
        return None
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--evidence", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--scene-reference-index", type=Path, default=DEFAULT_SCENE_REFERENCE_INDEX)
    parser.add_argument(
        "--authorize-paid-scene",
        choices=tuple(scene_id for scene_ids in EXPECTED.values() for scene_id in scene_ids),
        help=(
            "Fail closed before a paid native scene request. Authorization is only emitted when "
            "the declared scene is source-ready; it is not a provider-generation command."
        ),
    )
    args = parser.parse_args()
    errors: list[str] = []
    blockers: list[str] = []

    try:
        contract = load(args.contract)
        evidence = load(args.evidence)
        manifest = load(args.manifest)
        scene_reference_index = load(args.scene_reference_index)
    except ValueError as error:
        print(f"FAIL {error}")
        return 2

    actual_sot_hash = sha256(ROOT / "data/product-sot.json")
    for label, record in (("contract", contract), ("manifest", manifest)):
        lock = record.get("source_freshness_lock")
        if not isinstance(lock, dict) or lock.get("sha256") != actual_sot_hash:
            errors.append(f"{label}: PRODUCT_SOT_HASH_MISMATCH")
    if evidence.get("source_freshness_lock") != actual_sot_hash:
        errors.append("evidence: PRODUCT_SOT_HASH_MISMATCH")
    index_lock = scene_reference_index.get("source_freshness_lock")
    if not isinstance(index_lock, dict) or index_lock.get("sha256") != actual_sot_hash:
        errors.append("scene-reference-index: PRODUCT_SOT_HASH_MISMATCH")
    actual_index_hash = sha256(args.scene_reference_index)
    for label, record in (("contract", contract), ("manifest", manifest)):
        index_record = record.get("scene_reference_index")
        if not isinstance(index_record, dict):
            errors.append(f"{label}: SCENE_REFERENCE_INDEX_MISSING")
            continue
        if index_record.get("path") != "data/scroll-world-scene-reference-index.json":
            errors.append(f"{label}: SCENE_REFERENCE_INDEX_PATH_INVALID")
        if index_record.get("sha256") != actual_index_hash:
            errors.append(f"{label}: SCENE_REFERENCE_INDEX_HASH_MISMATCH")
    if contract.get("historical_five_world_pack", {}).get("status") != "ARCHIVAL_CONTEXT_ONLY":
        errors.append("contract: HISTORICAL_PACK_NOT_ARCHIVAL")
    if manifest.get("historical_five_world_pack") != "ARCHIVAL_CONTEXT_ONLY_NOT_RUNTIME_FALLBACK":
        errors.append("manifest: HISTORICAL_PACK_CAN_BE_RUNTIME_FALLBACK")
    if contract.get("status") not in {
        "BLOCKED_SOURCE_AND_PROMOTION_GATES",
        "SOURCE_STAGE_AUTHORIZED_IN_PROGRESS__SCENE_AND_PROMOTION_GATES_BLOCKED",
    }:
        errors.append("contract: BLOCKED_STATE_REQUIRED")
    dual_cast = contract.get("v2_dual_cast_on_model_authority")
    if not isinstance(dual_cast, dict):
        errors.append("contract: DUAL_CAST_ON_MODEL_AUTHORITY_POLICY_MISSING")
    else:
        if dual_cast.get("status") != "FOUNDER_DIRECTIVE_BINDING":
            errors.append("contract: DUAL_CAST_ON_MODEL_AUTHORITY_POLICY_STATUS_INVALID")
        coverage = str(dual_cast.get("coverage_requirement", "")).lower()
        if "women-led" not in coverage or "men-led" not in coverage or "every v2 sku" not in coverage:
            errors.append("contract: DUAL_CAST_ON_MODEL_AUTHORITY_COVERAGE_INVALID")
        completion = str(dual_cast.get("completion_rule", "")).lower()
        if "candidate" not in completion or "rejected" not in completion or "two distinct approved" not in completion:
            errors.append("contract: DUAL_CAST_ON_MODEL_AUTHORITY_COMPLETION_RULE_INVALID")
    hero_dual_cast = contract.get("hero_dual_cast_four_sku_rule")
    if not isinstance(hero_dual_cast, dict):
        errors.append("contract: HERO_DUAL_CAST_FOUR_SKU_POLICY_MISSING")
    else:
        if hero_dual_cast.get("status") != "FOUNDER_DIRECTIVE_BINDING":
            errors.append("contract: HERO_DUAL_CAST_FOUR_SKU_POLICY_STATUS_INVALID")
        if tuple(hero_dual_cast.get("required_roles", [])) != HERO_ROLES:
            errors.append("contract: HERO_DUAL_CAST_FOUR_SKU_ROLES_INVALID")
        if tuple(hero_dual_cast.get("required_scene_wearers", [])) != REQUIRED_SCENE_WEARERS:
            errors.append("contract: HERO_DUAL_CAST_FOUR_SKU_WEARERS_INVALID")
        if hero_dual_cast.get("minimum_distinct_exact_skus_across_two_hero_moments") != 4:
            errors.append("contract: HERO_DUAL_CAST_FOUR_SKU_MINIMUM_INVALID")
        scope = str(hero_dual_cast.get("scope", "")).lower()
        if "two hero-derived world moments together" not in scope:
            errors.append("contract: HERO_DUAL_CAST_FOUR_SKU_SCOPE_INVALID")
        allocation = str(hero_dual_cast.get("scene_composition", "")).lower()
        if "four distinct exact skus" not in allocation or "collection's approved scene edit" not in allocation:
            errors.append("contract: HERO_DUAL_CAST_FOUR_SKU_ALLOCATION_INVALID")
        if "does not count" not in str(hero_dual_cast.get("preorder_boundary", "")).lower():
            errors.append("contract: HERO_DUAL_CAST_FOUR_SKU_PREORDER_BOUNDARY_INVALID")
    manifest_dual_cast = manifest.get("hero_dual_cast_four_sku_rule")
    if not isinstance(manifest_dual_cast, dict):
        errors.append("manifest: HERO_DUAL_CAST_FOUR_SKU_POLICY_MISSING")
    else:
        if manifest_dual_cast.get("status") != "FOUNDER_DIRECTIVE_BINDING":
            errors.append("manifest: HERO_DUAL_CAST_FOUR_SKU_POLICY_STATUS_INVALID")
        if tuple(manifest_dual_cast.get("required_roles", [])) != HERO_ROLES:
            errors.append("manifest: HERO_DUAL_CAST_FOUR_SKU_ROLES_INVALID")
        if tuple(manifest_dual_cast.get("required_scene_wearers", [])) != REQUIRED_SCENE_WEARERS:
            errors.append("manifest: HERO_DUAL_CAST_FOUR_SKU_WEARERS_INVALID")
        if manifest_dual_cast.get("minimum_distinct_exact_skus_across_two_hero_moments") != 4:
            errors.append("manifest: HERO_DUAL_CAST_FOUR_SKU_MINIMUM_INVALID")
        if manifest_dual_cast.get("wiring_allowed_before_coverage") is not False:
            errors.append("manifest: HERO_DUAL_CAST_FOUR_SKU_PREMATURE_WIRING")
    index_dual_cast = scene_reference_index.get("hero_dual_cast_four_sku_rule")
    if not isinstance(index_dual_cast, dict):
        errors.append("scene-reference-index: HERO_DUAL_CAST_FOUR_SKU_POLICY_MISSING")
    else:
        if index_dual_cast.get("status") != "FOUNDER_DIRECTIVE_BINDING":
            errors.append("scene-reference-index: HERO_DUAL_CAST_FOUR_SKU_POLICY_STATUS_INVALID")
        if tuple(index_dual_cast.get("required_roles", [])) != HERO_ROLES:
            errors.append("scene-reference-index: HERO_DUAL_CAST_FOUR_SKU_ROLES_INVALID")
        if tuple(index_dual_cast.get("required_scene_wearers", [])) != REQUIRED_SCENE_WEARERS:
            errors.append("scene-reference-index: HERO_DUAL_CAST_FOUR_SKU_WEARERS_INVALID")
        if index_dual_cast.get("minimum_distinct_exact_skus_across_two_hero_moments") != 4:
            errors.append("scene-reference-index: HERO_DUAL_CAST_FOUR_SKU_MINIMUM_INVALID")
        scope = str(index_dual_cast.get("scope", "")).lower()
        if "two hero-derived world moments together" not in scope:
            errors.append("scene-reference-index: HERO_DUAL_CAST_FOUR_SKU_SCOPE_INVALID")
        allocation = str(index_dual_cast.get("sku_allocation", "")).lower()
        if "four distinct exact skus" not in allocation or "declared exact sku casts" not in allocation:
            errors.append("scene-reference-index: HERO_DUAL_CAST_FOUR_SKU_ALLOCATION_INVALID")
        if "not used" not in str(index_dual_cast.get("preorder_boundary", "")).lower():
            errors.append("scene-reference-index: HERO_DUAL_CAST_FOUR_SKU_PREORDER_BOUNDARY_INVALID")
    for label, record in (("contract", contract), ("manifest", manifest), ("scene-reference-index", scene_reference_index)):
        anchor_rule = record.get("hero_derived_cta_anchor_rule")
        if not isinstance(anchor_rule, dict):
            errors.append(f"{label}: HERO_DERIVED_CTA_ANCHOR_POLICY_MISSING")
            continue
        if anchor_rule.get("status") != "FOUNDER_DIRECTIVE_BINDING":
            errors.append(f"{label}: HERO_DERIVED_CTA_ANCHOR_POLICY_STATUS_INVALID")
        if tuple(anchor_rule.get("required_anchor_kinds", [])) != CTA_ANCHOR_KINDS:
            errors.append(f"{label}: HERO_DERIVED_CTA_ANCHOR_KINDS_INVALID")
        scope = str(anchor_rule.get("scope", "")).lower()
        if "all three declared cta scenes" not in scope or "black rose" not in scope or "signature" not in scope:
            errors.append(f"{label}: HERO_DERIVED_CTA_ANCHOR_SCOPE_INVALID")
    contract_anchor_rule = contract.get("hero_derived_cta_anchor_rule", {})
    manifest_anchor_rule = manifest.get("hero_derived_cta_anchor_rule", {})
    index_anchor_rule = scene_reference_index.get("hero_derived_cta_anchor_rule", {})
    for collection, assignments in CTA_HERO_ANCHORS.items():
        expected_assignment_ids = {scene_id: anchor["id"] for scene_id, anchor in assignments.items()}
        for label, rule in (("contract", contract_anchor_rule), ("manifest", manifest_anchor_rule)):
            if isinstance(rule, dict) and rule.get("assignments", {}).get(collection) != expected_assignment_ids:
                errors.append(f"{label}: {collection.upper().replace('-', '_')}_HERO_DERIVED_CTA_ASSIGNMENTS_INVALID")
        if isinstance(index_anchor_rule, dict) and tuple(index_anchor_rule.get("collection_coverage", {}).get(collection, [])) != tuple(assignments):
            errors.append(f"scene-reference-index: {collection.upper().replace('-', '_')}_HERO_DERIVED_CTA_COVERAGE_INVALID")
    for label, record in (("contract", contract), ("manifest", manifest), ("scene-reference-index", scene_reference_index)):
        rose_rule = record.get("love_hurts_enchanted_rose_story_rule")
        if not isinstance(rose_rule, dict):
            errors.append(f"{label}: LOVE_HURTS_ENCHANTED_ROSE_POLICY_MISSING")
            continue
        if rose_rule.get("status") != "FOUNDER_DIRECTIVE_BINDING":
            errors.append(f"{label}: LOVE_HURTS_ENCHANTED_ROSE_POLICY_STATUS_INVALID")
        scope = str(rose_rule.get("scope", "")).lower()
        if "all three declared love hurts cta scenes" not in scope:
            errors.append(f"{label}: LOVE_HURTS_ENCHANTED_ROSE_SCOPE_INVALID")
        if rose_rule.get("scene_assignments") != LOVE_HURTS_ROSE_ASSIGNMENTS:
            errors.append(f"{label}: LOVE_HURTS_ENCHANTED_ROSE_ASSIGNMENTS_INVALID")
        beast_perspective = str(rose_rule.get("beast_perspective", "")).lower()
        if "literal beast" not in beast_perspective or "perspective" not in beast_perspective:
            errors.append(f"{label}: LOVE_HURTS_BEAST_PERSPECTIVE_GUARD_MISSING")
        if label == "manifest":
            if rose_rule.get("protected_rose_anchor_id") != LOVE_HURTS_ROSE_ANCHOR["id"]:
                errors.append("manifest: LOVE_HURTS_ENCHANTED_ROSE_ANCHOR_ID_INVALID")
        else:
            anchor = rose_rule.get("protected_rose_anchor")
            if not isinstance(anchor, dict):
                errors.append(f"{label}: LOVE_HURTS_ENCHANTED_ROSE_ANCHOR_MISSING")
            elif any(anchor.get(key) != expected for key, expected in LOVE_HURTS_ROSE_ANCHOR.items()):
                errors.append(f"{label}: LOVE_HURTS_ENCHANTED_ROSE_ANCHOR_INVALID")
    rose_path = scene_contract_path(LOVE_HURTS_ROSE_ANCHOR["path"])
    if not rose_path or not rose_path.is_file():
        errors.append("love-hurts: PROTECTED_ENCHANTED_ROSE_SOURCE_MISSING")
    elif sha256(rose_path) != LOVE_HURTS_ROSE_ANCHOR["sha256"]:
        errors.append("love-hurts: PROTECTED_ENCHANTED_ROSE_SOURCE_HASH_MISMATCH")
    blockers.extend(str(item) for item in contract.get("current_blockers", []) if isinstance(item, str))
    promotion = manifest.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("wiring_allowed") is not False or promotion.get("deployment_allowed") is not False:
        errors.append("manifest: PROMOTION_BOUNDARY_INVALID")

    directions = contract.get("collection_galleries")
    galleries = manifest.get("collections")
    reference_collections = scene_reference_index.get("collection_scenes")
    if not isinstance(directions, dict) or not isinstance(galleries, dict):
        errors.append("collection galleries missing")
        directions, galleries = {}, {}
    if not isinstance(reference_collections, dict):
        errors.append("scene-reference-index: COLLECTION_SCENES_MISSING")
        reference_collections = {}
    for collection, expected_ids in EXPECTED.items():
        direction = directions.get(collection)
        gallery = galleries.get(collection)
        scene_references = reference_collections.get(collection)
        if not isinstance(direction, dict) or not isinstance(gallery, dict):
            errors.append(f"{collection}: GALLERY_MISSING")
            continue
        if not isinstance(scene_references, list) or len(scene_references) != 3:
            errors.append(f"{collection}: EXACTLY_THREE_SCENE_REFERENCES_REQUIRED")
            scene_references = []
        reference_by_id: dict[str, dict[str, Any]] = {}
        for reference in scene_references:
            if not isinstance(reference, dict):
                errors.append(f"{collection}: INVALID_SCENE_REFERENCE")
                continue
            reference_id = reference.get("scene_reference_id")
            if not isinstance(reference_id, str) or reference_id in reference_by_id:
                errors.append(f"{collection}: SCENE_REFERENCE_ID_INVALID")
                continue
            reference_by_id[reference_id] = reference
        if tuple(reference_by_id) != expected_ids:
            errors.append(f"{collection}: SCENE_REFERENCE_IDS_INVALID")
        if tuple(direction.get("scene_ids", [])) != expected_ids:
            errors.append(f"{collection}: CONTRACT_SCENE_IDS_INVALID")
        if tuple(direction.get("required_roles", [])) != ROLES or tuple(gallery.get("required_roles", [])) != ROLES:
            errors.append(f"{collection}: EXACT_THREE_ROLE_ORDER_REQUIRED")
        scenes = gallery.get("scenes")
        if not isinstance(scenes, list) or len(scenes) != 3:
            errors.append(f"{collection}: EXACTLY_THREE_SCENES_REQUIRED")
            continue
        if tuple(scene.get("id") for scene in scenes if isinstance(scene, dict)) != expected_ids:
            errors.append(f"{collection}: MANIFEST_SCENE_IDS_INVALID")
        if tuple(scene.get("role") for scene in scenes if isinstance(scene, dict)) != ROLES:
            errors.append(f"{collection}: MANIFEST_SCENE_ROLES_INVALID")
        declared_ctas = DECLARED_SCENE_CTA[collection]
        if tuple(declared_ctas) != expected_ids:
            errors.append(f"{collection}: DECLARED_SCENE_CTA_IDS_INVALID")
        scenes_by_id = {scene.get("id"): scene for scene in scenes if isinstance(scene, dict)}
        for scene_id, expected_cta in declared_ctas.items():
            scene = scenes_by_id.get(scene_id)
            if not isinstance(scene, dict):
                errors.append(f"{scene_id}: DECLARED_CTA_SCENE_MISSING")
                continue
            if scene.get("role") != expected_cta["role"]:
                errors.append(f"{scene_id}: DECLARED_CTA_ROLE_INVALID")
            if scene.get("cta") != expected_cta["cta"]:
                errors.append(f"{scene_id}: DECLARED_CTA_LABEL_OR_ROUTE_INVALID")
            if set(scene.get("product_skus", [])) != expected_cta["product_skus"]:
                errors.append(f"{scene_id}: DECLARED_CTA_CAST_INVALID")
        hero_scenes = [scene for scene in scenes if isinstance(scene, dict) and scene.get("role") in HERO_ROLES]
        if len(hero_scenes) != len(HERO_ROLES):
            errors.append(f"{collection}: HERO_DUAL_CAST_SCENE_COUNT_INVALID")
        hero_skus = {
            sku
            for scene in hero_scenes
            for sku in scene.get("product_skus", [])
            if isinstance(sku, str) and sku
        }
        if len(hero_skus) < 4:
            plan_status = (
                index_dual_cast.get("collection_plan_status", {}).get(collection)
                if isinstance(index_dual_cast, dict) and isinstance(index_dual_cast.get("collection_plan_status"), dict)
                else None
            )
            if plan_status != "BLOCKED_EXACT_FOUR_SKU_HERO_CAST_NOT_LOCALLY_VERIFIABLE":
                errors.append(f"{collection}: FOUR_DISTINCT_HERO_SKUS_REQUIRED")
            blockers.append(f"{collection.upper().replace('-', '_')}_FOUR_DISTINCT_HERO_SKUS_REQUIRED")
        if collection == "love-hurts" and hero_skus != {"lh-004", "lh-002", "lh-006", "lh-003"}:
            errors.append("love-hurts: FOUNDER_DECLARED_FOUR_SKU_HERO_CAST_INVALID")
        for scene in scenes:
            if not isinstance(scene, dict):
                errors.append(f"{collection}: INVALID_SCENE")
                continue
            if not isinstance(scene.get("product_skus"), list):
                errors.append(f"{scene.get('id')}: EXACT_CAST_REQUIRED")
            elif not scene["product_skus"]:
                blockers.append(f"{scene.get('id')}_EXACT_CAST_NOT_PRESENT")
            blockers.extend(str(item) for item in scene.get("missing_requirements", []) if isinstance(item, str))
            cta = scene.get("cta")
            if not isinstance(cta, dict) or not isinstance(cta.get("label"), str) or not isinstance(cta.get("href"), str):
                errors.append(f"{scene.get('id')}: EXACT_CTA_REQUIRED")
            scene_id = scene.get("id")
            scene_status = scene.get("status")
            scene_asset = scene.get("asset")
            if scene_status not in ALLOWED_SCENE_STATUSES:
                errors.append(f"{scene_id}: BLOCKED_PRESERVED_OR_FOUNDER_APPROVED_STATE_REQUIRED")
            if scene_asset is not None:
                if not isinstance(scene_status, str) or not scene_status.startswith("FOUNDER_APPROVED"):
                    errors.append(f"{scene_id}: CANDIDATE_ASSET_PRESENT_BEFORE_APPROVAL")
                if not isinstance(scene_asset, dict):
                    errors.append(f"{scene_id}: FOUNDER_APPROVED_ASSET_RECORD_INVALID")
                else:
                    asset_path = scene_contract_path(scene_asset.get("path"))
                    asset_hash = scene_asset.get("sha256")
                    approval_path = scene_contract_path(scene_asset.get("founder_approval_record"))
                    if scene_asset.get("runtime_wiring_allowed") is not False:
                        errors.append(f"{scene_id}: FOUNDER_APPROVED_ASSET_PREMATURE_WIRING")
                    if not asset_path or not asset_path.is_file():
                        errors.append(f"{scene_id}: FOUNDER_APPROVED_ASSET_MISSING")
                    elif not isinstance(asset_hash, str) or sha256(asset_path) != asset_hash:
                        errors.append(f"{scene_id}: FOUNDER_APPROVED_ASSET_HASH_MISMATCH")
                    if not approval_path or not approval_path.is_file():
                        errors.append(f"{scene_id}: FOUNDER_APPROVAL_RECORD_MISSING")
                    else:
                        try:
                            approval_record = load(approval_path)
                        except ValueError as error:
                            errors.append(str(error))
                        else:
                            approvals = approval_record.get("approvals")
                            approval = next(
                                (
                                    item
                                    for item in approvals
                                    if isinstance(item, dict)
                                    and item.get("scene_id") == scene_id
                                    and item.get("candidate_path") == scene_asset.get("path")
                                    and item.get("candidate_sha256") == asset_hash
                                    and item.get("decision") == "FOUNDER_APPROVED_VISUAL"
                                ),
                                None,
                            ) if isinstance(approvals, list) else None
                            if approval is None:
                                errors.append(f"{scene_id}: FOUNDER_APPROVAL_RECORD_BINDING_INVALID")
                            promotion_boundary = approval_record.get("promotion_boundary")
                            if not isinstance(promotion_boundary, dict) or any(
                                promotion_boundary.get(key) is not False
                                for key in ("runtime_wiring_allowed", "deployment_allowed")
                            ):
                                errors.append(f"{scene_id}: FOUNDER_APPROVAL_PROMOTION_BOUNDARY_INVALID")
            reference_id = scene.get("scene_reference_id")
            if reference_id != scene.get("id"):
                errors.append(f"{scene.get('id')}: EXACT_SCENE_REFERENCE_ID_REQUIRED")
            reference = reference_by_id.get(reference_id) if isinstance(reference_id, str) else None
            if not isinstance(reference, dict):
                errors.append(f"{scene.get('id')}: SCENE_REFERENCE_NOT_FOUND")
            else:
                if reference.get("role") != scene.get("role"):
                    errors.append(f"{scene.get('id')}: SCENE_REFERENCE_ROLE_MISMATCH")
                if reference.get("label") != scene.get("label"):
                    errors.append(f"{scene.get('id')}: SCENE_REFERENCE_LABEL_MISMATCH")
                if not isinstance(reference.get("journey_chapter"), str) or not reference["journey_chapter"]:
                    errors.append(f"{scene.get('id')}: SCENE_REFERENCE_JOURNEY_POSITION_MISSING")
                if reference.get("exact_skus") != scene.get("product_skus"):
                    errors.append(f"{scene.get('id')}: SCENE_REFERENCE_EXACT_CAST_MISMATCH")
                if reference.get("exact_cta") != scene.get("cta"):
                    errors.append(f"{scene.get('id')}: SCENE_REFERENCE_EXACT_CTA_MISMATCH")
                if reference.get("native_scene_contract") != scene.get("contract"):
                    errors.append(f"{scene.get('id')}: SCENE_REFERENCE_NATIVE_CONTRACT_MISMATCH")
                if reference.get("provider_generation_allowed") is not False:
                    errors.append(f"{scene.get('id')}: SCENE_REFERENCE_MUST_BE_BLOCKED")
                if collection == "love-hurts":
                    if reference.get("love_hurts_rose_anchor_id") != LOVE_HURTS_ROSE_ANCHOR["id"]:
                        errors.append(f"{scene.get('id')}: LOVE_HURTS_ENCHANTED_ROSE_REFERENCE_MISSING")
                    if reference.get("love_hurts_rose_assignment") != LOVE_HURTS_ROSE_ASSIGNMENTS[scene.get("id")]:
                        errors.append(f"{scene.get('id')}: LOVE_HURTS_ENCHANTED_ROSE_ASSIGNMENT_MISMATCH")
                    if scene.get("love_hurts_rose_anchor_id") != LOVE_HURTS_ROSE_ANCHOR["id"]:
                        errors.append(f"{scene.get('id')}: LOVE_HURTS_ENCHANTED_ROSE_MANIFEST_MISSING")
                    if scene.get("id") == "LH-COMMERCE-1":
                        if scene.get("product_skus") != VOW_AISLE_CAST:
                            errors.append("LH-COMMERCE-1: FOUNDER_DECLARED_MIXED_CAST_INVALID")
                        if scene.get("founder_declared_wearer_sku_assignments") != VOW_AISLE_WEARER_ASSIGNMENTS:
                            errors.append("LH-COMMERCE-1: FOUNDER_DECLARED_WEARER_ASSIGNMENTS_INVALID")
                        if reference.get("founder_declared_wearer_sku_assignments") != VOW_AISLE_WEARER_ASSIGNMENTS:
                            errors.append("LH-COMMERCE-1: ROOT_VOW_AISLE_WEARER_ASSIGNMENTS_INVALID")
                        if reference.get("reference_state") != "PRESERVED_LOCAL_HASH_VERIFIED_QUARANTINED":
                            errors.append("LH-COMMERCE-1: ROOT_PRESERVED_LOCAL_HASH_STATE_INVALID")
                        if scene.get("status") not in {
                            "PRESERVED_LOCAL_HASH_VERIFIED_QUARANTINED",
                            "FOUNDER_APPROVED_PRESERVE_DO_NOT_REGENERATE__RUNTIME_PROMOTION_PENDING",
                        }:
                            errors.append("LH-COMMERCE-1: MANIFEST_PRESERVED_LOCAL_HASH_STATE_INVALID")
                        for label, candidate in (("root", reference.get("local_preserved_candidate")), ("manifest", scene.get("local_preserved_candidate"))):
                            if not isinstance(candidate, dict):
                                errors.append(f"LH-COMMERCE-1: {label.upper()}_PRESERVED_CANDIDATE_MISSING")
                                continue
                            for key, expected in VOW_AISLE_CANDIDATE.items():
                                if candidate.get(key) != expected:
                                    errors.append(f"LH-COMMERCE-1: {label.upper()}_PRESERVED_CANDIDATE_{key.upper()}_INVALID")
                        candidate_path = scene_contract_path(VOW_AISLE_CANDIDATE["path"])
                        receipt_path = scene_contract_path(VOW_AISLE_CANDIDATE["receipt"])
                        if not candidate_path or not candidate_path.is_file():
                            errors.append("LH-COMMERCE-1: PRESERVED_CANDIDATE_FILE_MISSING")
                        elif sha256(candidate_path) != VOW_AISLE_CANDIDATE["sha256"]:
                            errors.append("LH-COMMERCE-1: PRESERVED_CANDIDATE_HASH_MISMATCH")
                        if not receipt_path or not receipt_path.is_file():
                            errors.append("LH-COMMERCE-1: PRESERVED_CANDIDATE_RECEIPT_MISSING")
                        else:
                            try:
                                receipt = load(receipt_path)
                            except ValueError as error:
                                errors.append(str(error))
                            else:
                                receipt_candidate = receipt.get("candidate")
                                if not isinstance(receipt_candidate, dict) or any(
                                    receipt_candidate.get(key) != expected
                                    for key, expected in VOW_AISLE_CANDIDATE.items()
                                    if key != "receipt"
                                ):
                                    errors.append("LH-COMMERCE-1: PRESERVED_CANDIDATE_RECEIPT_INVALID")
                                verification = receipt.get("verification")
                                if not isinstance(verification, dict) or verification.get("byte_identity_confirmed") is not True:
                                    errors.append("LH-COMMERCE-1: PRESERVED_CANDIDATE_BYTE_IDENTITY_UNCONFIRMED")
                expected_anchor = CTA_HERO_ANCHORS.get(collection, {}).get(scene.get("id"))
                if expected_anchor:
                    anchor = reference.get("hero_continuity_anchor")
                    if not isinstance(anchor, dict):
                        errors.append(f"{scene.get('id')}: HERO_DERIVED_CTA_ANCHOR_MISSING")
                    else:
                        for key, expected in expected_anchor.items():
                            if anchor.get(key) != expected:
                                errors.append(f"{scene.get('id')}: HERO_DERIVED_CTA_ANCHOR_{key.upper()}_MISMATCH")
                        anchor_path = scene_contract_path(anchor.get("path"))
                        if not anchor_path or not anchor_path.is_file():
                            errors.append(f"{scene.get('id')}: HERO_DERIVED_CTA_ANCHOR_SOURCE_MISSING")
                        elif sha256(anchor_path) != expected_anchor["sha256"]:
                            errors.append(f"{scene.get('id')}: HERO_DERIVED_CTA_ANCHOR_HASH_MISMATCH")
                    if scene.get("hero_continuity_anchor_id") != expected_anchor["id"]:
                        errors.append(f"{scene.get('id')}: MANIFEST_HERO_DERIVED_CTA_ANCHOR_MISMATCH")
            candidate_contract = scene_contract_path(scene.get("contract"))
            if candidate_contract:
                try:
                    native = load(candidate_contract)
                except ValueError as error:
                    errors.append(str(error))
                    continue
                native_scene = native.get("collection_scene")
                if not isinstance(native_scene, dict) or native_scene.get("scene_id") != scene.get("id"):
                    errors.append(f"{scene.get('id')}: NATIVE_CONTRACT_ID_MISMATCH")
                if native_scene.get("collection_id") != collection:
                    errors.append(f"{scene.get('id')}: NATIVE_CONTRACT_COLLECTION_MISMATCH")
                expected_anchor = CTA_HERO_ANCHORS.get(collection, {}).get(scene.get("id"))
                if expected_anchor:
                    native_anchor = native.get("hero_continuity_anchor")
                    if not isinstance(native_anchor, dict):
                        errors.append(f"{scene.get('id')}: NATIVE_HERO_DERIVED_CTA_ANCHOR_MISSING")
                    else:
                        for key, expected in expected_anchor.items():
                            if native_anchor.get(key) != expected:
                                errors.append(f"{scene.get('id')}: NATIVE_HERO_DERIVED_CTA_ANCHOR_{key.upper()}_MISMATCH")
                if collection == "love-hurts":
                    native_rose_anchor = native.get("love_hurts_rose_anchor")
                    if not isinstance(native_rose_anchor, dict):
                        errors.append(f"{scene.get('id')}: NATIVE_LOVE_HURTS_ENCHANTED_ROSE_ANCHOR_MISSING")
                    elif any(native_rose_anchor.get(key) != expected for key, expected in LOVE_HURTS_ROSE_ANCHOR.items()):
                        errors.append(f"{scene.get('id')}: NATIVE_LOVE_HURTS_ENCHANTED_ROSE_ANCHOR_INVALID")
                    references = native.get("references")
                    if not isinstance(references, list) or not any(
                        isinstance(item, dict)
                        and item.get("role") == "protected_world_authority"
                        and all(item.get(key) == expected for key, expected in LOVE_HURTS_ROSE_ANCHOR.items() if key != "id" and key != "kind")
                        for item in references
                    ):
                        errors.append(f"{scene.get('id')}: NATIVE_LOVE_HURTS_ENCHANTED_ROSE_REFERENCE_MISSING")
                    native_text = json.dumps(native).lower()
                    if "scene-love-hurts-cathedral.webp" in native_text:
                        errors.append(f"{scene.get('id')}: STALE_LOVE_HURTS_CATHEDRAL_ROUTE")
                blockers.extend(str(item) for item in native.get("pre_generation_blockers", []) if isinstance(item, str))

    signature_terms = json.dumps(galleries.get("signature", {})).lower()
    if "golden gate" in signature_terms or "san francisco" in signature_terms:
        errors.append("signature: GENERIC_SF_OR_GOLDEN_GATE_RUNTIME_ROUTE")
    if "obsolete displaced-cloud pairing" in json.dumps(galleries.get("black-rose", {})).lower():
        errors.append("black-rose: OBSOLETE_DISPLACED_CLOUD_ROUTE")

    generic_pipeline_paths = (
        ROOT / "scripts/oai_render/prompt.py",
        ROOT / "scripts/oai_render/scene_schema.py",
    )
    for pipeline_path in generic_pipeline_paths:
        try:
            pipeline_text = pipeline_path.read_text(encoding="utf-8").lower()
        except OSError:
            errors.append(f"scripts: SCENE_PIPELINE_MISSING_{pipeline_path.name}")
            continue
        if "golden gate bridge and bay area skyline" in pipeline_text:
            errors.append(f"scripts: STALE_SIGNATURE_GOLDEN_GATE_ROUTE_{pipeline_path.name}")
        if (
            "oakland pier viewpoint" not in pipeline_text
            or "exact bay bridge span" not in pipeline_text
            or "generic san francisco" not in pipeline_text
            or "landmark" not in pipeline_text
        ):
            errors.append(f"scripts: SIGNATURE_OAKLAND_BAY_BRIDGE_LOCK_MISSING_{pipeline_path.name}")
        if "exact protected enchanted rose under glass" not in pipeline_text:
            errors.append(f"scripts: LOVE_HURTS_PROTECTED_ROSE_LOCK_MISSING_{pipeline_path.name}")
        if "never depict a literal beast" not in pipeline_text and "never a literal beast" not in pipeline_text:
            errors.append(f"scripts: LOVE_HURTS_LITERAL_BEAST_PROHIBITION_MISSING_{pipeline_path.name}")

    br004_lock = contract.get("product_fidelity_overrides", {}).get("br-004")
    if not isinstance(br004_lock, dict):
        errors.append("br-004: FOUNDER_REGULAR_HOODIE_LOCK_MISSING")
    else:
        br004_prohibited = br004_lock.get("prohibited_on_br004")
        br004_required = {"side-body embroidery", "right-chest silicone cutout", "sublimated-rose hood lining"}
        if not isinstance(br004_prohibited, list) or not br004_required.issubset(set(br004_prohibited)):
            errors.append("br-004: REGULAR_HOODIE_SEPARATION_INCOMPLETE")
        if "longer pullover" not in str(br004_lock.get("silhouette_lock", "")).lower():
            errors.append("br-004: LONG_REGULAR_HOODIE_LOCK_MISSING")
        if "centered" not in str(br004_lock.get("branding_lock", "")).lower():
            errors.append("br-004: CENTERED_CHEST_EMBROIDERY_LOCK_MISSING")

    br005_lock = contract.get("product_fidelity_overrides", {}).get("br-005")
    if not isinstance(br005_lock, dict):
        errors.append("br-005: FOUNDER_SIDE_BODY_LOCK_MISSING")
    else:
        prohibited = br005_lock.get("prohibited_on_br005")
        required = {"Signature Edition lettering", "product-name lettering", "circular patch"}
        if not isinstance(prohibited, list) or not required.issubset(set(prohibited)):
            errors.append("br-005: FOUNDER_SIDE_BODY_PROHIBITIONS_INCOMPLETE")
        side_body_lock = str(br005_lock.get("side_body_lock", ""))
        if "exact photographed longitudinal Black Rose rose/cloud embroidery" not in side_body_lock or "side body" not in side_body_lock:
            errors.append("br-005: PHOTOGRAPHED_SIDE_BODY_EMBROIDERY_LOCK_MISSING")
        if "silicone-cut" not in str(br005_lock.get("right_chest_lock", "")):
            errors.append("br-005: RIGHT_CHEST_SILICONE_LOCK_MISSING")
        if "sublimated" not in str(br005_lock.get("hood_lining_lock", "")).lower():
            errors.append("br-005: SUBLIMATED_HOOD_LINING_LOCK_MISSING")
        material_locks = br005_lock.get("material_physics_locks")
        expected_material_regions = {
            "front-right-chest": ("silicone", "rubber-like", "thread"),
            "wearer's-left side body / viewer-right torso": (
                "embroidery thread",
                "stitch texture",
                "sleeve",
            ),
            "hood-inside / inner-hood-lining": ("sublimation dye", "zero raised", "patch"),
        }
        if not isinstance(material_locks, dict):
            errors.append("br-005: MATERIAL_PHYSICS_LOCKS_MISSING")
        else:
            for region, required_terms in expected_material_regions.items():
                lock = material_locks.get(region)
                if not isinstance(lock, dict):
                    errors.append(f"br-005: MATERIAL_PHYSICS_REGION_MISSING_{region}")
                    continue
                if not {"material", "surface", "attachment", "reject"}.issubset(lock):
                    errors.append(f"br-005: MATERIAL_PHYSICS_FIELDS_INCOMPLETE_{region}")
                lock_text = json.dumps(lock).lower()
                if any(term not in lock_text for term in required_terms):
                    errors.append(f"br-005: MATERIAL_PHYSICS_CUES_INCOMPLETE_{region}")

    for error in errors:
        print(f"FAIL {error}")
    if errors:
        return 1
    if args.authorize_paid_scene:
        requested_scene_id = args.authorize_paid_scene
        requested_collection = next(
            collection
            for collection, scene_ids in EXPECTED.items()
            if requested_scene_id in scene_ids
        )
        requested_reference = next(
            (
                reference
                for reference in reference_collections[requested_collection]
                if reference.get("scene_reference_id") == requested_scene_id
            ),
            None,
        )
        requested_scene = next(
            (
                scene
                for scene in galleries[requested_collection]["scenes"]
                if scene.get("id") == requested_scene_id
            ),
            None,
        )
        authorization_denials: list[str] = []
        if not isinstance(requested_reference, dict) or not isinstance(requested_scene, dict):
            authorization_denials.append("DECLARED_SCENE_RECORD_MISSING")
        else:
            if requested_reference.get("provider_generation_allowed") is not True:
                authorization_denials.append("SCENE_REFERENCE_PROVIDER_GENERATION_NOT_CLEARED")
            if requested_scene.get("status") != "SOURCE_READY_FOR_PAID_CANDIDATE":
                authorization_denials.append("MANIFEST_SOURCE_READY_STATUS_REQUIRED")
            contract_path = scene_contract_path(requested_reference.get("native_scene_contract"))
            if not contract_path or not contract_path.is_file():
                authorization_denials.append("NATIVE_SCENE_CONTRACT_REQUIRED")
            else:
                native_contract = load(contract_path)
                if native_contract.get("pre_generation_blockers"):
                    authorization_denials.append("NATIVE_CONTRACT_PRE_GENERATION_BLOCKERS_PRESENT")
                if native_contract.get("generator", {}).get("product_redraw_allowed") is not False:
                    authorization_denials.append("PROTECTED_COMPOSITE_ROUTE_REQUIRED")
        if authorization_denials:
            print("DENY_PAID_SCENE " + requested_scene_id + " " + ",".join(authorization_denials))
            return 3
        print(f"AUTHORIZE_PAID_SCENE {requested_scene_id}")
        return 0
    print("PASS V2 Scroll World contract is fail-closed: exactly three declared roles per collection; no legacy environment fallback or random product routing.")
    print("BLOCKED " + ",".join(sorted(set(blockers))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
