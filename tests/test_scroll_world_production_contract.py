"""Regression coverage for the V2 native Scroll World production gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "scripts/validate-scroll-world-production.py"
MANIFEST = ROOT / "wordpress-theme/skyyrose-flagship-2/data/scroll-world-production-manifest.json"
CONTRACT = ROOT / "docs/design/v2-remodel/native-scene-regeneration-v2/scroll-world-production/contract.json"
SCENE_REFERENCE_INDEX = ROOT / "data/scroll-world-scene-reference-index.json"


def _load_validator():
    spec = importlib.util.spec_from_file_location("scroll_world_production_validator", VALIDATOR)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_manifest_has_only_the_three_declared_roles_per_collection() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected_roles = ["hero_world_moment_1", "hero_world_moment_2", "collection_preorder"]
    for collection in ("black-rose", "love-hurts", "signature"):
        gallery = manifest["collections"][collection]
        assert gallery["required_roles"] == expected_roles
        assert [scene["role"] for scene in gallery["scenes"]] == expected_roles
        assert len(gallery["scenes"]) == 3
        for scene in gallery["scenes"]:
            asset = scene["asset"]
            if asset is None:
                continue
            assert scene["status"].startswith("FOUNDER_APPROVED")
            assert asset["runtime_wiring_allowed"] is False
            assert (ROOT / asset["path"]).is_file()


def test_two_hero_scenes_per_collection_require_mixed_casts_and_four_distinct_skus() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    index = json.loads(SCENE_REFERENCE_INDEX.read_text(encoding="utf-8"))
    for record in (
        contract["hero_dual_cast_four_sku_rule"],
        manifest["hero_dual_cast_four_sku_rule"],
        index["hero_dual_cast_four_sku_rule"],
    ):
        assert record["status"] == "FOUNDER_DIRECTIVE_BINDING"
        assert record["required_roles"] == ["hero_world_moment_1", "hero_world_moment_2"]
        assert record["required_scene_wearers"] == ["man", "woman"]
        assert record["minimum_distinct_exact_skus_across_two_hero_moments"] == 4
    assert manifest["hero_dual_cast_four_sku_rule"]["wiring_allowed_before_coverage"] is False
    assert "two hero-derived world moments together" in contract["hero_dual_cast_four_sku_rule"]["scope"].lower()
    assert "two hero-derived world moments together" in index["hero_dual_cast_four_sku_rule"]["scope"].lower()
    assert "four distinct exact skus" in contract["hero_dual_cast_four_sku_rule"]["scene_composition"].lower()
    assert "four distinct exact skus" in index["hero_dual_cast_four_sku_rule"]["sku_allocation"].lower()

    for collection in ("black-rose", "signature"):
        hero_skus = {
            sku
            for scene in manifest["collections"][collection]["scenes"]
            if scene["role"] in {"hero_world_moment_1", "hero_world_moment_2"}
            for sku in scene["product_skus"]
        }
        assert len(hero_skus) >= 4
    love_hurts_hero_skus = {
        sku
        for scene in manifest["collections"]["love-hurts"]["scenes"]
        if scene["role"] in {"hero_world_moment_1", "hero_world_moment_2"}
        for sku in scene["product_skus"]
    }
    assert love_hurts_hero_skus == {"lh-004", "lh-002", "lh-006", "lh-003"}
    assert (
        index["hero_dual_cast_four_sku_rule"]["collection_plan_status"]["love-hurts"]
        == "DECLARED_FOUR_SKU_HERO_CAST__PRESERVED_VOW_AISLE_LOCAL_HASH_VERIFIED_QUARANTINED"
    )


def test_every_runtime_scene_has_one_exact_root_owned_scene_reference() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    index = json.loads(SCENE_REFERENCE_INDEX.read_text(encoding="utf-8"))
    assert index["status"] == "ALL_COLLECTION_SCENE_DIRECTIONS_FOUNDER_APPROVED__SOURCE_AND_PROMOTION_GATES_REMAIN_BLOCKED"
    assert all(
        item is False
        for scenes in index["collection_scenes"].values()
        for item in (scene["provider_generation_allowed"] for scene in scenes)
    )
    for collection in ("black-rose", "love-hurts", "signature"):
        references = {
            scene["scene_reference_id"]: scene
            for scene in index["collection_scenes"][collection]
        }
        gallery = manifest["collections"][collection]
        assert len(references) == 3
        for scene in gallery["scenes"]:
            reference = references[scene["scene_reference_id"]]
            assert scene["scene_reference_id"] == scene["id"]
            assert reference["role"] == scene["role"]
            assert reference["exact_skus"] == scene["product_skus"]
            assert reference["exact_cta"] == scene["cta"]
            assert reference["native_scene_contract"] == scene.get("contract")


def test_all_collection_scenes_have_independently_bound_exact_ctas_and_casts() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    expected = {
        "black-rose": {
            "BR-COMMERCE-1": ("hero_world_moment_1", {"label": "Complete the Set", "href": "#shop"}, {"br-001", "br-002"}),
            "BR-COMMERCE-2": ("hero_world_moment_2", {"label": "Complete the Look", "href": "#shop"}, {"br-005", "br-007", "br-004"}),
            "BR-COMMERCE-3": ("collection_preorder", {"label": "Pre-Order the Jersey Series", "href": "#jersey-series"}, {"br-003", "br-008", "br-009", "br-010", "br-011", "br-012", "br-014", "br-015"}),
        },
        "love-hurts": {
            "LH-COMMERCE-1": ("hero_world_moment_1", {"label": "Complete the Look", "href": "#shop"}, {"lh-004", "lh-002", "lh-006"}),
            "LH-COMMERCE-2": ("hero_world_moment_2", {"label": "Shop the Shorts", "href": "#shop"}, {"lh-003"}),
            "LH-COMMERCE-3": ("collection_preorder", {"label": "Pre-Order The Fannie", "href": "/pre-order/"}, {"lh-005"}),
        },
        "signature": {
            "SIG-COMMERCE-1": ("hero_world_moment_1", {"label": "Complete the Look", "href": "#shop"}, {"sg-009", "sg-007"}),
            "SIG-COMMERCE-2": ("hero_world_moment_2", {"label": "Complete the Look", "href": "#shop"}, {"sg-013", "sg-014", "sg-006"}),
            "SIG-COMMERCE-3": ("collection_preorder", {"label": "Explore Pre-Orders", "href": "/pre-order/"}, {"sg-005", "sg-001", "sg-002", "sg-003", "sg-015"}),
        },
    }
    for collection, scene_locks in expected.items():
        scenes = {scene["id"]: scene for scene in manifest["collections"][collection]["scenes"]}
        assert set(scenes) == set(scene_locks)
        for scene_id, (role, cta, product_skus) in scene_locks.items():
            assert scenes[scene_id]["role"] == role
            assert scenes[scene_id]["cta"] == cta
            assert set(scenes[scene_id]["product_skus"]) == product_skus


def test_black_rose_and_signature_ctas_have_exact_hero_derived_anchor_coverage() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    index = json.loads(SCENE_REFERENCE_INDEX.read_text(encoding="utf-8"))
    expected = {
        "black-rose": {
            "BR-COMMERCE-1": ("BR_FONT_STATUE", "font_statue"),
            "BR-COMMERCE-2": ("BR_STAR_GRAPHIC", "hero_graphic"),
            "BR-COMMERCE-3": ("BR_BAY_BRIDGE_NIGHT_BACKDROP", "scene_backdrop"),
        },
        "signature": {
            "SIG-COMMERCE-1": ("SIG_FONT_STATUE", "font_statue"),
            "SIG-COMMERCE-2": ("SIG_GOLDEN_GATE_MONUMENTS_HERO", "scene_backdrop_and_gold_monuments"),
            "SIG-COMMERCE-3": ("SIG_SR_ROSE_GRAPHIC", "hero_graphic"),
        },
    }
    for record in (
        contract["hero_derived_cta_anchor_rule"],
        manifest["hero_derived_cta_anchor_rule"],
        index["hero_derived_cta_anchor_rule"],
    ):
        assert record["status"] == "FOUNDER_DIRECTIVE_BINDING"
        assert record["required_anchor_kinds"] == ["font_statue", "scene_backdrop", "hero_graphic", "scene_backdrop_and_gold_monuments"]
    for collection, assignments in expected.items():
        references = {item["scene_reference_id"]: item for item in index["collection_scenes"][collection]}
        scenes = {item["id"]: item for item in manifest["collections"][collection]["scenes"]}
        assert set(references) == set(assignments)
        for scene_id, (anchor_id, kind) in assignments.items():
            anchor = references[scene_id]["hero_continuity_anchor"]
            assert (anchor["id"], anchor["kind"]) == (anchor_id, kind)
            assert scenes[scene_id]["hero_continuity_anchor_id"] == anchor_id
            assert (ROOT / anchor["path"]).is_file()
    preorder_lock = contract["hero_derived_cta_anchor_rule"]["black_rose_preorder_lock"]
    assert "Jersey Series intimate luxury-night salon" in preorder_lock
    assert "Bay Bridge backdrop visible behind the collection" in preorder_lock


def test_love_hurts_keeps_its_protected_rose_and_preserved_mixed_bomber_cast() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    index = json.loads(SCENE_REFERENCE_INDEX.read_text(encoding="utf-8"))
    expected_assignments = {
        "LH-COMMERCE-1": "ceremonial_vow_aisle_approach_to_the_protected_rose",
        "LH-COMMERCE-2": "fractured_chamber_with_protected_rose_sharp_opposite_the_product",
        "LH-COMMERCE-3": "intimate_vitrine_with_the_fannie_foreground_and_protected_rose_legible",
    }
    for record in (
        contract["love_hurts_enchanted_rose_story_rule"],
        manifest["love_hurts_enchanted_rose_story_rule"],
        index["love_hurts_enchanted_rose_story_rule"],
    ):
        assert record["status"] == "FOUNDER_DIRECTIVE_BINDING"
        assert record["scene_assignments"] == expected_assignments
        assert "literal Beast" in record["beast_perspective"]
    anchor = index["love_hurts_enchanted_rose_story_rule"]["protected_rose_anchor"]
    assert anchor["id"] == "LH_ENCHANTED_ROSE_CATHEDRAL"
    assert (ROOT / anchor["path"]).is_file()
    vow_reference = index["collection_scenes"]["love-hurts"][0]
    vow_runtime = manifest["collections"]["love-hurts"]["scenes"][0]
    expected_cast = ["lh-004", "lh-002", "lh-006"]
    expected_wearers = [
        {"wearer": "man", "look_id": "lh-bomber-black", "skus": ["lh-004", "lh-002"]},
        {"wearer": "woman", "look_id": "lh-bomber-white", "skus": ["lh-004", "lh-006"]},
    ]
    assert vow_reference["exact_skus"] == expected_cast
    assert vow_runtime["product_skus"] == expected_cast
    assert vow_reference["founder_declared_wearer_sku_assignments"] == expected_wearers
    assert vow_runtime["founder_declared_wearer_sku_assignments"] == expected_wearers
    assert vow_runtime["status"] == "FOUNDER_APPROVED_PRESERVE_DO_NOT_REGENERATE__RUNTIME_PROMOTION_PENDING"
    assert vow_runtime["asset"] is None
    candidate = vow_runtime["local_preserved_candidate"]
    assert candidate["sha256"] == "1eb217504536634047eb3a682915a4d8ea0218b9968c9d3ff96addb231578f38"
    assert (ROOT / candidate["path"]).is_file()
    assert vow_runtime["missing_requirements"] == ["HASH_BOUND_RUNTIME_PROMOTION_RECEIPT_REQUIRED"]


def test_founder_approved_scene_assets_are_hash_bound_and_not_runtime_promoted() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    approval_record_path = ROOT / "data/candidates/founder-scene-approvals-2026-08-29.json"
    approval_record = json.loads(approval_record_path.read_text(encoding="utf-8"))
    approvals = {item["scene_id"]: item for item in approval_record["approvals"]}
    assert set(approvals) == {"BR-COMMERCE-1", "LH-COMMERCE-1", "LH-COMMERCE-3"}
    assert approval_record["promotion_boundary"] == {
        "runtime_wiring_allowed": False,
        "deployment_allowed": False,
        "required_next_evidence": "HASH_BOUND_RUNTIME_PROMOTION_RECEIPT_REQUIRED",
    }
    for collection, scene_id in (("black-rose", "BR-COMMERCE-1"), ("love-hurts", "LH-COMMERCE-3")):
        scene = next(item for item in manifest["collections"][collection]["scenes"] if item["id"] == scene_id)
        asset = scene["asset"]
        approval = approvals[scene_id]
        assert asset["path"] == approval["candidate_path"]
        assert asset["sha256"] == approval["candidate_sha256"]
        assert asset["runtime_wiring_allowed"] is False


def test_preserved_scenes_cannot_be_recreated_from_generic_descriptions() -> None:
    index = json.loads(SCENE_REFERENCE_INDEX.read_text(encoding="utf-8"))
    preserved = (
        index["collection_scenes"]["black-rose"][2],
        index["collection_scenes"]["love-hurts"][0],
    )
    assert [scene["reference_state"] for scene in preserved] == [
        "PRESERVE_ONLY_NOT_LOCALLY_VERIFIABLE",
        "PRESERVED_LOCAL_HASH_VERIFIED_QUARANTINED",
    ]
    assert all(scene["provider_generation_allowed"] is False for scene in preserved)
    assert all(scene["native_scene_contract"] is None for scene in preserved)


def test_signature_runtime_manifest_has_no_generic_sf_or_golden_gate_route() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    serialized = json.dumps(manifest["collections"]["signature"]).lower()
    assert "golden gate" not in serialized
    assert "san francisco" not in serialized


def test_general_on_model_scripts_cannot_reintroduce_stale_collection_worlds() -> None:
    prompt_script = (ROOT / "scripts/oai_render/prompt.py").read_text(encoding="utf-8").lower()
    scene_schema = (ROOT / "scripts/oai_render/scene_schema.py").read_text(encoding="utf-8").lower()
    for script in (prompt_script, scene_schema):
        assert "golden gate bridge and bay area skyline" not in script
        assert "exact bay bridge span" in script
        assert "generic san francisco" in script
        assert "landmark" in script
        assert "exact protected enchanted rose under glass" in script
        assert "literal beast" in script


def test_br005_side_body_lock_prohibits_catalog_name_and_invented_decoration() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    lock = contract["product_fidelity_overrides"]["br-005"]
    assert "exact photographed longitudinal Black Rose rose/cloud embroidery" in lock["side_body_lock"]
    assert "side body" in lock["side_body_lock"]
    assert {"Signature Edition lettering", "product-name lettering", "circular patch"}.issubset(lock["prohibited_on_br005"])
    materials = lock["material_physics_locks"]
    assert "rubber-like" in materials["front-right-chest"]["surface"]
    assert "thread" in materials["front-right-chest"]["reject"]
    side = materials["wearer's-left side body / viewer-right torso"]
    assert "stitch texture" in side["surface"]
    assert "sleeve" in side["reject"]
    lining = materials["hood-inside / inner-hood-lining"]
    assert "zero raised" in lining["surface"]
    assert "patch" in lining["reject"]


def test_black_rose_regular_and_signature_hoodies_are_separate_products() -> None:
    contract = json.loads(CONTRACT.read_text(encoding="utf-8"))
    regular = contract["product_fidelity_overrides"]["br-004"]
    signature = contract["product_fidelity_overrides"]["br-005"]
    assert "longer pullover" in regular["silhouette_lock"].lower()
    assert "centered" in regular["branding_lock"].lower()
    assert {"side-body embroidery", "right-chest silicone cutout", "sublimated-rose hood lining"}.issubset(regular["prohibited_on_br004"])
    assert "silicone-cut" in signature["right_chest_lock"]
    assert "sublimated" in signature["hood_lining_lock"].lower()
    assert "side body" in signature["side_body_lock"]


def test_validator_accepts_the_intentionally_blocked_candidate_state(monkeypatch) -> None:
    validator = _load_validator()
    monkeypatch.setattr(sys, "argv", [str(VALIDATOR)])
    assert validator.main() == 0


def test_paid_scene_authorization_fails_closed_until_a_declared_scene_is_source_ready(monkeypatch) -> None:
    validator = _load_validator()
    monkeypatch.setattr(sys, "argv", [str(VALIDATOR), "--authorize-paid-scene", "LH-COMMERCE-3"])
    assert validator.main() == 3
