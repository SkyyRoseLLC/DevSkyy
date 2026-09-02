from __future__ import annotations

import importlib.util
import json
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "scene_ooda.py"
SPEC = importlib.util.spec_from_file_location("scene_ooda", SCRIPT)
assert SPEC and SPEC.loader
scene_ooda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scene_ooda)


def test_higgsfield_command_is_enhance_only_and_source_bound(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    manifest = {
        "higgsfield": {
            "mode": "hero_banner",
            "prompt": "campaign",
            "aspect_ratio": "16:9",
            "count": 1,
            "brand_context": "brand",
            "product_context": "product",
        },
        "source_bindings": [
            {
                "path": str(source),
                "send_to_higgsfield": True,
            }
        ],
    }

    command = scene_ooda.higgsfield_command(manifest, enhance_only=True)

    assert command[:4] == [
        "higgsfield",
        "--json",
        "product-photoshoot",
        "create",
    ]
    assert command[-1] == "--enhance-only"
    assert command[command.index("--image") + 1] == str(source)


def test_observe_fails_closed_on_missing_dependency(tmp_path: Path) -> None:
    source = tmp_path / "source.png"
    source.write_bytes(b"source")
    digest = scene_ooda.sha256_file(source)
    manifest = {
        "scene_id": "TEST-SCENE",
        "catalog_snapshot": {
            "catalog_path": str(source),
            "catalog_sha256": digest,
            "image_manifest_path": str(source),
            "image_manifest_sha256": digest,
        },
        "source_bindings": [{"path": str(source), "sha256": digest}],
        "authority_dependencies": [{"path": str(tmp_path / "missing.png")}],
        "execute_blockers": [],
    }

    report = scene_ooda.observe(manifest)

    assert report["source_ready"] is True
    assert report["dependencies_ready"] is False
    assert report["paid_execution_ready"] is False


def test_enhanced_prompt_inspection_rejects_product_drift() -> None:
    manifest = {
        "higgsfield": {
            "prompt_assertions": {
                "required_phrases": ["right chest", "side-body"],
                "forbidden_phrases": ["left sleeve", "sleeve graphic"],
            }
        }
    }
    result = {
        "stdout": {
            "prompts": [{"prompt": "A left sleeve graphic replaces the right chest artwork."}]
        }
    }

    inspection = scene_ooda.inspect_enhanced_prompt(manifest, result)

    assert inspection["status"] == "REJECTED_ENHANCER_PRODUCT_DRIFT"
    assert inspection["missing_required_phrases"] == ["side-body"]
    assert inspection["present_forbidden_phrases"] == ["left sleeve", "sleeve graphic"]


def test_enhanced_prompt_inspection_accepts_bound_product_facts() -> None:
    manifest = {
        "higgsfield": {
            "prompt_assertions": {
                "required_phrases": ["right chest", "side-body", "OAKLAND"],
                "forbidden_phrases": ["left sleeve"],
            }
        }
    }
    result = {
        "stdout": {
            "prompts": [
                {
                    "prompt": "Small rose on the right chest, floral side-body artwork, and OAKLAND shorts."
                }
            ]
        }
    }

    inspection = scene_ooda.inspect_enhanced_prompt(manifest, result)

    assert inspection["status"] == "PASS"
    assert inspection["missing_required_phrases"] == []
    assert inspection["present_forbidden_phrases"] == []


def test_all_remaining_scene_manifests_follow_team_ooda() -> None:
    contract_root = Path(__file__).resolve().parents[1] / "scene-contracts"
    index = json.loads((contract_root / "scene-ooda-index.json").read_text(encoding="utf-8"))

    assert index["plugin"] == "fashion-theme-team@personal"
    assert index["phase_ownership"]["decide"]["independent"] is True
    assert index["candidate_rules"]["rejected_assets_are_evidence_only"] is True
    assert index["phase_ownership"]["promotion"]["required_state"] == "FOUNDER_APPROVED_VISUAL"

    expected_scenes = {
        "BR-COMMERCE-2",
        "LH-COMMERCE-2",
        "SIG-COMMERCE-1",
        "SIG-COMMERCE-2",
        "SIG-COMMERCE-3",
    }
    actual_scenes = set()
    for entry in index["scenes"]:
        manifest_path = scene_ooda.resolve_path(entry["contract"])
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        actual_scenes.add(manifest["scene_id"])
        assert manifest["higgsfield"]["mode"] == "hero_banner"
        assert manifest["higgsfield"]["aspect_ratio"] == "16:9"
        assert manifest["higgsfield"]["count"] == 1
        assert manifest["execute_blockers"]
        assert manifest["promotion_gate"] == "FOUNDER_APPROVED_VISUAL"
        assert scene_ooda.verify_team_contract(manifest)["status"] == "PASS"

    assert actual_scenes == expected_scenes


def test_rejected_source_cannot_be_routed_to_higgsfield(tmp_path: Path) -> None:
    allowed = tmp_path / "allowed.png"
    rejected = tmp_path / "rejected.png"
    allowed.write_bytes(b"allowed")
    rejected.write_bytes(b"rejected")
    manifest = {
        "scene_id": "TEST-SCENE",
        "catalog_snapshot": {
            "catalog_path": str(allowed),
            "catalog_sha256": scene_ooda.sha256_file(allowed),
            "image_manifest_path": str(allowed),
            "image_manifest_sha256": scene_ooda.sha256_file(allowed),
        },
        "source_bindings": [
            {
                "path": str(rejected),
                "sha256": scene_ooda.sha256_file(rejected),
                "send_to_higgsfield": True,
            }
        ],
        "forbidden_inputs": [{"path": str(rejected), "sha256": scene_ooda.sha256_file(rejected)}],
        "authority_dependencies": [],
        "execute_blockers": [],
        "fashion_theme_team": {
            "plugin": "fashion-theme-team@personal",
            "team_contract": "Comfy/scene-contracts/scene-ooda-index.json",
            "phase_id": "TEST-1",
            "accountable": "fashion-theme-lead",
            "next_authority": "visual-commerce-qa",
        },
    }

    report = scene_ooda.observe(manifest)

    assert report["forbidden_routing_clean"] is False
    assert report["paid_execution_ready"] is False
