from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path

import pytest
from PIL import Image

from Comfy.scripts.native_scene_workflows import (
    BIREFNET_MODEL,
    build_birefnet_segmentation_workflow,
    build_lh_native_composite_workflow,
    verify_protected_foreground_rgb,
)
from skyyrose.integrations import comfy_client


def _class_types(workflow: dict) -> set[str]:
    return {node["class_type"] for node in workflow.values()}


def _png(mode: str, pixels: list[object], size: tuple[int, int] = (2, 1)) -> bytes:
    image = Image.new(mode, size)
    image.putdata(pixels)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _coverage_contract(
    tmp_path: Path,
    source: bytes,
    *,
    minimum: float = 0.5,
    maximum: float = 1.0,
) -> Path:
    model = tmp_path / "birefnet.safetensors"
    model.write_bytes(b"test-birefnet-weights")
    source_sha256 = hashlib.sha256(source).hexdigest()
    baseline = tmp_path / "coverage-baseline.json"
    baseline.write_text(
        json.dumps(
            {
                "schema": "skyyrose.foreground-coverage-baseline/1",
                "scene_id": "LH-COMMERCE-2",
                "source_sha256": source_sha256,
                "reviewer_role": "visual-commerce-qa",
                "verdict": "PASS",
                "minimum_coverage_ratio": minimum,
                "maximum_coverage_ratio": maximum,
            }
        ),
        encoding="utf-8",
    )
    authority = tmp_path / "coverage.json"
    authority.write_text(
        json.dumps(
            {
                "schema": "skyyrose.foreground-coverage-authority/1",
                "state": "APPROVED",
                "scene_id": "LH-COMMERCE-2",
                "source_sha256": source_sha256,
                "minimum_coverage_ratio": minimum,
                "maximum_coverage_ratio": maximum,
                "independent_baseline_receipt": {
                    "path": str(baseline),
                    "sha256": hashlib.sha256(baseline.read_bytes()).hexdigest(),
                },
                "segmentation_model": {
                    "id": "birefnet",
                    "path": str(model),
                    "sha256": hashlib.sha256(model.read_bytes()).hexdigest(),
                },
            }
        ),
        encoding="utf-8",
    )
    contract = tmp_path / "scene.json"
    contract.write_text(
        json.dumps(
            {
                "schema": "skyyrose.scene-ooda/1",
                "scene_id": "LH-COMMERCE-2",
                "native_scene_workflow": {
                    "segmentation": {
                        "coverage_authority": {
                            "path": str(authority),
                            "sha256": hashlib.sha256(authority.read_bytes()).hexdigest(),
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return contract


def _trust_contract(monkeypatch: pytest.MonkeyPatch, contract: Path) -> tuple[str, str]:
    scene_id = "LH-COMMERCE-2"
    monkeypatch.setitem(comfy_client.CANONICAL_SCENE_CONTRACTS, scene_id, contract)
    return scene_id, hashlib.sha256(contract.read_bytes()).hexdigest()


def _protected_evidence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    source = _png("RGB", [(10, 20, 30), (40, 50, 60)])
    protected = _png("RGBA", [(10, 20, 30, 255), (40, 50, 60, 0)])
    mask = _png("L", [255, 0])
    contract = _coverage_contract(tmp_path, source)
    scene_id, contract_sha256 = _trust_contract(monkeypatch, contract)
    verification = verify_protected_foreground_rgb(
        source_bytes=source,
        protected_rgba_bytes=protected,
        alpha_mask_bytes=mask,
        governing_scene_id=scene_id,
        expected_scene_contract_sha256=contract_sha256,
    )
    receipt = {
        "schema": "skyyrose.comfy-execution-receipt/1",
        "node_classes": ["JoinImageWithAlpha", "SaveImage"],
        "protected_asset": {
            "source_upload": {"sha256": verification["source_sha256"]},
            "foreground_output": {
                "filename": "accepted-foreground.png",
                "sha256": verification["protected_output_sha256"],
            },
            "mask_output": {"sha256": verification["alpha_mask_sha256"]},
            "verification": verification,
        },
    }
    return {
        "source": source,
        "protected": protected,
        "mask": mask,
        "contract": contract,
        "scene_id": scene_id,
        "contract_sha256": contract_sha256,
        "verification": verification,
        "receipt": receipt,
    }


def test_segmentation_changes_alpha_without_a_rgb_generation_node() -> None:
    workflow = build_birefnet_segmentation_workflow(candidate_filename="accepted-vto.png")
    class_types = _class_types(workflow)

    assert "LoadBackgroundRemovalModel" in class_types
    assert "RemoveBackground" in class_types
    assert "JoinImageWithAlpha" in class_types
    assert workflow["2"]["inputs"]["bg_removal_name"] == BIREFNET_MODEL
    assert not any("Runway" in node for node in class_types)


def test_native_composite_uses_only_deterministic_compositing_nodes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _protected_evidence(tmp_path, monkeypatch)
    workflow = build_lh_native_composite_workflow(
        background_filename="background.png",
        protected_foreground_filename="accepted-foreground.png",
        protected_rose_filename="protected-rose.png",
        protected_asset_receipt=evidence["receipt"],
        protected_source_bytes=evidence["source"],
        protected_foreground_bytes=evidence["protected"],
        protected_mask_bytes=evidence["mask"],
        governing_scene_id=evidence["scene_id"],
        expected_scene_contract_sha256=evidence["contract_sha256"],
    )
    class_types = _class_types(workflow)

    assert "ImageCompositeMasked" in class_types
    assert "SaveImage" in class_types
    assert "RunwayTextToImageNode" not in class_types
    assert "KSampler" not in class_types
    assert workflow["12"]["_meta"]["title"] == "Protected LH-003 customer placement"
    assert workflow["22"]["_meta"]["title"] == "Protected rose monument placement"


def test_workflow_builders_fail_closed_on_missing_bindings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _protected_evidence(tmp_path, monkeypatch)
    with pytest.raises(ValueError):
        build_birefnet_segmentation_workflow(candidate_filename="")
    with pytest.raises(ValueError):
        build_lh_native_composite_workflow(
            background_filename="",
            protected_foreground_filename="foreground.png",
            protected_rose_filename="rose.png",
            protected_asset_receipt=evidence["receipt"],
            protected_source_bytes=evidence["source"],
            protected_foreground_bytes=evidence["protected"],
            protected_mask_bytes=evidence["mask"],
            governing_scene_id=evidence["scene_id"],
            expected_scene_contract_sha256=evidence["contract_sha256"],
        )


def test_protected_rgb_verifier_detects_one_changed_subject_pixel(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _png("RGB", [(10, 20, 30), (40, 50, 60)])
    protected = _png("RGBA", [(11, 20, 30, 255), (40, 50, 60, 0)])
    mask = _png("L", [255, 0])

    contract = _coverage_contract(tmp_path, source)
    scene_id, contract_sha256 = _trust_contract(monkeypatch, contract)
    receipt = verify_protected_foreground_rgb(
        source_bytes=source,
        protected_rgba_bytes=protected,
        alpha_mask_bytes=mask,
        governing_scene_id=scene_id,
        expected_scene_contract_sha256=contract_sha256,
    )

    assert receipt["status"] == "BLOCKED"
    assert receipt["changed_pixel_count"] == 1
    assert receipt["maximum_channel_delta"] == 1


def test_protected_rgb_verifier_rejects_erased_or_shrunk_foreground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _png("RGB", [(10, 20, 30), (40, 50, 60)])
    contract = _coverage_contract(tmp_path, source, minimum=0.75)
    scene_id, contract_sha256 = _trust_contract(monkeypatch, contract)
    for protected, mask in (
        (
            _png("RGBA", [(99, 99, 99, 0), (88, 88, 88, 0)]),
            _png("L", [0, 0]),
        ),
        (
            _png("RGBA", [(10, 20, 30, 255), (40, 50, 60, 0)]),
            _png("L", [255, 0]),
        ),
    ):
        receipt = verify_protected_foreground_rgb(
            source_bytes=source,
            protected_rgba_bytes=protected,
            alpha_mask_bytes=mask,
            governing_scene_id=scene_id,
            expected_scene_contract_sha256=contract_sha256,
        )
        assert receipt["status"] == "BLOCKED"
        assert receipt["coverage_pass"] is False

    erased = verify_protected_foreground_rgb(
        source_bytes=source,
        protected_rgba_bytes=_png("RGBA", [(99, 99, 99, 0), (88, 88, 88, 0)]),
        alpha_mask_bytes=_png("L", [0, 0]),
        governing_scene_id=scene_id,
        expected_scene_contract_sha256=contract_sha256,
    )
    assert erased["changed_pixel_count"] == 2


def test_composition_rejects_unverified_foreground(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    evidence = _protected_evidence(tmp_path, monkeypatch)
    receipt = evidence["receipt"]
    verification = receipt["protected_asset"]["verification"]
    verification["changed_pixel_count"] = 1
    verification["status"] = "BLOCKED"

    with pytest.raises(ValueError, match="bound passing"):
        build_lh_native_composite_workflow(
            background_filename="background.png",
            protected_foreground_filename="foreground.png",
            protected_rose_filename="rose.png",
            protected_asset_receipt=receipt,
            protected_source_bytes=evidence["source"],
            protected_foreground_bytes=evidence["protected"],
            protected_mask_bytes=evidence["mask"],
            governing_scene_id=evidence["scene_id"],
            expected_scene_contract_sha256=evidence["contract_sha256"],
        )
