"""Focused state-machine tests for the native-scene image workflow."""

from __future__ import annotations

import copy
import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

THEME_DIR = Path(__file__).resolve().parents[1]
SCRIPT = THEME_DIR / "scripts/run-native-scene-image-workflow.py"
CONTRACT = (
    THEME_DIR / "assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/preflight-v1/"
    "vision-authored-prompts/lh-commerce-1-native-scene-regeneration-plan-v1.json"
)
SPEC = importlib.util.spec_from_file_location("native_scene_workflow", SCRIPT)
assert SPEC and SPEC.loader
WORKFLOW = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WORKFLOW)


def contract_copy() -> dict:
    return copy.deepcopy(WORKFLOW.load_json(CONTRACT))


def valid_safe_zone_receipt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setattr(WORKFLOW, "ROOT", tmp_path)
    monkeypatch.setattr(WORKFLOW, "EXPECTED_RENDER_BINDINGS", ["render.css"])
    monkeypatch.setattr(WORKFLOW, "is_git_tracked", lambda _path: True)
    monkeypatch.setattr(
        WORKFLOW.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(stdout="a" * 40 + "\n", returncode=0),
    )
    (tmp_path / "render.css").write_text("body{}\n", encoding="utf-8")
    source = tmp_path / "scene.png"
    screenshot = tmp_path / "safe-zone.png"
    Image.new("RGB", (1672, 941), "black").save(source)
    Image.new("RGB", (100, 100), "white").save(screenshot)
    now = datetime.now(UTC).isoformat()
    source_hash = WORKFLOW.sha256(source)

    def breakpoint(identifier: str) -> dict:
        return {
            "id": identifier,
            "measurement_source": "rendered_v2_dom",
            "source_image_sha256": source_hash,
            "source_dimensions": [1672, 941],
            "crop_transform": {"object_fit": "cover", "object_position": "50% 50%"},
            "screenshot": "safe-zone.png",
            "screenshot_sha256": WORKFLOW.sha256(screenshot),
            "keep_clear": [{"id": "ui", "rect": [0.0, 0.0, 0.25, 0.25]}],
        }

    return {
        "schema": "skyyrose.native-scene-responsive-safe-zones.v1",
        "status": "PASS_MEASURED_SAFE_ZONES",
        "contract_sha256": "c" * 64,
        "coordinate_space": "normalized_generation_frame",
        "measured_at": now,
        "preview_url": "http://127.0.0.1:18888/tools/v2-theme-preview.php?route=love-hurts",
        "preview_identity": {
            "route": "love-hurts",
            "template": "template-collection.php",
            "theme": "SkyyRose Flagship 2 2.4.4",
            "commit": "a" * 40,
        },
        "render_source_bindings": [
            {
                "path": "render.css",
                "sha256": WORKFLOW.sha256(tmp_path / "render.css"),
            }
        ],
        "source_asset": {
            "path": "scene.png",
            "sha256": source_hash,
            "dimensions": [1672, 941],
        },
        "breakpoints": [breakpoint(name) for name in ("desktop", "tablet", "mobile")],
    }


def safe_zone_time(receipt: dict) -> datetime:
    return WORKFLOW.parse_time(receipt["measured_at"], "safe-zone measurement")


def test_candidate_jobs_are_derived_from_the_sealed_batch_count() -> None:
    jobs = WORKFLOW.candidate_job_ids(contract_copy())

    assert jobs == [f"lh-commerce-1-native-r{index:02d}" for index in range(1, 10)]


def test_boolean_candidate_count_cannot_alias_integer() -> None:
    contract = contract_copy()
    contract["batch_and_review_contract"]["candidate_count"] = True

    with pytest.raises(WORKFLOW.WorkflowBlocked, match="candidate count"):
        WORKFLOW.candidate_job_ids(contract)


def test_safe_zone_receipt_requires_rendered_measurements(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = valid_safe_zone_receipt(tmp_path, monkeypatch)
    now = safe_zone_time(receipt)
    WORKFLOW.validate_safe_zones(receipt, receipt["contract_sha256"], now)

    receipt["breakpoints"][0]["measurement_source"] = "inferred_from_css"
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="safe zones are inferred"):
        WORKFLOW.validate_safe_zones(receipt, receipt["contract_sha256"], now)


def test_safe_zone_receipt_is_contract_bound_and_type_strict(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = valid_safe_zone_receipt(tmp_path, monkeypatch)
    now = safe_zone_time(receipt)
    receipt["contract_sha256"] = "0" * 64
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="stale contract"):
        WORKFLOW.validate_safe_zones(receipt, "c" * 64, now)

    receipt = valid_safe_zone_receipt(tmp_path, monkeypatch)
    now = safe_zone_time(receipt)
    receipt["breakpoints"][0]["keep_clear"][0]["rect"][0] = False
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="outside normalized frame"):
        WORKFLOW.validate_safe_zones(receipt, receipt["contract_sha256"], now)


def test_output_parser_rejects_duplicate_job_ids() -> None:
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="invalid or duplicate"):
        WORKFLOW.parse_outputs(["job=/tmp/a.png", "job=/tmp/b.png"])


def test_verify_forwards_explicit_founder_override(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[str] = []

    def fake_run(command, **_kwargs):
        captured.extend(command)
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(WORKFLOW.subprocess, "run", fake_run)

    assert WORKFLOW.verify(Path("founder-override.json")) == 0
    assert captured[-2:] == ["--founder-override", "founder-override.json"]


def test_artifact_mapping_rejects_duplicate_paths_across_jobs() -> None:
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="paths are not unique"):
        WORKFLOW.validate_exact_artifact_mapping(
            {"job-a": Path("/tmp/same.png"), "job-b": Path("/tmp/same.png")},
            ["job-a", "job-b"],
            "outputs",
        )


def test_freshness_gate_rejects_future_and_expired_evidence() -> None:
    now = datetime.now(UTC)
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="future-dated"):
        WORKFLOW.validate_fresh_time(
            (now + timedelta(days=2)).isoformat(),
            "founder authorization",
            timedelta(hours=24),
            now,
        )
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="stale"):
        WORKFLOW.validate_fresh_time(
            (now - timedelta(hours=24, seconds=1)).isoformat(),
            "founder authorization",
            timedelta(hours=24),
            now,
        )


def test_judge_receipt_requires_exact_live_roster() -> None:
    now = datetime.now(UTC)
    receipt = {
        "schema": "skyyrose.image-judge-availability.v1",
        "status": "PASS_ALL_JUDGES_AVAILABLE",
        "checked_at": now.isoformat(),
        "all_judges_available": True,
        "secrets_in_receipt": False,
        "judges": [
            {
                "model": model,
                "available": True,
                "reason": "live_model_probe_passed",
                "configured_credentials_tried": 1,
            }
            for model in WORKFLOW.EXPECTED_JUDGES
        ],
    }
    WORKFLOW.validate_judge_receipt(receipt, now)
    receipt["judges"][0]["model"] = "different-model"
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="roster drifted"):
        WORKFLOW.validate_judge_receipt(receipt, now)


def test_safe_zone_receipt_rejects_changed_screenshot_hash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = valid_safe_zone_receipt(tmp_path, monkeypatch)
    now = safe_zone_time(receipt)
    receipt["breakpoints"][0]["screenshot_sha256"] = "0" * 64
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="screenshot changed"):
        WORKFLOW.validate_safe_zones(receipt, receipt["contract_sha256"], now)


def test_tournament_policy_link_cannot_redirect_to_another_repo_file() -> None:
    contract = contract_copy()
    policy = next(
        item
        for item in contract["bindings"]["validation_only_evidence"]
        if item["role"] == "mandatory_tournament_policy"
    )
    policy_path = WORKFLOW.root_path(policy["path"])
    release = {
        "tournament_policy": {
            "path": policy["path"],
            "sha256": WORKFLOW.sha256(policy_path),
        }
    }
    WORKFLOW.validate_tournament_policy_link(release, {"contract": contract})

    release["tournament_policy"] = {
        "path": str(CONTRACT.relative_to(WORKFLOW.ROOT)),
        "sha256": WORKFLOW.sha256(CONTRACT),
    }
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="policy path drifted"):
        WORKFLOW.validate_tournament_policy_link(release, {"contract": contract})


def test_png_inspection_fully_decodes_pixels(tmp_path: Path) -> None:
    output = tmp_path / "candidate.png"
    Image.new("RGB", (16, 9), "red").save(output)
    assert WORKFLOW.inspect_png(output, [16, 9]) == ([16, 9], "RGB")

    data = output.read_bytes()
    output.write_bytes(data[:-20])
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="failed full decode"):
        WORKFLOW.inspect_png(output, [16, 9])


def test_provider_job_identity_must_be_unique() -> None:
    seen: set[str] = set()
    assert WORKFLOW.add_unique_provider_job_id(" provider-1 ", seen) == "provider-1"
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="duplicate provider"):
        WORKFLOW.add_unique_provider_job_id("provider-1", seen)
    with pytest.raises(WORKFLOW.WorkflowBlocked, match="missing"):
        WORKFLOW.add_unique_provider_job_id("   ", seen)


@pytest.mark.parametrize("shim_directory", ["repo", "sibling", "local-bin", "tmp"])
def test_user_writable_shim_cannot_impersonate_provider_verifier(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, shim_directory: str
) -> None:
    monkeypatch.setattr(WORKFLOW, "ROOT", tmp_path)
    directory = tmp_path / shim_directory
    directory.mkdir()
    fake = directory / "codex-image-receipt-verify"
    fake.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    fake.chmod(0o755)
    monkeypatch.setattr(
        WORKFLOW,
        "TRUSTED_PROVIDER_VERIFIER_ALLOWLIST",
        {str(fake): WORKFLOW.sha256(fake)},
    )

    assert WORKFLOW.provider_verifier_path() is None


def test_self_declared_provider_receipt_outside_system_store_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(WORKFLOW, "ROOT", tmp_path)
    monkeypatch.setattr(WORKFLOW, "RELEASE_RECEIPT", tmp_path / "release.json")
    monkeypatch.setattr(WORKFLOW, "SYSTEM_GENERATED_IMAGE_ROOT", tmp_path / "system-generated")
    candidate_dir = tmp_path / "candidates"
    candidate_dir.mkdir()
    output = candidate_dir / "candidate.png"
    Image.new("RGB", (16, 9), "red").save(output)
    provider_id = "exec-11111111-1111-4111-8111-111111111111"
    now = datetime.now(UTC)
    release = {
        "generated_at": (now - timedelta(seconds=1)).isoformat(),
        "prompt_contract": {"sha256": "c" * 64},
    }
    WORKFLOW.RELEASE_RECEIPT.write_text(json.dumps(release), encoding="utf-8")
    raw_result = {
        "tool_call_id": provider_id,
        "generated_image_path": str(output),
        "output_hint": "candidate",
    }
    provider = {
        "schema": "skyyrose.image-tool-result.v1",
        "provider": "openai",
        "tool_name": "image_gen.imagegen",
        "tool_call_id": provider_id,
        "raw_result": raw_result,
        "raw_result_sha256": WORKFLOW.canonical_json_sha256(raw_result),
        "system_managed_source": {
            "path": str(output),
            "sha256": WORKFLOW.sha256(output),
            "bytes": output.stat().st_size,
        },
        "output": {
            "path": str(output.relative_to(tmp_path)),
            "sha256": WORKFLOW.sha256(output),
        },
    }
    provider_path = candidate_dir / "provider.json"
    provider_path.write_text(json.dumps(provider), encoding="utf-8")
    contract = {
        "output_contract": {"candidate_directory": "candidates"},
        "prompt_payload": {},
        "bindings": {"generator_conditioning_references": [{"sha256": "1" * 64}]},
    }
    sidecar = {
        "schema": "skyyrose.image-generation-job-evidence.v1",
        "job_id": "job-1",
        "scene_id": "LH-COMMERCE-1",
        "model_id": "gpt-image-2",
        "operation": "multi_reference_native_scene_regeneration",
        "release_receipt_sha256": WORKFLOW.sha256(WORKFLOW.RELEASE_RECEIPT),
        "prompt_contract_sha256": "c" * 64,
        "prompt_payload_sha256": WORKFLOW.canonical_json_sha256({}),
        "ordered_input_hashes": ["1" * 64],
        "provider_job_id": provider_id,
        "provider_result_receipt": {
            "path": str(provider_path.relative_to(tmp_path)),
            "sha256": WORKFLOW.sha256(provider_path),
        },
        "generated_at": now.isoformat(),
        "output": {
            "path": str(output.relative_to(tmp_path)),
            "sha256": WORKFLOW.sha256(output),
        },
    }
    sidecar_path = candidate_dir / "sidecar.json"
    sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

    with pytest.raises(WORKFLOW.WorkflowBlocked, match="system-managed Codex"):
        WORKFLOW.validate_generation_sidecar(sidecar_path, "job-1", output, release, contract)
