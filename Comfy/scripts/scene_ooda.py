#!/usr/bin/env python3
"""Fail-closed Higgsfield -> Comfy OODA runner for SkyyRose scene candidates."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
COMFY_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def execution_fingerprint(manifest: dict[str, Any]) -> str:
    """Hash the executable contract without the receipt path that authorizes it."""
    normalized = json.loads(json.dumps(manifest))
    normalized.setdefault("credit_control", {})["approval_receipt"] = None
    return sha256_text(json.dumps(normalized, sort_keys=True, separators=(",", ":")))


def resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_manifest(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "skyyrose.scene-ooda/1":
        raise ValueError("unsupported or missing scene OODA schema")
    for key in ("scene_id", "source_bindings", "higgsfield", "execute_blockers"):
        if key not in data:
            raise ValueError(f"manifest missing required key: {key}")
    return data


def verify_file(path: Path, expected_sha256: str | None) -> dict[str, Any]:
    result: dict[str, Any] = {"path": str(path), "exists": path.is_file()}
    if not result["exists"]:
        result["status"] = "MISSING"
        return result
    actual = sha256_file(path)
    result["actual_sha256"] = actual
    if expected_sha256:
        result["expected_sha256"] = expected_sha256
        result["status"] = "PASS" if actual == expected_sha256 else "HASH_MISMATCH"
    else:
        result["status"] = "PRESENT_UNHASHED"
    return result


def verify_team_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    team = manifest.get("fashion_theme_team", {})
    contract_path = resolve_path(team.get("team_contract", ""))
    result: dict[str, Any] = {
        "path": str(contract_path),
        "plugin": team.get("plugin"),
        "phase_id": team.get("phase_id"),
        "current_owner": team.get("current_owner"),
    }
    if not contract_path.is_file():
        result["status"] = "MISSING_TEAM_CONTRACT"
        return result
    index = json.loads(contract_path.read_text(encoding="utf-8"))
    indexed_scenes = {scene.get("scene_id") for scene in index.get("scenes", [])}
    required_roles = {
        "catalog-sot-integrator",
        "brand-experience-architect",
        "visual-commerce-qa",
        "theme-release-engineer",
    }
    serialized_ownership = json.dumps(index.get("phase_ownership", {}))
    missing_roles = sorted(role for role in required_roles if role not in serialized_ownership)
    failures = []
    if index.get("schema") != "skyyrose.scene-ooda-team-index/1":
        failures.append("invalid team-index schema")
    if team.get("plugin") != "fashion-theme-team@personal":
        failures.append("wrong or missing Fashion Theme Team plugin")
    if manifest["scene_id"] not in indexed_scenes:
        failures.append("scene missing from team index")
    if not team.get("phase_id") or not team.get("accountable") or not team.get("next_authority"):
        failures.append("incomplete durable phase assignment")
    if missing_roles:
        failures.append(f"team index missing required roles: {', '.join(missing_roles)}")
    result["failures"] = failures
    result["status"] = "PASS" if not failures else "INVALID_TEAM_CONTRACT"
    return result


def verify_creative_direction(manifest: dict[str, Any]) -> dict[str, Any]:
    direction = manifest.get("creative_direction")
    failures: list[str] = []
    if not isinstance(direction, dict):
        return {"status": "MISSING_CREATIVE_DIRECTION", "failures": ["missing creative_direction"]}
    if direction.get("style") != "cinematic_urban_grit":
        failures.append("style must be cinematic_urban_grit")
    for key in ("story_role", "logo_off_recognition"):
        if not isinstance(direction.get(key), str) or not direction[key].strip():
            failures.append(f"{key} must be a non-empty string")
    for key, minimum in (("focal_hierarchy", 3), ("material_language", 3), ("anti_generic", 3)):
        value = direction.get(key)
        if not isinstance(value, list) or len(value) < minimum:
            failures.append(f"{key} must contain at least {minimum} entries")
    return {
        "status": "PASS" if not failures else "INVALID_CREATIVE_DIRECTION",
        "failures": failures,
    }


def verify_optical_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    optical = manifest.get("optical_contract")
    failures: list[str] = []
    if not isinstance(optical, dict):
        return {"status": "MISSING_OPTICAL_CONTRACT", "failures": ["missing optical_contract"]}

    output = optical.get("output", {})
    width = output.get("width") if isinstance(output, dict) else None
    height = output.get("height") if isinstance(output, dict) else None
    if not isinstance(width, int) or not isinstance(height, int) or width <= 0 or height <= 0:
        failures.append("output width and height must be positive integers")
        width = height = 0

    horizon_y = optical.get("horizon_y")
    if not isinstance(horizon_y, int) or horizon_y < 0 or (height and horizon_y > height):
        failures.append("horizon_y must be inside the output geometry")

    subject_boxes = optical.get("subject_boxes")
    if not isinstance(subject_boxes, list) or not subject_boxes:
        failures.append("subject_boxes must contain at least one measured box")
    else:
        for index, box in enumerate(subject_boxes):
            if not isinstance(box, dict):
                failures.append(f"subject_boxes[{index}] must be an object")
                continue
            x, y, box_width, box_height = (box.get(key) for key in ("x", "y", "width", "height"))
            if not all(isinstance(value, int) for value in (x, y, box_width, box_height)):
                failures.append(f"subject_boxes[{index}] geometry must use integers")
                continue
            if x < 0 or y < 0 or box_width <= 0 or box_height <= 0:
                failures.append(f"subject_boxes[{index}] has invalid geometry")
            if width and height and (x + box_width > width or y + box_height > height):
                failures.append(f"subject_boxes[{index}] exceeds output geometry")

    contact_points = optical.get("contact_points")
    if not isinstance(contact_points, list) or not contact_points:
        failures.append("contact_points must contain at least one measured point")
    else:
        for index, point in enumerate(contact_points):
            if not isinstance(point, dict):
                failures.append(f"contact_points[{index}] must be an object")
                continue
            x, y = point.get("x"), point.get("y")
            if not isinstance(x, int) or not isinstance(y, int):
                failures.append(f"contact_points[{index}] coordinates must be integers")
            elif x < 0 or y < 0 or (width and x > width) or (height and y > height):
                failures.append(f"contact_points[{index}] exceeds output geometry")

    if optical.get("lens_class") not in {"wide", "normal", "telephoto"}:
        failures.append("lens_class must be wide, normal, or telephoto")
    if optical.get("camera_height") not in {"low", "eye_level", "high"}:
        failures.append("camera_height must be low, eye_level, or high")
    if optical.get("floor_response") not in {"dry_matte", "damp_satin", "wet_reflective"}:
        failures.append("floor_response must be dry_matte, damp_satin, or wet_reflective")
    if optical.get("reflection_mode") not in {"none", "subtle", "controlled"}:
        failures.append("reflection_mode must be none, subtle, or controlled")
    for key in ("key_light", "shadow_behavior"):
        if not isinstance(optical.get(key), dict) or not optical[key]:
            failures.append(f"{key} must be a non-empty object")
    requirements = optical.get("requirements")
    required_flags = ("contact", "occlusion", "depth", "grain_match", "edge_spill_match")
    if not isinstance(requirements, dict) or any(
        requirements.get(flag) is not True for flag in required_flags
    ):
        failures.append("all native-integration requirements must be true")

    return {
        "status": "PASS" if not failures else "INVALID_OPTICAL_CONTRACT",
        "failures": failures,
    }


def verify_comfy_contract(manifest: dict[str, Any]) -> dict[str, Any]:
    comfy = manifest.get("comfy")
    failures: list[str] = []
    if not isinstance(comfy, dict):
        return {"status": "MISSING_COMFY_CONTRACT", "failures": ["comfy contract must not be null"]}
    if comfy.get("routing") != "local" or comfy.get("server") != "127.0.0.1:8188":
        failures.append("Comfy must route only to local 127.0.0.1:8188")
    if comfy.get("role") != "POST_CAPTURE_PROTECTED_CORRECTION_ONLY":
        failures.append("Comfy role must be POST_CAPTURE_PROTECTED_CORRECTION_ONLY")
    for key in ("allowed_operations", "forbidden_operations"):
        if not isinstance(comfy.get(key), list) or not comfy[key]:
            failures.append(f"{key} must be a non-empty list")
    for key in (
        "requires_candidate_hash",
        "requires_mask_for_product_pixels",
        "requires_independent_review",
    ):
        if comfy.get(key) is not True:
            failures.append(f"{key} must be true")
    return {
        "status": "PASS" if not failures else "INVALID_COMFY_CONTRACT",
        "failures": failures,
    }


def verify_authorization_receipt(
    manifest: dict[str, Any], raw_path: str | None, expected_action: str
) -> dict[str, Any]:
    if not raw_path:
        return {"status": "MISSING_RECEIPT", "expected_action": expected_action}
    path = resolve_path(raw_path)
    if not path.is_file():
        return {
            "status": "MISSING_RECEIPT_FILE",
            "path": str(path),
            "expected_action": expected_action,
        }
    try:
        receipt = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return {"status": "INVALID_RECEIPT", "path": str(path), "error": str(exc)}
    failures: list[str] = []
    if receipt.get("schema") != "skyyrose.scene-ooda-receipt/1":
        failures.append("invalid receipt schema")
    if receipt.get("scene_id") != manifest.get("scene_id"):
        failures.append("receipt scene does not match manifest")
    if receipt.get("action") != expected_action:
        failures.append("receipt action does not match required action")
    payload = receipt.get("payload", {})
    if expected_action == "higgsfield-enhance-review":
        if payload.get("inspection", {}).get("status") != "PASS":
            failures.append("enhanced prompt review did not pass")
        if payload.get("generation_submitted") is not False:
            failures.append("prompt-review receipt must record generation_submitted=false")
        expected_prompt_sha256 = manifest.get("credit_control", {}).get("prompt_sha256")
        if expected_prompt_sha256 and payload.get("prompt_sha256") != expected_prompt_sha256:
            failures.append("prompt-review receipt is stale for the current prompt")
    if expected_action == "paid-generation-approval":
        if payload.get("allow_paid_attempts") != 1:
            failures.append("approval must authorize exactly one paid attempt")
        if payload.get("candidate_bound") is not True:
            failures.append("approval must be candidate-bound")
        if (
            not isinstance(payload.get("planned_candidate_id"), str)
            or not payload["planned_candidate_id"].strip()
        ):
            failures.append("approval must name a planned candidate id")
        if payload.get("manifest_sha256") != execution_fingerprint(manifest):
            failures.append("approval is not bound to the current executable manifest")
    return {
        "status": "PASS" if not failures else "INVALID_RECEIPT",
        "path": str(path),
        "failures": failures,
    }


def verify_credit_control(manifest: dict[str, Any]) -> dict[str, Any]:
    control = manifest.get("credit_control")
    failures: list[str] = []
    if not isinstance(control, dict):
        return {
            "status": "MISSING_CREDIT_CONTROL",
            "failures": ["missing credit_control"],
            "prompt_review": {"status": "MISSING_RECEIPT"},
            "paid_approval": {"status": "MISSING_RECEIPT"},
            "attempt_available": False,
        }
    if control.get("mode") != "ONE_CANDIDATE_ONE_PAID_ATTEMPT":
        failures.append("mode must be ONE_CANDIDATE_ONE_PAID_ATTEMPT")
    if control.get("max_paid_generations") != 1:
        failures.append("max_paid_generations must equal 1")
    recorded = control.get("paid_generations_recorded")
    if not isinstance(recorded, int) or recorded < 0:
        failures.append("paid_generations_recorded must be a non-negative integer")
    if control.get("automatic_paid_retries") is not False:
        failures.append("automatic_paid_retries must be false")
    for key in (
        "require_current_observe_pass",
        "require_prompt_review_pass",
        "require_candidate_bound_approval",
    ):
        if control.get(key) is not True:
            failures.append(f"{key} must be true")
    if control.get("rejected_candidate_may_be_next_input") is not False:
        failures.append("rejected_candidate_may_be_next_input must be false")
    if control.get("on_failure") != "OPEN_ISSUE_AND_STOP":
        failures.append("on_failure must be OPEN_ISSUE_AND_STOP")
    if manifest.get("higgsfield", {}).get("count") != 1:
        failures.append("Higgsfield count must equal 1")
    current_prompt_sha256 = sha256_text(manifest.get("higgsfield", {}).get("prompt", ""))
    if control.get("prompt_sha256") != current_prompt_sha256:
        failures.append("prompt_sha256 does not match the current prompt")

    prompt_review = verify_authorization_receipt(
        manifest, control.get("prompt_review_receipt"), "higgsfield-enhance-review"
    )
    paid_approval = verify_authorization_receipt(
        manifest, control.get("approval_receipt"), "paid-generation-approval"
    )
    return {
        "status": "PASS" if not failures else "INVALID_CREDIT_CONTROL",
        "failures": failures,
        "prompt_review": prompt_review,
        "paid_approval": paid_approval,
        "attempt_available": isinstance(recorded, int)
        and recorded < control.get("max_paid_generations", 0),
    }


def observe(manifest: dict[str, Any]) -> dict[str, Any]:
    catalog = manifest["catalog_snapshot"]
    catalog_checks = [
        verify_file(resolve_path(catalog["catalog_path"]), catalog["catalog_sha256"]),
        verify_file(
            resolve_path(catalog["image_manifest_path"]),
            catalog["image_manifest_sha256"],
        ),
    ]
    source_checks = [
        verify_file(resolve_path(binding["path"]), binding["sha256"])
        for binding in manifest["source_bindings"]
    ]
    dependency_checks = [
        verify_file(resolve_path(dependency["path"]), dependency.get("expected_sha256"))
        for dependency in manifest.get("authority_dependencies", [])
    ]
    forbidden_input_checks = [
        verify_file(resolve_path(binding["path"]), binding.get("sha256"))
        for binding in manifest.get("forbidden_inputs", [])
    ]
    forbidden_paths = {
        str(resolve_path(binding["path"])) for binding in manifest.get("forbidden_inputs", [])
    }
    routed_paths = {
        str(resolve_path(binding["path"]))
        for binding in manifest["source_bindings"]
        if binding.get("send_to_higgsfield")
    }
    forbidden_routing_clean = not forbidden_paths.intersection(routed_paths)
    team_contract_check = verify_team_contract(manifest)
    creative_direction_check = verify_creative_direction(manifest)
    optical_contract_check = verify_optical_contract(manifest)
    comfy_contract_check = verify_comfy_contract(manifest)
    credit_control_check = verify_credit_control(manifest)
    source_ready = all(check["status"] == "PASS" for check in catalog_checks + source_checks)
    dependencies_ready = all(check["status"] == "PASS" for check in dependency_checks)
    configuration_ready = (
        source_ready
        and forbidden_routing_clean
        and team_contract_check["status"] == "PASS"
        and creative_direction_check["status"] == "PASS"
        and optical_contract_check["status"] == "PASS"
        and comfy_contract_check["status"] == "PASS"
        and credit_control_check["status"] == "PASS"
    )
    paid_authorized = (
        credit_control_check["prompt_review"]["status"] == "PASS"
        and credit_control_check["paid_approval"]["status"] == "PASS"
        and credit_control_check["attempt_available"]
    )
    paid_route_permitted = (
        manifest.get("execution_strategy", {}).get("full_frame_paid_generation", True) is not False
    )
    return {
        "scene_id": manifest["scene_id"],
        "catalog_checks": catalog_checks,
        "source_checks": source_checks,
        "dependency_checks": dependency_checks,
        "forbidden_input_checks": forbidden_input_checks,
        "forbidden_routing_clean": forbidden_routing_clean,
        "team_contract_check": team_contract_check,
        "creative_direction_check": creative_direction_check,
        "optical_contract_check": optical_contract_check,
        "comfy_contract_check": comfy_contract_check,
        "credit_control_check": credit_control_check,
        "source_ready": source_ready,
        "dependencies_ready": dependencies_ready,
        "configuration_ready": configuration_ready,
        "paid_authorized": paid_authorized,
        "paid_route_permitted": paid_route_permitted,
        "execute_blockers": manifest["execute_blockers"],
        "paid_execution_ready": configuration_ready
        and dependencies_ready
        and not manifest["execute_blockers"]
        and paid_authorized
        and paid_route_permitted,
    }


def higgsfield_command(manifest: dict[str, Any], enhance_only: bool) -> list[str]:
    config = manifest["higgsfield"]
    command = [
        "higgsfield",
        "--json",
        "product-photoshoot",
        "create",
        "--mode",
        config["mode"],
        "--prompt",
        config["prompt"],
        "--aspect_ratio",
        config["aspect_ratio"],
        "--count",
        str(config["count"]),
        "--brand_context",
        config["brand_context"],
        "--product_context",
        config["product_context"],
    ]
    for binding in manifest["source_bindings"]:
        if binding.get("send_to_higgsfield"):
            command.extend(["--image", str(resolve_path(binding["path"]))])
    if enhance_only:
        command.append("--enhance-only")
    return command


def run_command(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    parsed: Any = None
    if completed.stdout.strip():
        try:
            parsed = json.loads(completed.stdout)
        except json.JSONDecodeError:
            parsed = completed.stdout.strip()
    return {
        "command": command,
        "exit_code": completed.returncode,
        "stdout": parsed,
        "stderr": completed.stderr.strip(),
    }


def inspect_enhanced_prompt(manifest: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    stdout = result.get("stdout")
    prompts = stdout.get("prompts", []) if isinstance(stdout, dict) else []
    prompt_text = "\n".join(item.get("prompt", "") for item in prompts if isinstance(item, dict))
    assertions = manifest["higgsfield"].get("prompt_assertions", {})
    lowered = prompt_text.casefold()
    missing_required = [
        phrase
        for phrase in assertions.get("required_phrases", [])
        if phrase.casefold() not in lowered
    ]
    present_forbidden = [
        phrase for phrase in assertions.get("forbidden_phrases", []) if phrase.casefold() in lowered
    ]
    passed = bool(prompt_text) and not missing_required and not present_forbidden
    return {
        "status": "PASS" if passed else "REJECTED_ENHANCER_PRODUCT_DRIFT",
        "prompt_present": bool(prompt_text),
        "missing_required_phrases": missing_required,
        "present_forbidden_phrases": present_forbidden,
    }


def write_receipt(scene_id: str, action: str, payload: dict[str, Any]) -> Path:
    receipt_dir = COMFY_ROOT / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    recorded_at = datetime.now(UTC)
    stamp = recorded_at.strftime("%Y%m%dT%H%M%S%fZ")
    receipt_path = receipt_dir / f"{scene_id.lower()}-{action}-{stamp}.json"
    document = {
        "schema": "skyyrose.scene-ooda-receipt/1",
        "scene_id": scene_id,
        "action": action,
        "recorded_at": recorded_at.isoformat(),
        "payload": payload,
    }
    receipt_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return receipt_path


def require_cli(name: str) -> None:
    if shutil.which(name) is None:
        raise RuntimeError(f"required CLI not found on PATH: {name}")


def action_validate(manifest: dict[str, Any]) -> int:
    report = observe(manifest)
    report["higgsfield_command_preview"] = higgsfield_command(manifest, True)
    print(json.dumps(report, indent=2))
    return 0 if report["configuration_ready"] else 2


def action_enhance(manifest: dict[str, Any]) -> int:
    require_cli("higgsfield")
    report = observe(manifest)
    if not report["configuration_ready"]:
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 2
    result = run_command(higgsfield_command(manifest, True))
    inspection = inspect_enhanced_prompt(manifest, result)
    receipt = write_receipt(
        manifest["scene_id"],
        "higgsfield-enhance",
        {"observe": report, "result": result, "inspection": inspection},
    )
    print(
        json.dumps(
            {"receipt": str(receipt), "inspection": inspection, "result": result},
            indent=2,
        )
    )
    if result["exit_code"] != 0:
        return result["exit_code"]
    return 0 if inspection["status"] == "PASS" else 4


def action_execute(manifest: dict[str, Any], allow_spend: bool) -> int:
    if not allow_spend:
        raise RuntimeError("paid generation requires the explicit --allow-spend flag")
    require_cli("higgsfield")
    report = observe(manifest)
    if not report["paid_execution_ready"]:
        print(json.dumps(report, indent=2), file=sys.stderr)
        return 3
    result = run_command(higgsfield_command(manifest, False))
    receipt = write_receipt(
        manifest["scene_id"], "higgsfield-generate", {"observe": report, "result": result}
    )
    print(json.dumps({"receipt": str(receipt), "result": result}, indent=2))
    return 0 if result["exit_code"] == 0 else result["exit_code"]


def action_review_enhance(manifest: dict[str, Any], receipt_path: Path | None) -> int:
    if receipt_path is None:
        raise RuntimeError("enhancement review requires --receipt /absolute/path/to/receipt.json")
    receipt_path = receipt_path.expanduser().resolve()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    result = receipt.get("payload", {}).get("result")
    if not isinstance(result, dict):
        raise ValueError("receipt does not contain a Higgsfield result payload")
    inspection = inspect_enhanced_prompt(manifest, result)
    review_receipt = write_receipt(
        manifest["scene_id"],
        "higgsfield-enhance-review",
        {
            "reviewed_receipt": str(receipt_path),
            "inspection": inspection,
            "generation_submitted": False,
            "prompt_sha256": sha256_text(manifest["higgsfield"]["prompt"]),
        },
    )
    print(json.dumps({"receipt": str(review_receipt), "inspection": inspection}, indent=2))
    return 0 if inspection["status"] == "PASS" else 4


def action_stage(manifest: dict[str, Any], candidate: Path | None) -> int:
    if candidate is None:
        raise RuntimeError("staging requires --candidate /absolute/path/to/image")
    candidate = candidate.expanduser().resolve()
    check = verify_file(candidate, None)
    if not check["exists"]:
        print(json.dumps(check, indent=2), file=sys.stderr)
        return 2
    require_cli("comfy")
    result = run_command(["comfy", "--where", "local", "upload", str(candidate)])
    receipt = write_receipt(
        manifest["scene_id"],
        "comfy-stage-candidate",
        {
            "candidate": check,
            "candidate_status": "CANDIDATE_ONLY_PENDING_INDEPENDENT_REVIEW",
            "result": result,
        },
    )
    print(json.dumps({"receipt": str(receipt), "result": result}, indent=2))
    return 0 if result["exit_code"] == 0 else result["exit_code"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "action",
        choices=("validate", "enhance", "review-enhance", "execute", "stage"),
    )
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--allow-spend", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--receipt", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        manifest = load_manifest(args.manifest.resolve())
        if args.action == "validate":
            return action_validate(manifest)
        if args.action == "enhance":
            return action_enhance(manifest)
        if args.action == "review-enhance":
            return action_review_enhance(manifest, args.receipt)
        if args.action == "execute":
            return action_execute(manifest, args.allow_spend)
        return action_stage(manifest, args.candidate)
    except (OSError, RuntimeError, ValueError) as exc:
        print(json.dumps({"error": str(exc)}), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
