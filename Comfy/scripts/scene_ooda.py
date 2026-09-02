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
    source_ready = all(check["status"] == "PASS" for check in catalog_checks + source_checks)
    dependencies_ready = all(check["status"] == "PASS" for check in dependency_checks)
    configuration_ready = (
        source_ready and forbidden_routing_clean and team_contract_check["status"] == "PASS"
    )
    return {
        "scene_id": manifest["scene_id"],
        "catalog_checks": catalog_checks,
        "source_checks": source_checks,
        "dependency_checks": dependency_checks,
        "forbidden_input_checks": forbidden_input_checks,
        "forbidden_routing_clean": forbidden_routing_clean,
        "team_contract_check": team_contract_check,
        "source_ready": source_ready,
        "dependencies_ready": dependencies_ready,
        "configuration_ready": configuration_ready,
        "execute_blockers": manifest["execute_blockers"],
        "paid_execution_ready": configuration_ready
        and dependencies_ready
        and not manifest["execute_blockers"],
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
