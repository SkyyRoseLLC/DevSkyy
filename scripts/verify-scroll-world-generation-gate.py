#!/usr/bin/env python3
"""Fail-closed verification for paid Scroll World scene submissions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CRITERIA = ROOT / "data/scroll-world-one-shot-verification-criteria.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_scene(scene_id: str, scene: dict[str, Any], global_provider_allowed: bool) -> dict[str, Any]:
    failures: list[str] = []
    checks: list[dict[str, Any]] = []

    assignments = scene.get("wearer_assignments", [])
    wearers = {entry.get("wearer") for entry in assignments}
    if not {"man", "woman"}.issubset(wearers):
        failures.append("DUAL_CAST_WEARER_ASSIGNMENT_MISSING")

    for item in scene.get("required_inputs", []):
        relative = item.get("path", "")
        expected = item.get("sha256", "")
        candidate = ROOT / relative
        result: dict[str, Any] = {
            "role": item.get("role"),
            "path": relative,
            "expected_sha256": expected,
        }
        if not candidate.is_file():
            result["status"] = "MISSING"
            failures.append(f"MISSING_REQUIRED_INPUT:{relative}")
        else:
            actual = sha256(candidate)
            result["actual_sha256"] = actual
            result["status"] = "PASS" if actual == expected else "HASH_MISMATCH"
            if actual != expected:
                failures.append(f"HASH_MISMATCH:{relative}")
        checks.append(result)

    required_nonempty = ("product_locks", "scene_locks", "hard_rejects")
    for key in required_nonempty:
        if not scene.get(key):
            failures.append(f"EMPTY_{key.upper()}")

    forbidden = set(scene.get("forbidden_inputs", []))
    required_paths = {item.get("path") for item in scene.get("required_inputs", [])}
    overlap = sorted(path for path in forbidden.intersection(required_paths) if path)
    if overlap:
        failures.append("FORBIDDEN_INPUT_DECLARED_AS_REQUIRED:" + ",".join(overlap))

    source_ready = not failures
    submission_control = scene.get("provider_submission_control", {})
    scene_provider_allowed = submission_control.get("allowed", True)
    provider_ready = source_ready and global_provider_allowed and scene_provider_allowed
    provider_blockers = list(submission_control.get("blocking_items", []))
    if not global_provider_allowed:
        provider_blockers.insert(0, "GLOBAL_PAID_PROVIDER_PAUSE_ACTIVE")

    return {
        "scene_id": scene_id,
        "status": (
            "PASS_SOURCE_PREFLIGHT__READY_FOR_PROVIDER"
            if provider_ready
            else "PASS_SOURCE_PREFLIGHT__BLOCKED_PROVIDER_SUBMISSION"
            if source_ready
            else "BLOCKED_SOURCE_PREFLIGHT"
        ),
        "checks": checks,
        "failures": failures,
        "source_preflight_passed": source_ready,
        "provider_submission_allowed": provider_ready,
        "provider_blockers": provider_blockers,
        "runtime_wiring_allowed": False,
        "deployment_allowed": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--criteria", type=Path, default=DEFAULT_CRITERIA)
    parser.add_argument("--scene", action="append", dest="scenes")
    parser.add_argument("--write-receipt", type=Path)
    args = parser.parse_args()

    criteria_path = args.criteria if args.criteria.is_absolute() else ROOT / args.criteria
    criteria = json.loads(criteria_path.read_text())
    sot_lock = criteria["source_freshness_lock"]
    sot_path = ROOT / sot_lock["path"]
    global_failures: list[str] = []
    if not sot_path.is_file():
        global_failures.append("ROOT_PRODUCT_SOT_MISSING")
    elif sha256(sot_path) != sot_lock["sha256"]:
        global_failures.append("ROOT_PRODUCT_SOT_HASH_MISMATCH")

    requested = args.scenes or list(criteria["scene_packs"])
    unknown = sorted(set(requested).difference(criteria["scene_packs"]))
    if unknown:
        global_failures.extend(f"UNKNOWN_SCENE:{scene_id}" for scene_id in unknown)

    global_provider_allowed = criteria.get("provider_policy", {}).get("provider_submission_allowed", False)
    results = [
        validate_scene(scene_id, criteria["scene_packs"][scene_id], global_provider_allowed)
        for scene_id in requested
        if scene_id in criteria["scene_packs"]
    ]
    if global_failures:
        for result in results:
            result["failures"] = global_failures + result["failures"]
            result["status"] = "BLOCKED_SOURCE_PREFLIGHT"
            result["source_preflight_passed"] = False
            result["provider_submission_allowed"] = False

    receipt = {
        "schema": "skyyrose.scroll-world-generation-gate-receipt.v1",
        "criteria_path": str(criteria_path.relative_to(ROOT)),
        "criteria_sha256": sha256(criteria_path),
        "source_sot_sha256": sha256(sot_path) if sot_path.is_file() else None,
        "status": (
            "READY_FOR_PROVIDER_SUBMISSION"
            if results and all(r["provider_submission_allowed"] for r in results)
            else "PASS_SOURCE_PREFLIGHT__BLOCKED_PROVIDER_SUBMISSION"
            if results and all(r["source_preflight_passed"] for r in results)
            else "BLOCKED_SOURCE_PREFLIGHT"
        ),
        "global_failures": global_failures,
        "scenes": results,
        "boundary": {
            "source_preflight_is_not_product_postflight": True,
            "source_preflight_is_not_founder_promotion": True,
            "source_preflight_is_not_provider_submission_authorization": True,
            "runtime_wiring_allowed": False,
            "deployment_allowed": False,
        },
    }
    rendered = json.dumps(receipt, indent=2) + "\n"
    if args.write_receipt:
        output = args.write_receipt if args.write_receipt.is_absolute() else ROOT / args.write_receipt
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")
    return 0 if receipt["status"] == "READY_FOR_PROVIDER_SUBMISSION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
