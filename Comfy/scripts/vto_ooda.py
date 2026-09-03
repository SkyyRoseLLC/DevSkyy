#!/usr/bin/env python3
"""Fail-closed FASHN Try-On Max candidate runner for SkyyRose scenes."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skyyrose.integrations.fashn_client import (  # noqa: E402
    TRYON_MAX_CREDITS,
    TRYON_MAX_LIFECYCLE,
    TRYON_MAX_MODEL,
    FashnClient,
)

CONTRACT_SCHEMA = "skyyrose.vto-ooda/1"
APPROVAL_SCHEMA = "skyyrose.vto-paid-approval/1"
RECEIPT_SCHEMA = "skyyrose.vto-execution-receipt/1"
REVIEW_SCHEMA = "skyyrose.vto-independent-review/1"
APPROVED_MODEL_AUTHORITY = "FOUNDER_APPROVED_FULL_BODY_MODEL"
MAX_INPUT_BYTES = 30 * 1024 * 1024
IMAGE_MIME_TYPES = {
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def execution_fingerprint(contract: dict[str, Any]) -> str:
    """Hash executable intent without the self-referential approval path."""
    normalized = deepcopy(contract)
    normalized.setdefault("credit_control", {})["approval_receipt"] = None
    return sha256_text(json.dumps(normalized, sort_keys=True, separators=(",", ":")))


def resolve_path(raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("path must be a non-empty string")
    candidate = Path(raw_path).expanduser()
    resolved = (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"path escapes project root: {raw_path}")
    return resolved


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"unsupported or missing VTO contract schema: {contract.get('schema')}")
    for key in (
        "pilot_id",
        "scene_id",
        "sku",
        "model",
        "inputs",
        "request",
        "product_invariants",
        "credit_control",
        "output",
    ):
        if key not in contract:
            raise ValueError(f"contract missing required key: {key}")
    return contract


def verify_file(binding: dict[str, Any], *, label: str) -> dict[str, Any]:
    raw_path = binding.get("path")
    expected = binding.get("sha256")
    result: dict[str, Any] = {"label": label, "path": raw_path, "status": "BLOCKED"}
    if not raw_path:
        result["code"] = "MISSING_PATH"
        return result
    try:
        path = resolve_path(raw_path)
    except ValueError as exc:
        result["code"] = "INVALID_PATH"
        result["message"] = str(exc)
        return result
    result["path"] = str(path)
    if not path.is_file():
        result["code"] = "MISSING_FILE"
        return result
    actual = sha256_file(path)
    result["actual_sha256"] = actual
    result["expected_sha256"] = expected
    if not isinstance(expected, str) or actual != expected:
        result["code"] = "HASH_MISMATCH"
        return result
    result["status"] = "PASS"
    result["code"] = "HASH_VERIFIED"
    return result


def verify_authority_receipt(binding: dict[str, Any], *, label: str) -> dict[str, Any]:
    receipt = binding.get("authority_receipt")
    if not isinstance(receipt, dict):
        return {"label": label, "status": "BLOCKED", "code": "MISSING_AUTHORITY_RECEIPT"}
    return verify_file(receipt, label=f"{label}.authority_receipt")


def verify_registry(contract: dict[str, Any]) -> dict[str, Any]:
    model = contract["model"]
    result: dict[str, Any] = {"status": "BLOCKED", "code": "INVALID_MODEL_REGISTRY"}
    if model.get("id") != TRYON_MAX_MODEL or model.get("lifecycle") != TRYON_MAX_LIFECYCLE:
        result["code"] = "MODEL_OR_LIFECYCLE_MISMATCH"
        return result
    registry_binding = model.get("registry")
    if not isinstance(registry_binding, dict):
        result["code"] = "MISSING_MODEL_REGISTRY"
        return result
    registry_file = verify_file(registry_binding, label="model.registry")
    if registry_file["status"] != "PASS":
        return registry_file
    registry = json.loads(Path(registry_file["path"]).read_text(encoding="utf-8"))
    for entry in registry.get("models", []):
        names = {entry.get("id"), *entry.get("aliases", [])}
        if TRYON_MAX_MODEL in names:
            if "virtual_tryon_candidate" not in entry.get("allowed_operations", []):
                result["code"] = "VTO_OPERATION_NOT_ALLOWED"
                return result
            return {
                "status": "PASS",
                "code": "REGISTERED_CANDIDATE_ONLY_MODEL",
                "model": entry.get("id"),
                "fidelity_tier": entry.get("fidelity_tier"),
                "live_availability_verified": False,
            }
    result["code"] = "UNKNOWN_MODEL"
    return result


def verify_request(contract: dict[str, Any]) -> dict[str, Any]:
    request = contract["request"]
    failures: list[str] = []
    resolution = request.get("resolution")
    generation_mode = request.get("generation_mode")
    if resolution not in {"1k", "2k", "4k"}:
        failures.append("resolution must be 1k, 2k, or 4k")
    if generation_mode not in {"fast", "balanced", "quality"}:
        failures.append("generation_mode must be fast, balanced, or quality")
    seed = request.get("seed")
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
        failures.append("seed must be an integer from 0 to 2^32 - 1")
    if request.get("num_images") != 1:
        failures.append("pilot num_images must equal 1")
    if request.get("output_format") != "png":
        failures.append("pilot output_format must equal png")
    if request.get("return_base64") is not True:
        failures.append("pilot return_base64 must be true")
    if not isinstance(request.get("prompt"), str) or not request["prompt"].strip():
        failures.append("prompt must be a non-empty string")
    invariants = contract.get("product_invariants")
    if not isinstance(invariants, list) or len(invariants) < 4:
        failures.append("product_invariants must contain at least four checks")

    estimated_credits = None
    if not failures:
        estimated_credits = TRYON_MAX_CREDITS[(generation_mode, resolution)]
    return {
        "status": "PASS" if not failures else "BLOCKED",
        "code": "REQUEST_CONTRACT_VERIFIED" if not failures else "INVALID_REQUEST_CONTRACT",
        "estimated_credits": estimated_credits,
        "failures": failures,
    }


def verify_approval(contract: dict[str, Any], estimated_credits: int | None) -> dict[str, Any]:
    control = contract["credit_control"]
    failures: list[str] = []
    if control.get("max_paid_generations") != 1:
        failures.append("max_paid_generations must equal 1")
    if control.get("paid_generations_recorded") != 0:
        failures.append("paid_generations_recorded must equal 0 before the pilot")
    if control.get("automatic_paid_retries") is not False:
        failures.append("automatic_paid_retries must be false")
    max_credits = control.get("max_credits")
    if estimated_credits is not None and (
        isinstance(max_credits, bool)
        or not isinstance(max_credits, int)
        or max_credits < estimated_credits
    ):
        failures.append("max_credits is below the contracted request cost")
    if failures:
        return {"status": "BLOCKED", "code": "INVALID_CREDIT_CONTROL", "failures": failures}

    receipt_binding = control.get("approval_receipt")
    if not isinstance(receipt_binding, dict):
        return {
            "status": "BLOCKED",
            "code": "MISSING_PAID_APPROVAL_RECEIPT",
            "failures": ["a contract-bound approval receipt is required before execution"],
        }
    receipt_file = verify_file(receipt_binding, label="credit_control.approval_receipt")
    if receipt_file["status"] != "PASS":
        return receipt_file
    receipt = json.loads(Path(receipt_file["path"]).read_text(encoding="utf-8"))
    expected_fingerprint = execution_fingerprint(contract)
    approved_by = receipt.get("approved_by")
    approved_at = receipt.get("approved_at")
    if (
        receipt.get("schema") != APPROVAL_SCHEMA
        or receipt.get("pilot_id") != contract["pilot_id"]
        or receipt.get("execution_fingerprint") != expected_fingerprint
        or receipt.get("max_credits") != max_credits
        or receipt.get("approved") is not True
        or not isinstance(approved_by, str)
        or not approved_by.strip()
        or not isinstance(approved_at, str)
        or not approved_at.strip()
    ):
        return {
            "status": "BLOCKED",
            "code": "INVALID_PAID_APPROVAL_RECEIPT",
            "failures": ["approval receipt does not bind the current executable contract"],
        }
    return {"status": "PASS", "code": "PAID_ATTEMPT_APPROVED", "max_credits": max_credits}


def validate(contract_path: Path) -> dict[str, Any]:
    contract = load_contract(contract_path)
    inputs = contract["inputs"]
    product = inputs.get("product") if isinstance(inputs, dict) else None
    model = inputs.get("model") if isinstance(inputs, dict) else None
    supporting = inputs.get("supporting_product_views", []) if isinstance(inputs, dict) else []

    checks: dict[str, Any] = {
        "registry": verify_registry(contract),
        "request": verify_request(contract),
    }
    if not isinstance(product, dict):
        checks["product"] = {"status": "BLOCKED", "code": "MISSING_PRODUCT_INPUT"}
        checks["product_authority"] = {
            "status": "BLOCKED",
            "code": "MISSING_PRODUCT_AUTHORITY",
        }
    else:
        checks["product"] = verify_file(product, label="inputs.product")
        checks["product_authority"] = verify_authority_receipt(product, label="inputs.product")

    support_checks: list[dict[str, Any]] = []
    if not isinstance(supporting, list) or len(supporting) < 4:
        support_checks.append(
            {"status": "BLOCKED", "code": "FOUR_DIRECTIONAL_PRODUCT_VIEWS_REQUIRED"}
        )
    else:
        observed_views: list[str] = []
        for index, binding in enumerate(supporting):
            if not isinstance(binding, dict):
                support_checks.append(
                    {"status": "BLOCKED", "code": "INVALID_SUPPORTING_VIEW", "index": index}
                )
                continue
            observed_views.append(str(binding.get("view", "")))
            support_checks.append(verify_file(binding, label=f"supporting_product_views[{index}]"))
            support_checks.append(
                verify_authority_receipt(binding, label=f"supporting_product_views[{index}]")
            )
        required_views = {"front", "back", "wearer_left", "wearer_right"}
        if set(observed_views) != required_views or len(observed_views) != len(required_views):
            support_checks.append(
                {
                    "status": "BLOCKED",
                    "code": "INVALID_DIRECTIONAL_PRODUCT_VIEW_SET",
                    "required": sorted(required_views),
                    "observed": observed_views,
                }
            )
    checks["supporting_product_views"] = support_checks

    if not isinstance(model, dict):
        checks["model"] = {"status": "BLOCKED", "code": "MISSING_MODEL_INPUT"}
        checks["model_authority"] = {"status": "BLOCKED", "code": "MISSING_MODEL_AUTHORITY"}
    else:
        checks["model"] = verify_file(model, label="inputs.model")
        checks["model_authority"] = verify_authority_receipt(model, label="inputs.model")
        if model.get("authority_state") != APPROVED_MODEL_AUTHORITY:
            checks["model_authority"] = {
                "status": "BLOCKED",
                "code": "FULL_BODY_MODEL_NOT_APPROVED",
                "observed": model.get("authority_state"),
                "required": APPROVED_MODEL_AUTHORITY,
            }

    estimated_credits = checks["request"].get("estimated_credits")
    checks["approval"] = verify_approval(contract, estimated_credits)

    output = contract["output"]
    try:
        output_path = resolve_path(output.get("candidate_path"))
        receipt_path = resolve_path(output.get("execution_receipt"))
        output_check: dict[str, Any] = {
            "status": "PASS",
            "code": "OUTPUT_PATHS_RESERVED",
            "candidate_path": str(output_path),
            "execution_receipt": str(receipt_path),
        }
        if output_path.exists() or receipt_path.exists():
            output_check = {
                "status": "BLOCKED",
                "code": "OUTPUT_OR_RECEIPT_ALREADY_EXISTS",
            }
    except (TypeError, ValueError) as exc:
        output_check = {"status": "BLOCKED", "code": "INVALID_OUTPUT_PATH", "message": str(exc)}
    checks["output"] = output_check

    flat_checks = [value for value in checks.values() if isinstance(value, dict)]
    flat_checks.extend(support_checks)
    blockers = [
        check.get("code", "UNKNOWN") for check in flat_checks if check.get("status") != "PASS"
    ]
    return {
        "schema": "skyyrose.vto-preflight/1",
        "pilot_id": contract["pilot_id"],
        "scene_id": contract["scene_id"],
        "sku": contract["sku"],
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "execution_fingerprint": execution_fingerprint(contract),
        "checks": checks,
        "estimated_credits": estimated_credits,
        "execution_ready": not blockers,
        "status": "PASS" if not blockers else "BLOCKED",
        "blockers": blockers,
    }


def image_data_uri(path: Path) -> str:
    mime = IMAGE_MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"unsupported input image format: {path.suffix}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_INPUT_BYTES:
        raise ValueError(f"input image must be between 1 and {MAX_INPUT_BYTES} bytes")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def decode_png_data_uri(value: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not value.startswith(prefix):
        raise ValueError("Try-On Max output is not a PNG data URI")
    payload = base64.b64decode(value[len(prefix) :], validate=True)
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("decoded Try-On Max output is not a PNG")
    return payload


async def execute(contract_path: Path, *, allow_spend: bool) -> dict[str, Any]:
    if not allow_spend:
        raise ValueError("execution requires --allow-spend")
    preflight = validate(contract_path)
    if not preflight["execution_ready"]:
        raise ValueError(f"VTO preflight blocked: {', '.join(preflight['blockers'])}")

    contract = load_contract(contract_path)
    product_path = resolve_path(contract["inputs"]["product"]["path"])
    model_path = resolve_path(contract["inputs"]["model"]["path"])
    request = contract["request"]
    output_path = resolve_path(contract["output"]["candidate_path"])
    receipt_path = resolve_path(contract["output"]["execution_receipt"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.parent.mkdir(parents=True, exist_ok=True)

    async with FashnClient.from_env() as client:
        result = await client.run_tryon_max(
            model_image=image_data_uri(model_path),
            product_image=image_data_uri(product_path),
            prompt=request["prompt"],
            resolution=request["resolution"],
            generation_mode=request["generation_mode"],
            seed=request["seed"],
            num_images=request["num_images"],
            output_format=request["output_format"],
            return_base64=request["return_base64"],
        )

    expected_credits = preflight["estimated_credits"]
    if result.credits_used != expected_credits:
        raise ValueError(
            f"provider credit result {result.credits_used!r} does not match "
            f"contracted estimate {expected_credits!r}"
        )

    output_bytes = decode_png_data_uri(result.output_urls[0])
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "pilot_id": contract["pilot_id"],
        "scene_id": contract["scene_id"],
        "sku": contract["sku"],
        "contract_sha256": preflight["contract_sha256"],
        "execution_fingerprint": preflight["execution_fingerprint"],
        "provider": "fashn",
        "requested_model": TRYON_MAX_MODEL,
        "model_lifecycle": TRYON_MAX_LIFECYCLE,
        "job_id": result.job_id,
        "request_parameters": request,
        "product_sha256": contract["inputs"]["product"]["sha256"],
        "model_sha256": contract["inputs"]["model"]["sha256"],
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "output_sha256": sha256_file(output_path),
        "credits_used": result.credits_used,
        "candidate_only": True,
        "independent_review_required": True,
        "founder_approval_required": True,
        "promotion_authorized": False,
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    return receipt


def verify_review(
    contract_path: Path,
    *,
    candidate_path: Path,
    execution_receipt_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    """Verify an independent, candidate-bound product-fidelity review."""
    contract = load_contract(contract_path)
    expected_candidate = resolve_path(contract["output"]["candidate_path"])
    candidate_path = candidate_path.expanduser().resolve()
    if candidate_path != expected_candidate:
        raise ValueError("review candidate does not match the contracted output path")
    candidate_hash = sha256_file(candidate_path)

    expected_execution_receipt = resolve_path(contract["output"]["execution_receipt"])
    execution_receipt_path = resolve_path(str(execution_receipt_path))
    review_path = resolve_path(str(review_path))
    if execution_receipt_path != expected_execution_receipt:
        raise ValueError("review execution receipt does not match the contracted receipt path")
    execution_receipt = json.loads(execution_receipt_path.read_text(encoding="utf-8"))
    review = json.loads(review_path.read_text(encoding="utf-8"))
    contract_hash = sha256_file(contract_path)
    failures: list[str] = []
    if execution_receipt.get("schema") != RECEIPT_SCHEMA:
        failures.append("invalid execution receipt schema")
    if execution_receipt.get("contract_sha256") != contract_hash:
        failures.append("execution receipt is stale for the current contract")
    if execution_receipt.get("output_sha256") != candidate_hash:
        failures.append("execution receipt is not bound to the current candidate")
    if execution_receipt.get("candidate_only") is not True:
        failures.append("execution receipt must retain candidate_only=true")

    expected_checks = contract["review_gate"]["required_checks"]
    checks = review.get("checks")
    reviewer = str(review.get("reviewer", "")).strip()
    candidate_author = str(contract["review_gate"].get("candidate_author", "")).strip()
    if review.get("schema") != REVIEW_SCHEMA:
        failures.append("invalid independent review schema")
    if review.get("pilot_id") != contract["pilot_id"]:
        failures.append("review pilot_id does not match the contract")
    if review.get("contract_sha256") != contract_hash:
        failures.append("review is stale for the current contract")
    if review.get("candidate_sha256") != candidate_hash:
        failures.append("review is not bound to the current candidate")
    if not reviewer:
        failures.append("reviewer must be named")
    elif candidate_author and reviewer.casefold() == candidate_author.casefold():
        failures.append("reviewer must be independent from the candidate author")
    if review.get("verdict") != "PASS":
        failures.append("independent review verdict must be PASS")
    if review.get("findings") != []:
        failures.append("passing review cannot contain unresolved findings")
    if not isinstance(checks, dict) or set(checks) != set(expected_checks):
        failures.append("review checks must exactly match the contracted check set")
    elif any(value != "PASS" for value in checks.values()):
        failures.append("every contracted product-fidelity check must pass")

    return {
        "schema": "skyyrose.vto-review-verification/1",
        "pilot_id": contract["pilot_id"],
        "contract_sha256": contract_hash,
        "candidate_sha256": candidate_hash,
        "execution_receipt_sha256": sha256_file(execution_receipt_path),
        "review_sha256": sha256_file(review_path),
        "reviewer": reviewer or None,
        "status": "PASS" if not failures else "BLOCKED",
        "failures": failures,
        "candidate_only": True,
        "founder_approval_required": True,
        "scene_input_authorized": False,
        "promotion_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--contract", type=Path, required=True)
    execute_parser = subparsers.add_parser("execute")
    execute_parser.add_argument("--contract", type=Path, required=True)
    execute_parser.add_argument("--allow-spend", action="store_true")
    review_parser = subparsers.add_parser("review")
    review_parser.add_argument("--contract", type=Path, required=True)
    review_parser.add_argument("--candidate", type=Path, required=True)
    review_parser.add_argument("--execution-receipt", type=Path, required=True)
    review_parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()

    try:
        contract_path = args.contract.expanduser().resolve()
        if args.command == "validate":
            result = validate(contract_path)
        elif args.command == "execute":
            result = asyncio.run(execute(contract_path, allow_spend=args.allow_spend))
        else:
            result = verify_review(
                contract_path,
                candidate_path=args.candidate,
                execution_receipt_path=args.execution_receipt,
                review_path=args.review,
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "message": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result.get("status", "PASS") == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
