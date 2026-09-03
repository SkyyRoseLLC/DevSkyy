#!/usr/bin/env python3
"""Fail-closed FASHN Try-On Max candidate runner for SkyyRose scenes."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import math
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def execution_fingerprint(contract: dict[str, Any]) -> str:
    """Hash executable intent without the self-referential approval path."""
    normalized = deepcopy(contract)
    normalized.setdefault("credit_control", {})["approval_receipt"] = None
    return sha256_text(json.dumps(normalized, sort_keys=True, separators=(",", ":")))


def attempt_marker_path(contract: dict[str, Any]) -> Path:
    """Return the immutable paid-attempt marker, derived if not explicit."""
    configured = contract["output"].get("attempt_marker")
    if isinstance(configured, str) and configured.strip():
        return resolve_path(configured)
    receipt = resolve_path(contract["output"]["execution_receipt"])
    return receipt.with_name(f"{receipt.stem}-paid-attempt.json")


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


def verify_approval(
    contract: dict[str, Any],
    estimated_credits: int | None,
    *,
    require_receipt: bool = True,
) -> dict[str, Any]:
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

    if not require_receipt:
        return {
            "status": "PENDING",
            "code": "PAID_APPROVAL_PENDING",
            "max_credits": max_credits,
        }

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


def validate(
    contract_path: Path,
    *,
    require_approval: bool = True,
    credit_balance: int | None = None,
) -> dict[str, Any]:
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
    checks["approval"] = verify_approval(
        contract,
        estimated_credits,
        require_receipt=require_approval,
    )

    if credit_balance is not None:
        if isinstance(credit_balance, bool) or not isinstance(credit_balance, int):
            checks["credits"] = {"status": "BLOCKED", "code": "INVALID_CREDIT_BALANCE"}
        elif estimated_credits is not None and credit_balance < estimated_credits:
            checks["credits"] = {
                "status": "BLOCKED",
                "code": "INSUFFICIENT_FASHN_CREDITS",
                "available": credit_balance,
                "required": estimated_credits,
            }
        else:
            checks["credits"] = {
                "status": "PASS",
                "code": "SUFFICIENT_FASHN_CREDITS",
                "available": credit_balance,
                "required": estimated_credits,
            }

    output = contract["output"]
    try:
        output_path = resolve_path(output.get("candidate_path"))
        receipt_path = resolve_path(output.get("execution_receipt"))
        marker_path = attempt_marker_path(contract)
        output_check: dict[str, Any] = {
            "status": "PASS",
            "code": "OUTPUT_PATHS_RESERVED",
            "candidate_path": str(output_path),
            "execution_receipt": str(receipt_path),
            "attempt_marker": str(marker_path),
        }
        if output_path.exists() or receipt_path.exists() or marker_path.exists():
            output_check = {
                "status": "BLOCKED",
                "code": "OUTPUT_OR_RECEIPT_ALREADY_EXISTS",
            }
    except (TypeError, ValueError) as exc:
        output_check = {"status": "BLOCKED", "code": "INVALID_OUTPUT_PATH", "message": str(exc)}
    checks["output"] = output_check

    flat_checks = [value for value in checks.values() if isinstance(value, dict)]
    flat_checks.extend(support_checks)
    blockers = []
    for check in flat_checks:
        if check.get("status") == "PASS":
            continue
        if not require_approval and check.get("code") == "PAID_APPROVAL_PENDING":
            continue
        blockers.append(check.get("code", "UNKNOWN"))
    technical_ready = not blockers
    approval_ready = checks["approval"].get("status") == "PASS"
    execution_ready = technical_ready and approval_ready
    status = "PASS" if execution_ready else "BLOCKED"
    if technical_ready and not require_approval:
        status = "PASS_PENDING_APPROVAL"
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
        "credit_balance": credit_balance,
        "technical_ready": technical_ready,
        "execution_ready": execution_ready,
        "status": status,
        "blockers": blockers,
    }


async def prepare(contract_path: Path) -> dict[str, Any]:
    """Run zero-credit technical validation plus authenticated balance lookup."""
    local = validate(contract_path, require_approval=False)
    if not local["technical_ready"]:
        return local
    async with FashnClient.from_default() as client:
        balance = await client.get_credits()
    prepared = validate(
        contract_path,
        require_approval=False,
        credit_balance=balance.total,
    )
    prepared["fashn_balance"] = {
        "total": balance.total,
        "subscription": balance.subscription,
        "on_demand": balance.on_demand,
    }
    return prepared


def _verified_binding_bytes(binding: dict[str, Any], *, label: str) -> tuple[Path, bytes, str]:
    path = resolve_path(binding["path"])
    payload = path.read_bytes()
    if not payload or len(payload) > MAX_INPUT_BYTES:
        raise ValueError(f"{label} must be between 1 and {MAX_INPUT_BYTES} bytes")
    actual = sha256_bytes(payload)
    if actual != binding.get("sha256"):
        raise ValueError(f"{label} changed after approval")
    return path, payload, actual


def image_data_uri_bytes(payload: bytes, *, suffix: str) -> str:
    mime = IMAGE_MIME_TYPES.get(suffix.lower())
    if mime is None:
        raise ValueError(f"unsupported input image format: {suffix}")
    if not payload or len(payload) > MAX_INPUT_BYTES:
        raise ValueError(f"input image must be between 1 and {MAX_INPUT_BYTES} bytes")
    return f"data:{mime};base64,{base64.b64encode(payload).decode('ascii')}"


def image_data_uri(path: Path) -> str:
    mime = IMAGE_MIME_TYPES.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"unsupported input image format: {path.suffix}")
    size = path.stat().st_size
    if size <= 0 or size > MAX_INPUT_BYTES:
        raise ValueError(f"input image must be between 1 and {MAX_INPUT_BYTES} bytes")
    return image_data_uri_bytes(path.read_bytes(), suffix=path.suffix)


def _write_attempt_marker(
    *,
    contract: dict[str, Any],
    contract_sha256: str,
    fingerprint: str,
    source_sha256s: dict[str, str],
) -> Path:
    """Atomically consume the one approved paid attempt before provider submission."""
    marker_path = attempt_marker_path(contract)
    marker_path.parent.mkdir(parents=True, exist_ok=True)
    approval = contract["credit_control"]["approval_receipt"]
    approval_path = resolve_path(approval["path"])
    approval_sha256 = sha256_file(approval_path)
    if approval_sha256 != approval.get("sha256"):
        raise ValueError("paid approval receipt changed before attempt consumption")
    marker = {
        "schema": "skyyrose.paid-attempt-marker/1",
        "created_at": datetime.now(UTC).isoformat(),
        "operation": "fashn_tryon_max",
        "pilot_id": contract["pilot_id"],
        "contract_sha256": contract_sha256,
        "execution_fingerprint": fingerprint,
        "approval_receipt_sha256": approval_sha256,
        "source_sha256s": source_sha256s,
        "state": "CONSUMED_BEFORE_SUBMISSION",
        "automatic_retry": False,
        "immutable": True,
    }
    try:
        with marker_path.open("x", encoding="utf-8") as handle:
            json.dump(marker, handle, indent=2)
            handle.write("\n")
    except FileExistsError as exc:
        raise ValueError("approved paid attempt has already been consumed") from exc
    return marker_path


def decode_png_data_uri(value: str) -> bytes:
    prefix = "data:image/png;base64,"
    if not value.startswith(prefix):
        raise ValueError("Try-On Max output is not a PNG data URI")
    payload = base64.b64decode(value[len(prefix) :], validate=True)
    if not payload.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("decoded Try-On Max output is not a PNG")
    return payload


def png_dimensions(payload: bytes) -> dict[str, int]:
    """Fully decode a PNG so truncated or corrupt candidates cannot be receipted."""
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG":
                raise ValueError("candidate output is not PNG")
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = image.size
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise ValueError("PNG output is corrupt or truncated") from exc
    if width <= 0 or height <= 0:
        raise ValueError("PNG output dimensions must be positive")
    return {"width": width, "height": height}


def validate_output_geometry(
    dimensions: dict[str, int], *, resolution: str, expected_aspect_ratio: float
) -> None:
    minimum_long_edge = {"1k": 768, "2k": 1536, "4k": 3072}[resolution]
    width, height = dimensions["width"], dimensions["height"]
    if max(width, height) < minimum_long_edge:
        raise ValueError(f"candidate output is below the contracted {resolution} tier")
    observed = width / height
    if abs(observed - expected_aspect_ratio) / expected_aspect_ratio > 0.03:
        raise ValueError("candidate output aspect ratio does not match the contracted model plate")


def image_aspect_ratio(payload: bytes) -> float:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            image.verify()
        with Image.open(io.BytesIO(payload)) as image:
            image.load()
            width, height = image.size
    except (OSError, SyntaxError, UnidentifiedImageError) as exc:
        raise ValueError("model image is corrupt or truncated") from exc
    return width / height


def _audit_field_failures(receipt: dict[str, Any], *, expected_credits: int) -> list[str]:
    failures: list[str] = []
    try:
        created_at = datetime.fromisoformat(str(receipt.get("created_at", "")))
    except ValueError:
        created_at = None
    if created_at is None or created_at.tzinfo is None:
        failures.append("execution receipt created_at is invalid")
    latency = receipt.get("provider_latency_seconds")
    if (
        isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not math.isfinite(float(latency))
        or float(latency) < 0
    ):
        failures.append("execution receipt provider latency is invalid")
    before = receipt.get("balance_before")
    after = receipt.get("balance_after")
    status = receipt.get("balance_after_status")
    if isinstance(before, bool) or not isinstance(before, int) or before < expected_credits:
        failures.append("execution receipt balance_before is invalid")
    if status == "RECORDED":
        if isinstance(after, bool) or not isinstance(after, int) or after < 0:
            failures.append("execution receipt balance_after is invalid")
        elif isinstance(before, int) and before - after != expected_credits:
            failures.append("execution receipt balance delta does not match provider credits")
    elif status == "RECONCILIATION_REQUIRED":
        if after is not None:
            failures.append("reconciliation-required receipt must not claim balance_after")
    else:
        failures.append("execution receipt balance_after_status is invalid")
    return failures


async def execute(contract_path: Path, *, allow_spend: bool) -> dict[str, Any]:
    if not allow_spend:
        raise ValueError("execution requires --allow-spend")
    contract_bytes = contract_path.read_bytes()
    contract_sha256 = sha256_bytes(contract_bytes)
    contract = load_contract(contract_path)
    fingerprint = execution_fingerprint(contract)
    local_preflight = validate(contract_path, require_approval=True)
    if not local_preflight["execution_ready"]:
        raise ValueError(f"VTO preflight blocked: {', '.join(local_preflight['blockers'])}")
    if (
        local_preflight["contract_sha256"] != contract_sha256
        or local_preflight["execution_fingerprint"] != fingerprint
    ):
        raise ValueError("contract changed while sealing the paid execution")

    product_path, product_bytes, product_sha256 = _verified_binding_bytes(
        contract["inputs"]["product"], label="inputs.product"
    )
    model_path, model_bytes, model_sha256 = _verified_binding_bytes(
        contract["inputs"]["model"], label="inputs.model"
    )
    source_sha256s = {"product": product_sha256, "model": model_sha256}
    for binding in contract["inputs"]["supporting_product_views"]:
        _, _, actual = _verified_binding_bytes(
            binding, label=f"supporting_product_views.{binding['view']}"
        )
        source_sha256s[f"supporting_{binding['view']}"] = actual
    request = contract["request"]
    output_path = resolve_path(contract["output"]["candidate_path"])
    receipt_path = resolve_path(contract["output"]["execution_receipt"])
    approval_path = resolve_path(contract["credit_control"]["approval_receipt"]["path"])
    approval_sha256 = sha256_file(approval_path)

    async with FashnClient.from_default() as client:
        balance_before = await client.get_credits()
        preflight = validate(
            contract_path,
            require_approval=True,
            credit_balance=balance_before.total,
        )
        if not preflight["execution_ready"]:
            raise ValueError(f"VTO preflight blocked: {', '.join(preflight['blockers'])}")
        if (
            preflight["contract_sha256"] != contract_sha256
            or preflight["execution_fingerprint"] != fingerprint
            or sha256_file(contract_path) != contract_sha256
        ):
            raise ValueError("contract changed after approval or during paid preflight")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        receipt_path.parent.mkdir(parents=True, exist_ok=True)
        marker_path = _write_attempt_marker(
            contract=contract,
            contract_sha256=contract_sha256,
            fingerprint=fingerprint,
            source_sha256s=source_sha256s,
        )
        result = await client.run_tryon_max(
            model_image=image_data_uri_bytes(model_bytes, suffix=model_path.suffix),
            product_image=image_data_uri_bytes(product_bytes, suffix=product_path.suffix),
            prompt=request["prompt"],
            resolution=request["resolution"],
            generation_mode=request["generation_mode"],
            seed=request["seed"],
            num_images=request["num_images"],
            output_format=request["output_format"],
            return_base64=request["return_base64"],
        )
        try:
            balance_after = await client.get_credits()
        except Exception:  # provider call is consumed; preserve evidence for reconciliation
            balance_after = None

    expected_credits = preflight["estimated_credits"]
    if result.expected_credits != expected_credits or result.actual_credits != expected_credits:
        raise ValueError(
            f"provider credit result {result.actual_credits!r} does not match "
            f"contracted estimate {expected_credits!r}"
        )

    output_bytes = decode_png_data_uri(result.output_urls[0])
    dimensions = png_dimensions(output_bytes)
    validate_output_geometry(
        dimensions,
        resolution=request["resolution"],
        expected_aspect_ratio=image_aspect_ratio(model_bytes),
    )
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "pilot_id": contract["pilot_id"],
        "scene_id": contract["scene_id"],
        "sku": contract["sku"],
        "contract_sha256": contract_sha256,
        "execution_fingerprint": fingerprint,
        "attempt_marker": str(marker_path.relative_to(PROJECT_ROOT)),
        "attempt_marker_sha256": sha256_file(marker_path),
        "approval_receipt": str(approval_path.relative_to(PROJECT_ROOT)),
        "approval_receipt_sha256": approval_sha256,
        "provider": "fashn",
        "requested_model": TRYON_MAX_MODEL,
        "model_lifecycle": TRYON_MAX_LIFECYCLE,
        "job_id": result.job_id,
        "request_parameters": request,
        "product_sha256": product_sha256,
        "model_sha256": model_sha256,
        "source_sha256s": source_sha256s,
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "output_sha256": sha256_file(output_path),
        "output_dimensions": dimensions,
        "contracted_credits": expected_credits,
        "provider_reported_credits": result.actual_credits,
        "credits_used": result.actual_credits,
        "balance_before": balance_before.total,
        "balance_after": balance_after.total if balance_after is not None else None,
        "balance_after_status": (
            "RECORDED" if balance_after is not None else "RECONCILIATION_REQUIRED"
        ),
        "provider_latency_seconds": result.latency_s,
        "output_transport": "base64_png",
        "candidate_only": True,
        "independent_review_required": True,
        "founder_approval_required": True,
        "promotion_authorized": False,
    }
    with receipt_path.open("x", encoding="utf-8") as handle:
        json.dump(receipt, handle, indent=2)
        handle.write("\n")
    return receipt


def _validate_execution_receipt(
    *, contract_path: Path, contract: dict[str, Any], candidate_hash: str, receipt: dict[str, Any]
) -> list[str]:
    failures: list[str] = []
    contract_hash = sha256_file(contract_path)
    fingerprint = execution_fingerprint(contract)
    expected_credits = (
        TRYON_MAX_CREDITS[
            (contract["request"]["generation_mode"], contract["request"]["resolution"])
        ]
        * contract["request"]["num_images"]
    )
    required_equal = {
        "schema": RECEIPT_SCHEMA,
        "pilot_id": contract["pilot_id"],
        "scene_id": contract["scene_id"],
        "sku": contract["sku"],
        "contract_sha256": contract_hash,
        "execution_fingerprint": fingerprint,
        "provider": "fashn",
        "requested_model": TRYON_MAX_MODEL,
        "model_lifecycle": TRYON_MAX_LIFECYCLE,
        "request_parameters": contract["request"],
        "output_sha256": candidate_hash,
        "output_path": contract["output"]["candidate_path"],
        "contracted_credits": expected_credits,
        "provider_reported_credits": expected_credits,
        "credits_used": expected_credits,
        "candidate_only": True,
        "independent_review_required": True,
        "founder_approval_required": True,
        "promotion_authorized": False,
        "output_transport": "base64_png",
        "approval_receipt": contract["credit_control"]["approval_receipt"]["path"],
        "approval_receipt_sha256": sha256_file(
            resolve_path(contract["credit_control"]["approval_receipt"]["path"])
        ),
    }
    for key, expected in required_equal.items():
        if receipt.get(key) != expected:
            failures.append(f"execution receipt {key} mismatch")
    if not str(receipt.get("job_id", "")).strip():
        failures.append("execution receipt requires provider job_id")
    candidate_path = resolve_path(contract["output"]["candidate_path"])
    try:
        actual_dimensions = png_dimensions(candidate_path.read_bytes())
        model_bytes = resolve_path(contract["inputs"]["model"]["path"]).read_bytes()
        validate_output_geometry(
            actual_dimensions,
            resolution=contract["request"]["resolution"],
            expected_aspect_ratio=image_aspect_ratio(model_bytes),
        )
    except (OSError, ValueError) as exc:
        failures.append(f"candidate image validation failed: {exc}")
    else:
        if receipt.get("output_dimensions") != actual_dimensions:
            failures.append("execution receipt dimensions do not match candidate")
    failures.extend(_audit_field_failures(receipt, expected_credits=expected_credits))

    actual_sources: dict[str, str] = {}
    for name in ("product", "model"):
        binding = contract["inputs"][name]
        actual = sha256_file(resolve_path(binding["path"]))
        actual_sources[name] = actual
        if actual != binding["sha256"] or receipt.get(f"{name}_sha256") != actual:
            failures.append(f"execution receipt {name} source mismatch")
    for binding in contract["inputs"]["supporting_product_views"]:
        key = f"supporting_{binding['view']}"
        actual = sha256_file(resolve_path(binding["path"]))
        actual_sources[key] = actual
        if actual != binding["sha256"]:
            failures.append(f"execution receipt {key} source mismatch")
    if receipt.get("source_sha256s") != actual_sources:
        failures.append("execution receipt submitted source set mismatch")

    marker_path = attempt_marker_path(contract)
    expected_marker_relative = str(marker_path.relative_to(PROJECT_ROOT))
    if receipt.get("attempt_marker") != expected_marker_relative or not marker_path.is_file():
        failures.append("execution receipt paid-attempt marker is missing")
        return failures
    marker_hash = sha256_file(marker_path)
    if receipt.get("attempt_marker_sha256") != marker_hash:
        failures.append("execution receipt paid-attempt marker hash mismatch")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        failures.append("paid-attempt marker is invalid")
        return failures
    approval = contract["credit_control"]["approval_receipt"]
    approval_hash = sha256_file(resolve_path(approval["path"]))
    marker_required = {
        "schema": "skyyrose.paid-attempt-marker/1",
        "operation": "fashn_tryon_max",
        "pilot_id": contract["pilot_id"],
        "contract_sha256": contract_hash,
        "execution_fingerprint": fingerprint,
        "approval_receipt_sha256": approval_hash,
        "source_sha256s": actual_sources,
        "state": "CONSUMED_BEFORE_SUBMISSION",
        "automatic_retry": False,
        "immutable": True,
    }
    if any(marker.get(key) != value for key, value in marker_required.items()):
        failures.append("paid-attempt marker does not bind the approved execution")
    return failures


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
    failures.extend(
        _validate_execution_receipt(
            contract_path=contract_path,
            contract=contract,
            candidate_hash=candidate_hash,
            receipt=execution_receipt,
        )
    )

    review_gate = contract["review_gate"]
    expected_checks = review_gate["required_checks"]
    expected_pocket_evidence = {
        "wearer_left_side": {
            "candidate_proof": "DIRECTLY_VISIBLE_ZIPPERED",
            "required_disposition": "PASS",
        },
        "wearer_right_side": {
            "candidate_proof": "DIRECTLY_VISIBLE_ZIPPERED",
            "required_disposition": "PASS",
        },
        "wearer_left_rear": {
            "candidate_proof": "NOT_OBSERVABLE_IN_FRONT_CANDIDATE",
            "source_authority": "PASS",
        },
        "wearer_right_rear": {
            "candidate_proof": "NOT_OBSERVABLE_IN_FRONT_CANDIDATE",
            "source_authority": "PASS",
        },
    }
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
    if review_gate.get("pocket_evidence") != expected_pocket_evidence:
        failures.append("contract must preserve the front-candidate pocket evidence matrix")
    if review.get("pocket_evidence") != expected_pocket_evidence:
        failures.append(
            "review must directly pass both side zippers and mark both rear pockets "
            "NOT_OBSERVABLE_IN_FRONT_CANDIDATE with source-authority PASS"
        )

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
        "front_scene_input_scope_only": True,
        "rear_view_catalog_authorized": False,
        "full_360_fidelity_claimed": False,
        "founder_approval_required": True,
        "scene_input_authorized": False,
        "promotion_authorized": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare_parser = subparsers.add_parser("prepare")
    prepare_parser.add_argument("--contract", type=Path, required=True)
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
        if args.command == "prepare":
            result = asyncio.run(prepare(contract_path))
        elif args.command == "validate":
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
    return 0 if result.get("status", "PASS") in {"PASS", "PASS_PENDING_APPROVAL"} else 2


if __name__ == "__main__":
    sys.exit(main())
