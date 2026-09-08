#!/usr/bin/env python3
"""Govern one FASHN Model Create full-body identity plate candidate."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Comfy.scripts.vto_ooda import (  # noqa: E402
    _audit_field_failures,
    decode_png_data_uri,
    image_data_uri_bytes,
    png_dimensions,
    sha256_bytes,
    sha256_file,
    sha256_text,
    validate_output_geometry,
)
from skyyrose.integrations.fashn_client import (  # noqa: E402
    MODEL_CREATE_LIFECYCLE,
    MODEL_CREATE_MODEL,
    FashnClient,
)

CONTRACT_SCHEMA = "skyyrose.model-plate-ooda/1"
APPROVAL_SCHEMA = "skyyrose.model-plate-paid-approval/1"
RECEIPT_SCHEMA = "skyyrose.model-plate-execution-receipt/1"
REVIEW_SCHEMA = "skyyrose.model-plate-independent-review/1"
FOUNDER_AUTHORITY = "FOUNDER_APPROVED_FULL_BODY_MODEL"
EXPECTED_REQUEST = {
    "face_reference_mode": "match_base",
    "aspect_ratio": "9:16",
    "resolution": "1k",
    "generation_mode": "fast",
    "seed": 42,
    "num_images": 1,
    "output_format": "png",
    "return_base64": True,
}
EXPECTED_CREDITS = 4


def resolve_path(raw_path: str) -> Path:
    if not isinstance(raw_path, str) or not raw_path.strip():
        raise ValueError("path must be a non-empty string")
    candidate = Path(raw_path).expanduser()
    resolved = (candidate if candidate.is_absolute() else PROJECT_ROOT / candidate).resolve()
    if not resolved.is_relative_to(PROJECT_ROOT.resolve()):
        raise ValueError(f"path escapes project root: {raw_path}")
    return resolved


def verify_file(binding: dict[str, Any], *, label: str) -> dict[str, Any]:
    raw_path = binding.get("path")
    expected = binding.get("sha256")
    if not raw_path:
        return {"label": label, "status": "BLOCKED", "code": "MISSING_PATH"}
    try:
        path = resolve_path(raw_path)
    except ValueError as exc:
        return {
            "label": label,
            "status": "BLOCKED",
            "code": "INVALID_PATH",
            "message": str(exc),
        }
    if not path.is_file():
        return {
            "label": label,
            "path": str(path),
            "status": "BLOCKED",
            "code": "MISSING_FILE",
        }
    actual = sha256_file(path)
    if not isinstance(expected, str) or actual != expected:
        return {
            "label": label,
            "path": str(path),
            "status": "BLOCKED",
            "code": "HASH_MISMATCH",
            "expected_sha256": expected,
            "actual_sha256": actual,
        }
    return {
        "label": label,
        "path": str(path),
        "status": "PASS",
        "code": "HASH_VERIFIED",
        "actual_sha256": actual,
    }


def load_contract(path: Path) -> dict[str, Any]:
    contract = json.loads(path.read_text(encoding="utf-8"))
    if contract.get("schema") != CONTRACT_SCHEMA:
        raise ValueError(f"unsupported model-plate schema: {contract.get('schema')}")
    for key in (
        "plate_id",
        "identity_id",
        "model",
        "inputs",
        "request",
        "credit_control",
        "review_gate",
        "output",
    ):
        if key not in contract:
            raise ValueError(f"contract missing required key: {key}")
    return contract


def execution_fingerprint(contract: dict[str, Any]) -> str:
    normalized = deepcopy(contract)
    normalized.setdefault("credit_control", {})["approval_receipt"] = None
    return sha256_text(json.dumps(normalized, sort_keys=True, separators=(",", ":")))


def attempt_marker_path(contract: dict[str, Any]) -> Path:
    configured = contract["output"].get("attempt_marker")
    if isinstance(configured, str) and configured.strip():
        return resolve_path(configured)
    receipt = resolve_path(contract["output"]["execution_receipt"])
    return receipt.with_name(f"{receipt.stem}-paid-attempt.json")


def _verified_binding_bytes(binding: dict[str, Any], *, label: str) -> tuple[Path, bytes, str]:
    path = resolve_path(binding["path"])
    payload = path.read_bytes()
    if not payload:
        raise ValueError(f"{label} must not be empty")
    actual = sha256_bytes(payload)
    if actual != binding.get("sha256"):
        raise ValueError(f"{label} changed after approval")
    return path, payload, actual


def _write_attempt_marker(
    *,
    contract: dict[str, Any],
    contract_sha256: str,
    fingerprint: str,
    source_sha256s: dict[str, str],
) -> Path:
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
        "operation": "fashn_model_create",
        "plate_id": contract["plate_id"],
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


def _verify_request(contract: dict[str, Any]) -> dict[str, Any]:
    request = contract["request"]
    failures: list[str] = []
    if not isinstance(request.get("prompt"), str) or not request["prompt"].strip():
        failures.append("prompt must be non-empty")
    for key, expected in EXPECTED_REQUEST.items():
        if request.get(key) != expected:
            failures.append(f"{key} must equal {expected!r}")
    if request.get("face_reference_input") != "front_bust":
        failures.append("face_reference_input must equal front_bust")
    return {
        "status": "PASS" if not failures else "BLOCKED",
        "code": "MODEL_CREATE_REQUEST_VERIFIED" if not failures else "INVALID_REQUEST",
        "expected_credits": EXPECTED_CREDITS,
        "failures": failures,
    }


def _verify_approval(contract: dict[str, Any], *, require_receipt: bool) -> dict[str, Any]:
    control = contract["credit_control"]
    failures: list[str] = []
    if control.get("max_paid_generations") != 1:
        failures.append("max_paid_generations must equal 1")
    if control.get("paid_generations_recorded") != 0:
        failures.append("paid_generations_recorded must equal 0")
    if control.get("automatic_paid_retries") is not False:
        failures.append("automatic_paid_retries must be false")
    if control.get("max_credits") != EXPECTED_CREDITS:
        failures.append("max_credits must equal 4")
    if failures:
        return {"status": "BLOCKED", "code": "INVALID_CREDIT_CONTROL", "failures": failures}
    if not require_receipt:
        return {"status": "PENDING", "code": "PAID_APPROVAL_PENDING"}
    binding = control.get("approval_receipt")
    if not isinstance(binding, dict):
        return {"status": "BLOCKED", "code": "MISSING_PAID_APPROVAL_RECEIPT"}
    checked = verify_file(binding, label="credit_control.approval_receipt")
    if checked["status"] != "PASS":
        return checked
    receipt = json.loads(Path(checked["path"]).read_text(encoding="utf-8"))
    required = {
        "schema": APPROVAL_SCHEMA,
        "plate_id": contract["plate_id"],
        "execution_fingerprint": execution_fingerprint(contract),
        "max_credits": EXPECTED_CREDITS,
        "approved": True,
    }
    if any(receipt.get(key) != value for key, value in required.items()):
        return {"status": "BLOCKED", "code": "INVALID_PAID_APPROVAL_RECEIPT"}
    if (
        not str(receipt.get("approved_by", "")).strip()
        or not str(receipt.get("approved_at", "")).strip()
    ):
        return {"status": "BLOCKED", "code": "INVALID_PAID_APPROVAL_RECEIPT"}
    return {"status": "PASS", "code": "PAID_ATTEMPT_APPROVED"}


def validate(
    contract_path: Path,
    *,
    require_approval: bool,
    credit_balance: int | None = None,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    model = contract["model"]
    checks: dict[str, Any] = {
        "model": {
            "status": (
                "PASS"
                if model.get("id") == MODEL_CREATE_MODEL
                and model.get("lifecycle") == MODEL_CREATE_LIFECYCLE
                else "BLOCKED"
            ),
            "code": (
                "MODEL_CREATE_EXPERIMENTAL_CANDIDATE_ONLY"
                if model.get("id") == MODEL_CREATE_MODEL
                and model.get("lifecycle") == MODEL_CREATE_LIFECYCLE
                else "MODEL_OR_LIFECYCLE_MISMATCH"
            ),
        },
        "request": _verify_request(contract),
        "approval": _verify_approval(contract, require_receipt=require_approval),
    }
    inputs = contract["inputs"]
    for name in ("identity_manifest", "founder_identity_approval", "front_bust", "rear_profile"):
        binding = inputs.get(name)
        checks[name] = (
            verify_file(binding, label=f"inputs.{name}")
            if isinstance(binding, dict)
            else {"status": "BLOCKED", "code": f"MISSING_{name.upper()}"}
        )
    if credit_balance is not None:
        checks["credits"] = {
            "status": "PASS" if credit_balance >= EXPECTED_CREDITS else "BLOCKED",
            "code": (
                "SUFFICIENT_FASHN_CREDITS"
                if credit_balance >= EXPECTED_CREDITS
                else "INSUFFICIENT_FASHN_CREDITS"
            ),
            "available": credit_balance,
            "required": EXPECTED_CREDITS,
        }
    output = contract["output"]
    try:
        candidate = resolve_path(output["candidate_path"])
        receipt = resolve_path(output["execution_receipt"])
        marker = attempt_marker_path(contract)
        output_ok = not candidate.exists() and not receipt.exists() and not marker.exists()
        checks["output"] = {
            "status": "PASS" if output_ok else "BLOCKED",
            "code": "OUTPUT_PATHS_RESERVED" if output_ok else "OUTPUT_OR_RECEIPT_ALREADY_EXISTS",
            "attempt_marker": str(marker),
        }
    except (KeyError, TypeError, ValueError) as exc:
        checks["output"] = {
            "status": "BLOCKED",
            "code": "INVALID_OUTPUT_PATH",
            "message": str(exc),
        }

    blockers: list[str] = []
    for check in checks.values():
        if check.get("status") == "PASS":
            continue
        if not require_approval and check.get("code") == "PAID_APPROVAL_PENDING":
            continue
        blockers.append(check.get("code", "UNKNOWN"))
    technical_ready = not blockers
    execution_ready = technical_ready and checks["approval"].get("status") == "PASS"
    status = "PASS" if execution_ready else "BLOCKED"
    if technical_ready and not require_approval:
        status = "PASS_PENDING_APPROVAL"
    return {
        "schema": "skyyrose.model-plate-preflight/1",
        "plate_id": contract["plate_id"],
        "identity_id": contract["identity_id"],
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
        "execution_fingerprint": execution_fingerprint(contract),
        "checks": checks,
        "expected_credits": EXPECTED_CREDITS,
        "credit_balance": credit_balance,
        "technical_ready": technical_ready,
        "execution_ready": execution_ready,
        "status": status,
        "blockers": blockers,
    }


async def preflight(contract_path: Path) -> dict[str, Any]:
    local = validate(contract_path, require_approval=False)
    if not local["technical_ready"]:
        return local
    async with FashnClient.from_default() as client:
        balance = await client.get_credits()
    result = validate(contract_path, require_approval=False, credit_balance=balance.total)
    result["fashn_balance"] = {
        "total": balance.total,
        "subscription": balance.subscription,
        "on_demand": balance.on_demand,
    }
    return result


async def execute(contract_path: Path, *, allow_spend: bool) -> dict[str, Any]:
    if not allow_spend:
        raise ValueError("execution requires --allow-spend")
    contract_bytes = contract_path.read_bytes()
    contract_sha256 = sha256_bytes(contract_bytes)
    contract = load_contract(contract_path)
    fingerprint = execution_fingerprint(contract)
    local = validate(contract_path, require_approval=True)
    if not local["execution_ready"]:
        raise ValueError(f"model-plate preflight blocked: {', '.join(local['blockers'])}")
    if local["contract_sha256"] != contract_sha256 or local["execution_fingerprint"] != fingerprint:
        raise ValueError("contract changed while sealing the paid execution")
    request = contract["request"]
    source_payloads: dict[str, tuple[Path, bytes, str]] = {}
    for name in ("identity_manifest", "founder_identity_approval", "front_bust", "rear_profile"):
        source_payloads[name] = _verified_binding_bytes(
            contract["inputs"][name], label=f"inputs.{name}"
        )
    front_bust, front_bust_bytes, _ = source_payloads["front_bust"]
    source_sha256s = {name: payload[2] for name, payload in source_payloads.items()}
    output_path = resolve_path(contract["output"]["candidate_path"])
    receipt_path = resolve_path(contract["output"]["execution_receipt"])
    approval_path = resolve_path(contract["credit_control"]["approval_receipt"]["path"])
    approval_sha256 = sha256_file(approval_path)

    async with FashnClient.from_default() as client:
        balance_before = await client.get_credits()
        checked = validate(
            contract_path,
            require_approval=True,
            credit_balance=balance_before.total,
        )
        if not checked["execution_ready"]:
            raise ValueError(f"model-plate preflight blocked: {', '.join(checked['blockers'])}")
        if (
            checked["contract_sha256"] != contract_sha256
            or checked["execution_fingerprint"] != fingerprint
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
        result = await client.run_model_create(
            prompt=request["prompt"],
            face_reference=image_data_uri_bytes(front_bust_bytes, suffix=front_bust.suffix),
            face_reference_mode=request["face_reference_mode"],
            aspect_ratio=request["aspect_ratio"],
            resolution=request["resolution"],
            generation_mode=request["generation_mode"],
            seed=request["seed"],
            num_images=request["num_images"],
            output_format=request["output_format"],
            return_base64=request["return_base64"],
        )
        try:
            balance_after = await client.get_credits()
        except Exception:
            balance_after = None

    if result.expected_credits != EXPECTED_CREDITS or result.actual_credits != EXPECTED_CREDITS:
        raise ValueError("provider-reported Model Create credits do not match the contract")
    output_bytes = decode_png_data_uri(result.output_urls[0])
    dimensions = png_dimensions(output_bytes)
    ratio_left, ratio_right = request["aspect_ratio"].split(":", 1)
    validate_output_geometry(
        dimensions,
        resolution=request["resolution"],
        expected_aspect_ratio=int(ratio_left) / int(ratio_right),
    )
    with output_path.open("xb") as handle:
        handle.write(output_bytes)
    receipt = {
        "schema": RECEIPT_SCHEMA,
        "created_at": datetime.now(UTC).isoformat(),
        "plate_id": contract["plate_id"],
        "identity_id": contract["identity_id"],
        "contract_sha256": contract_sha256,
        "execution_fingerprint": fingerprint,
        "attempt_marker": str(marker_path.relative_to(PROJECT_ROOT)),
        "attempt_marker_sha256": sha256_file(marker_path),
        "approval_receipt": str(approval_path.relative_to(PROJECT_ROOT)),
        "approval_receipt_sha256": approval_sha256,
        "provider": "fashn",
        "requested_model": MODEL_CREATE_MODEL,
        "model_lifecycle": MODEL_CREATE_LIFECYCLE,
        "job_id": result.job_id,
        "request_parameters": request,
        "source_sha256s": source_sha256s,
        "output_path": str(output_path.relative_to(PROJECT_ROOT)),
        "output_sha256": sha256_file(output_path),
        "output_dimensions": dimensions,
        "contracted_credits": EXPECTED_CREDITS,
        "provider_reported_credits": result.actual_credits,
        "balance_before": balance_before.total,
        "balance_after": balance_after.total if balance_after is not None else None,
        "balance_after_status": (
            "RECORDED" if balance_after is not None else "RECONCILIATION_REQUIRED"
        ),
        "provider_latency_seconds": result.latency_s,
        "output_transport": "base64_png",
        "candidate_only": True,
        "founder_authority": None,
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
    required_equal = {
        "schema": RECEIPT_SCHEMA,
        "plate_id": contract["plate_id"],
        "identity_id": contract["identity_id"],
        "contract_sha256": contract_hash,
        "execution_fingerprint": fingerprint,
        "provider": "fashn",
        "requested_model": MODEL_CREATE_MODEL,
        "model_lifecycle": MODEL_CREATE_LIFECYCLE,
        "request_parameters": contract["request"],
        "output_sha256": candidate_hash,
        "output_path": contract["output"]["candidate_path"],
        "contracted_credits": EXPECTED_CREDITS,
        "provider_reported_credits": EXPECTED_CREDITS,
        "candidate_only": True,
        "output_transport": "base64_png",
        "approval_receipt": contract["credit_control"]["approval_receipt"]["path"],
        "approval_receipt_sha256": sha256_file(
            resolve_path(contract["credit_control"]["approval_receipt"]["path"])
        ),
        "founder_authority": None,
    }
    for key, expected in required_equal.items():
        if receipt.get(key) != expected:
            failures.append(f"execution receipt {key} mismatch")
    if not str(receipt.get("job_id", "")).strip():
        failures.append("execution receipt requires provider job_id")
    candidate_path = resolve_path(contract["output"]["candidate_path"])
    try:
        actual_dimensions = png_dimensions(candidate_path.read_bytes())
        ratio_left, ratio_right = contract["request"]["aspect_ratio"].split(":", 1)
        validate_output_geometry(
            actual_dimensions,
            resolution=contract["request"]["resolution"],
            expected_aspect_ratio=int(ratio_left) / int(ratio_right),
        )
    except (OSError, ValueError) as exc:
        failures.append(f"candidate image validation failed: {exc}")
    else:
        if receipt.get("output_dimensions") != actual_dimensions:
            failures.append("execution receipt dimensions do not match candidate")
    failures.extend(_audit_field_failures(receipt, expected_credits=EXPECTED_CREDITS))

    actual_sources = {
        name: sha256_file(resolve_path(contract["inputs"][name]["path"]))
        for name in ("identity_manifest", "founder_identity_approval", "front_bust", "rear_profile")
    }
    if (
        any(actual_sources[name] != contract["inputs"][name]["sha256"] for name in actual_sources)
        or receipt.get("source_sha256s") != actual_sources
    ):
        failures.append("execution receipt submitted source set mismatch")

    marker_path = attempt_marker_path(contract)
    expected_marker_relative = str(marker_path.relative_to(PROJECT_ROOT))
    if receipt.get("attempt_marker") != expected_marker_relative or not marker_path.is_file():
        failures.append("execution receipt paid-attempt marker is missing")
        return failures
    if receipt.get("attempt_marker_sha256") != sha256_file(marker_path):
        failures.append("execution receipt paid-attempt marker hash mismatch")
    try:
        marker = json.loads(marker_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        failures.append("paid-attempt marker is invalid")
        return failures
    approval = contract["credit_control"]["approval_receipt"]
    marker_required = {
        "schema": "skyyrose.paid-attempt-marker/1",
        "operation": "fashn_model_create",
        "plate_id": contract["plate_id"],
        "contract_sha256": contract_hash,
        "execution_fingerprint": fingerprint,
        "approval_receipt_sha256": sha256_file(resolve_path(approval["path"])),
        "source_sha256s": actual_sources,
        "state": "CONSUMED_BEFORE_SUBMISSION",
        "automatic_retry": False,
        "immutable": True,
    }
    if any(marker.get(key) != value for key, value in marker_required.items()):
        failures.append("paid-attempt marker does not bind the approved execution")
    return failures


def review(
    contract_path: Path,
    *,
    candidate_path: Path,
    execution_receipt_path: Path,
    review_path: Path,
) -> dict[str, Any]:
    contract = load_contract(contract_path)
    candidate = resolve_path(str(candidate_path))
    receipt_path = resolve_path(str(execution_receipt_path))
    review_file = resolve_path(str(review_path))
    if candidate != resolve_path(contract["output"]["candidate_path"]):
        raise ValueError("candidate does not match contracted output")
    if receipt_path != resolve_path(contract["output"]["execution_receipt"]):
        raise ValueError("execution receipt does not match contracted output")
    candidate_hash = sha256_file(candidate)
    contract_hash = sha256_file(contract_path)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    review_data = json.loads(review_file.read_text(encoding="utf-8"))
    failures: list[str] = []
    failures.extend(
        _validate_execution_receipt(
            contract_path=contract_path,
            contract=contract,
            candidate_hash=candidate_hash,
            receipt=receipt,
        )
    )
    expected_checks = contract["review_gate"]["required_checks"]
    reviewer = str(review_data.get("reviewer", "")).strip()
    author = str(contract["review_gate"].get("candidate_author", "")).strip()
    if review_data.get("schema") != REVIEW_SCHEMA:
        failures.append("invalid independent review schema")
    if review_data.get("plate_id") != contract["plate_id"]:
        failures.append("review plate_id mismatch")
    if review_data.get("contract_sha256") != contract_hash:
        failures.append("review is stale")
    if review_data.get("candidate_sha256") != candidate_hash:
        failures.append("review is not candidate-bound")
    if not reviewer or reviewer.casefold() == author.casefold():
        failures.append("reviewer must be named and independent")
    if review_data.get("verdict") != "PASS" or review_data.get("findings") != []:
        failures.append("review must pass without findings")
    checks = review_data.get("checks")
    if not isinstance(checks, dict) or set(checks) != set(expected_checks):
        failures.append("review checks must exactly match the contract")
    elif any(value != "PASS" for value in checks.values()):
        failures.append("all model-plate review checks must pass")
    return {
        "schema": "skyyrose.model-plate-review-verification/1",
        "plate_id": contract["plate_id"],
        "contract_sha256": contract_hash,
        "candidate_sha256": candidate_hash,
        "execution_receipt_sha256": sha256_file(receipt_path),
        "review_sha256": sha256_file(review_file),
        "reviewer": reviewer or None,
        "status": "PASS" if not failures else "BLOCKED",
        "failures": failures,
        "candidate_only": True,
        "founder_approval_required": True,
        "authority_state": "PENDING_FOUNDER_APPROVAL",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    preflight_parser = commands.add_parser("preflight")
    preflight_parser.add_argument("--contract", type=Path, required=True)
    execute_parser = commands.add_parser("execute")
    execute_parser.add_argument("--contract", type=Path, required=True)
    execute_parser.add_argument("--allow-spend", action="store_true")
    review_parser = commands.add_parser("review")
    review_parser.add_argument("--contract", type=Path, required=True)
    review_parser.add_argument("--candidate", type=Path, required=True)
    review_parser.add_argument("--execution-receipt", type=Path, required=True)
    review_parser.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    try:
        contract_path = args.contract.expanduser().resolve()
        if args.command == "preflight":
            result = asyncio.run(preflight(contract_path))
        elif args.command == "execute":
            result = asyncio.run(execute(contract_path, allow_spend=args.allow_spend))
        else:
            result = review(
                contract_path,
                candidate_path=args.candidate,
                execution_receipt_path=args.execution_receipt,
                review_path=args.review,
            )
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"status": "BLOCKED", "message": str(exc)}, indent=2))
        return 2
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") in {"PASS", "PASS_PENDING_APPROVAL"} else 2


if __name__ == "__main__":
    sys.exit(main())
