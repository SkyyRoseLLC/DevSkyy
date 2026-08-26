"""Fail-closed validation for the V2 storefront on-model media intake.

The product SOT supplies the authoritative product hash and permitted media
path.  The opening-media registry supplies the review decision.  This module
joins them only when the bytes on disk match both records; a filename or a
``role`` value alone never authorizes card media.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


THEME_RELATIVE_PREFIX = "wordpress-theme/skyyrose-flagship/"
SUPPORTED_STATES = {"APPROVED_CURRENT_STOREFRONT_ON_MODEL_FRONT", None}


class MediaIntakeError(ValueError):
    """Raised when the SOT/registry contract itself is invalid."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise MediaIntakeError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise MediaIntakeError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise MediaIntakeError(f"{label} must be a JSON object: {path}")
    return value


def _repository_path(repo_root: Path, relative: str) -> Path | None:
    candidate = (repo_root / relative).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError:
        return None
    return candidate


def _registered_source_matches(expected: Path, source: object, repo_root: Path) -> bool:
    if not isinstance(source, str) or not source:
        return False

    candidates = [_repository_path(repo_root, source)]
    if not source.startswith(THEME_RELATIVE_PREFIX):
        candidates.append(_repository_path(repo_root, f"{THEME_RELATIVE_PREFIX}{source}"))
    return any(candidate == expected for candidate in candidates if candidate is not None)


def _integrity_record(registry: dict[str, Any], sku: str, view: dict[str, Any]) -> tuple[object, object]:
    direct_hash = view.get("sha256")
    direct_bytes = view.get("bytes")
    if direct_hash is not None or direct_bytes is not None:
        return direct_hash, direct_bytes

    asset_integrity = registry.get("asset_integrity", {})
    record = asset_integrity.get(sku, {}).get("on_model_front", {})
    if not isinstance(record, dict):
        return None, None
    return record.get("source_sha256"), record.get("source_bytes")


def _approval_reference(product: dict[str, Any], view: dict[str, Any]) -> object:
    return view.get("approval_reference") or product.get("approval_reference")


def _contains_url(value: object) -> bool:
    return isinstance(value, str) and "://" in value


def _validate_local_artifact(
    artifact: object,
    *,
    repo_root: Path,
    label: str,
) -> list[str]:
    if not isinstance(artifact, dict):
        return [f"{label}_INVALID"]
    local_path = artifact.get("local_path")
    if _contains_url(local_path):
        return [f"{label}_URL_NOT_ALLOWED"]
    if not isinstance(local_path, str) or not local_path:
        return [f"{label}_PATH_MISSING"]
    path = _repository_path(repo_root, local_path)
    if path is None:
        return [f"{label}_PATH_ESCAPES_REPOSITORY"]
    if not path.is_file():
        return [f"{label}_MISSING"]

    blockers: list[str] = []
    expected_hash = artifact.get("sha256")
    expected_bytes = artifact.get("bytes")
    if _sha256_file(path) != expected_hash:
        blockers.append(f"{label}_HASH_MISMATCH")
    if path.stat().st_size != expected_bytes:
        blockers.append(f"{label}_BYTES_MISMATCH")
    return blockers


def validate_generation_receipt(
    product_sot_path: Path,
    receipt_path: Path,
    *,
    repo_root: Path,
    require_output: bool = False,
) -> dict[str, Any]:
    """Validate a local-original receipt before or after a candidate render.

    ``require_output`` is false for the preflight receipt written before a paid
    call and true for a generated-candidate receipt.  Both stages require at
    least one retained local product authority; derived assets alone fail.
    """
    product_sot_bytes = product_sot_path.read_bytes()
    product_sot = _load_json(product_sot_path, "product SOT")
    receipt = _load_json(receipt_path, "generation receipt")
    products = product_sot.get("products")
    if not isinstance(products, dict):
        raise MediaIntakeError("product SOT requires a products object")

    blockers: list[str] = []
    sku = receipt.get("sku")
    product = products.get(sku) if isinstance(sku, str) else None
    if not isinstance(product, dict):
        blockers.append("SKU_NOT_IN_PRODUCT_SOT")
    else:
        if receipt.get("product_sot_sha256") != _sha256_bytes(product_sot_bytes):
            blockers.append("PRODUCT_SOT_HASH_MISMATCH")
        if receipt.get("product_hash") != product.get("product_hash"):
            blockers.append("PRODUCT_HASH_MISMATCH")

    originals = receipt.get("originals")
    if not isinstance(originals, list) or not originals:
        blockers.append("ORIGINALS_MISSING")
    else:
        product_authority_present = False
        for original in originals:
            if isinstance(original, dict) and original.get("role") in {
                "physical_product_authority",
                "approved_on_model_source",
            }:
                product_authority_present = True
            blockers.extend(_validate_local_artifact(original, repo_root=repo_root, label="ORIGINAL"))
        if not product_authority_present:
            blockers.append("PRODUCT_AUTHORITY_ORIGINAL_MISSING")

    if receipt.get("operation") != "candidate_only":
        blockers.append("OPERATION_NOT_CANDIDATE_ONLY")
    if receipt.get("model") != "gpt-image-2":
        blockers.append("MODEL_NOT_CANONICAL_PRODUCT_ENGINE")
    if receipt.get("view") not in {"front", "left", "right", "back", "top", "bottom"}:
        blockers.append("GEOMETRY_VIEW_INVALID")

    output = receipt.get("output")
    if require_output or output is not None:
        blockers.extend(_validate_local_artifact(output, repo_root=repo_root, label="OUTPUT"))

    return {"valid": not blockers, "blockers": blockers}


def validate_media_intake(
    product_sot_path: Path,
    registry_path: Path,
    *,
    repo_root: Path,
    require_explicit_approval: bool = False,
) -> dict[str, Any]:
    """Validate every SKU in an opening-media registry against product SOT.

    Existing reviewed records may use the legacy top-level ``reviewed_at`` field
    for ordinary card support.  ``require_explicit_approval`` deliberately
    rejects those records unless each accepted view has an approval reference;
    callers should use that mode for promotion or any broader authorization.
    """
    product_sot_bytes = product_sot_path.read_bytes()
    product_sot = _load_json(product_sot_path, "product SOT")
    registry = _load_json(registry_path, "on-model media registry")
    products = product_sot.get("products")
    registry_products = registry.get("products")
    product_hashes = registry.get("product_hashes")
    if not all(isinstance(value, dict) for value in (products, registry_products, product_hashes)):
        raise MediaIntakeError("product SOT and registry require products/product_hashes objects")
    if set(products) != set(registry_products) or set(products) != set(product_hashes):
        raise MediaIntakeError("product SOT and registry SKU sets differ")
    if registry.get("product_sot_sha256") != _sha256_bytes(product_sot_bytes):
        raise MediaIntakeError("on-model media registry has a stale product SOT hash")

    results: list[dict[str, Any]] = []
    for sku in sorted(products):
        product = products[sku]
        media_entry = registry_products[sku]
        blockers: list[str] = []
        warnings: list[str] = []
        if not isinstance(product, dict) or not isinstance(media_entry, dict):
            raise MediaIntakeError(f"invalid product/registry record for {sku}")
        if product_hashes[sku] != product.get("product_hash"):
            blockers.append("PRODUCT_HASH_MISMATCH")

        views = media_entry.get("views", [])
        if not isinstance(views, list):
            blockers.append("VIEWS_INVALID")
            views = []
        front_views = [
            view
            for view in views
            if isinstance(view, dict) and view.get("role") == "on_model_front"
        ]
        if len(front_views) != 1:
            blockers.append("CURRENT_APPROVED_ON_MODEL_FRONT_MISSING")
        elif media_entry.get("status") not in SUPPORTED_STATES:
            blockers.append("MEDIA_STATUS_NOT_APPROVED")
        else:
            view = front_views[0]
            sot_media = product.get("media", {}).get("on_model_front", {})
            expected_path = _repository_path(repo_root, sot_media.get("path", ""))
            if expected_path is None or not expected_path.is_file():
                blockers.append("SOT_ON_MODEL_SOURCE_MISSING")
            elif not _registered_source_matches(expected_path, view.get("source"), repo_root):
                blockers.append("SOURCE_PATH_MISMATCH")
            else:
                registered_hash, registered_bytes = _integrity_record(registry, sku, view)
                actual_hash = _sha256_file(expected_path)
                actual_bytes = expected_path.stat().st_size
                if registered_hash != sot_media.get("sha256") or actual_hash != registered_hash:
                    blockers.append("SOURCE_HASH_MISMATCH")
                if registered_bytes != sot_media.get("bytes") or actual_bytes != registered_bytes:
                    blockers.append("SOURCE_BYTES_MISMATCH")
            approval_reference = _approval_reference(media_entry, view)
            if require_explicit_approval and not approval_reference:
                blockers.append("EXPLICIT_APPROVAL_REFERENCE_MISSING")
            elif not approval_reference:
                warnings.append("LEGACY_APPROVAL_REFERENCE_MISSING")
            if not view.get("reviewed_at") and not registry.get("reviewed_at"):
                blockers.append("REVIEW_TIMESTAMP_MISSING")

        results.append(
            {
                "sku": sku,
                "state": "SUPPORTED" if not blockers else "BLOCKED",
                "blockers": blockers,
                "warnings": warnings,
                "registry_status": media_entry.get("status") or "SUPPORTED",
            }
        )

    supported = sum(result["state"] == "SUPPORTED" for result in results)
    return {
        "schema": "skyyrose.on-model-media-intake-report.v1",
        "product_sot_sha256": _sha256_bytes(product_sot_bytes),
        "require_explicit_approval": require_explicit_approval,
        "summary": {"supported": supported, "blocked": len(results) - supported},
        "products": results,
    }
