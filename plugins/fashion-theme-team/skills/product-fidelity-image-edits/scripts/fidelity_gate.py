#!/usr/bin/env python3
"""Fail-closed preflight, reference packing, and output verification for product edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, TypeGuard, cast

import model_registry
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

SCHEMA = "product-fidelity-edit.v2"
LEGACY_SCHEMAS = {"product-fidelity-edit.v1"}
SEMANTIC_REVIEW_SCHEMA = "product-fidelity-semantic-review.v2"
NATIVE_REVIEW_SCHEMA = "product-fidelity-native-review.v2"
SOURCE_AUTHORITY_SCHEMA = "product-fidelity-source-authority.v1"
OPERATIONS = {
    "localized_product_patch",
    "native_collection_scene",
    "protected_scene_composite",
}
REFERENCE_ROLES = {
    "physical_product_authority",
    "approved_techflat",
    "exact_logo_or_patch_authority",
    "protected_product_layer",
    "protected_garment_matte",
    "scene_authority",
    "context_only",
}
PRODUCT_ROLES = REFERENCE_ROLES - {"scene_authority", "context_only"}
VALID_VIEWS = {
    "front",
    "side",
    "back",
    "on_model_front",
    "on_model_side",
    "on_model_back",
    "scene",
    "detail",
}
HEX64 = re.compile(r"^[0-9a-f]{64}$")
NATIVE_ROUTES = {
    "environment_rebuild_around_source",
    "protected_garment_reconstruction",
    "full_native_redraw_candidate",
}
NATIVE_SCORE_DIMENSIONS = {
    "product_fidelity",
    "optical_integration",
    "anatomy_pose",
    "collection_story",
    "commerce_readiness",
}
NATIVE_POLICY_SCORE_FLOORS = {
    "product_fidelity": 95,
    "optical_integration": 90,
    "anatomy_pose": 90,
    "collection_story": 90,
    "commerce_readiness": 90,
}
SEMANTIC_BASE_CHECKS = {"construction", "color_and_material", "placement"}
SEMANTIC_ALLOWED_CHECKS = SEMANTIC_BASE_CHECKS | {
    "logo_or_artwork",
    "verbatim_text",
    "stitching_and_trim",
}
AUTHORITY_STATES = {"founder_approved", "sot_verified"}
MODEL_OPERATION_MAP = {
    "localized_product_patch": "bounded_image_edit",
    "protected_scene_composite": "protected_composite",
    "native_collection_scene": "scene_generation",
}
MODEL_CAPABILITY_MAP = {
    "localized_product_patch": "explicit_mask_edit",
    "protected_scene_composite": "deterministic_composite",
    "native_collection_scene": "multi_reference",
}
MAX_REGISTRY_AGE_DAYS = 30


class GateError(RuntimeError):
    """Raised for malformed inputs that prevent a trustworthy gate result."""

    def __init__(self, message: str, code: str = "GATE_ERROR") -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True)
class Finding:
    level: str
    code: str
    message: str
    path: str | None = None

    def as_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "level": self.level,
            "code": self.code,
            "message": self.message,
        }
        if self.path is not None:
            result["path"] = self.path
        return result


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bytes_snapshot(path: Path, *, code: str = "FILE_UNREADABLE") -> tuple[bytes, str]:
    """Read once and bind every later judgment to the exact bytes returned."""
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise GateError(f"Cannot read {path}: {exc}", code) from exc
    return payload, hashlib.sha256(payload).hexdigest()


def sha256_json(value: Any) -> str:
    """Return a stable digest for a JSON-compatible value."""
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def same_identity(left: Any, right: Any) -> bool:
    """Compare non-empty actor identifiers without case or surrounding-space bypasses."""
    return (
        isinstance(left, str)
        and isinstance(right, str)
        and bool(left.strip())
        and bool(right.strip())
        and left.strip().casefold() == right.strip().casefold()
    )


def is_strict_int(value: Any) -> TypeGuard[int]:
    """Accept JSON integers while rejecting booleans, which subclass int in Python."""
    return isinstance(value, int) and not isinstance(value, bool)


def resolve_path(value: str, workspace: Path) -> Path:
    # Resolve both sides first. On macOS temporary directories may be exposed
    # through both /var and /private/var; comparing an unresolved workspace
    # against a resolved candidate incorrectly turns a valid in-workspace file
    # into a path-escape finding.
    resolved_workspace = workspace.expanduser().resolve()
    candidate = Path(value).expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (resolved_workspace / candidate).resolve()
    )
    try:
        resolved.relative_to(resolved_workspace)
    except ValueError as exc:
        raise GateError(f"Path escapes declared workspace: {value}") from exc
    return resolved


def _decode_json_object(payload: bytes, path: Path) -> dict[str, Any]:
    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise GateError(f"Duplicate JSON object key: {key}", "INVALID_CONTRACT_JSON")
            result[key] = value
        return result

    def reject_non_finite(value: str) -> None:
        raise GateError(
            f"Non-finite JSON number is forbidden: {value}",
            "INVALID_CONTRACT_JSON",
        )

    try:
        data = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=reject_duplicate_keys,
            parse_constant=reject_non_finite,
        )
    except GateError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise GateError(f"Cannot read contract {path}: {exc}", "INVALID_CONTRACT_JSON") from exc
    if not isinstance(data, dict):
        raise GateError("Contract root must be a JSON object", "INVALID_CONTRACT_JSON")
    return data


def load_contract_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    payload, digest = read_bytes_snapshot(path, code="INVALID_CONTRACT_JSON")
    return _decode_json_object(payload, path), digest


def load_contract(path: Path) -> dict[str, Any]:
    return load_contract_snapshot(path)[0]


def load_image_snapshot(path: Path, *, mode: str | None = None) -> tuple[Image.Image, str]:
    """Decode an image from the same immutable byte snapshot that is hashed."""
    payload, digest = read_bytes_snapshot(path, code="IMAGE_UNREADABLE")
    try:
        with Image.open(BytesIO(payload)) as opened:
            opened.load()
            image = ImageOps.exif_transpose(opened).copy()
        if mode is not None:
            image = image.convert(mode)
    except Exception as exc:  # Pillow uses format-specific exception classes.
        raise GateError(f"Cannot decode image {path}: {exc}", "IMAGE_UNREADABLE") from exc
    return image, digest


def image_metadata(path: Path) -> dict[str, Any]:
    image, _ = load_image_snapshot(path)
    return _image_metadata(image)


def _image_metadata(image: Image.Image) -> dict[str, Any]:
    width, height = image.size
    has_alpha = "A" in image.getbands()
    alpha_min = 255
    alpha_max = 255
    transparent_fraction = 0.0
    opaque_fraction = 1.0
    if has_alpha:
        alpha = np.asarray(image.getchannel("A"), dtype=np.uint8)
        alpha_min = int(alpha.min())
        alpha_max = int(alpha.max())
        transparent_fraction = float(np.count_nonzero(alpha == 0) / alpha.size)
        opaque_fraction = float(np.count_nonzero(alpha == 255) / alpha.size)
    return {
        "width": width,
        "height": height,
        "mode": image.mode,
        "has_alpha": has_alpha,
        "alpha_min": alpha_min,
        "alpha_max": alpha_max,
        "transparent_fraction": round(transparent_fraction, 8),
        "opaque_fraction": round(opaque_fraction, 8),
    }


def _validate_hash_entry(
    entry: dict[str, Any], workspace: Path, label: str, findings: list[Finding]
) -> tuple[Path | None, dict[str, Any] | None]:
    raw_path = entry.get("path")
    expected = entry.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        findings.append(Finding("error", "PATH_REQUIRED", f"{label} path is required"))
        return None, None
    try:
        path = resolve_path(raw_path, workspace)
    except GateError as exc:
        findings.append(Finding("error", "PATH_OUTSIDE_WORKSPACE", str(exc), raw_path))
        return None, None
    if not path.is_file():
        findings.append(Finding("error", "FILE_MISSING", f"{label} file is missing", str(path)))
        return path, None
    if not isinstance(expected, str) or not HEX64.fullmatch(expected):
        findings.append(
            Finding(
                "error",
                "SHA256_REQUIRED",
                f"{label} requires a lowercase SHA-256",
                str(path),
            )
        )
        return path, None
    try:
        image, actual = load_image_snapshot(path)
    except GateError as exc:
        findings.append(Finding("error", "IMAGE_UNREADABLE", f"{label}: {exc}", str(path)))
        return path, None
    if actual != expected:
        findings.append(
            Finding(
                "error",
                "SOURCE_HASH_DRIFT",
                f"{label} SHA-256 drift: expected {expected}, got {actual}",
                str(path),
            )
        )
        return path, None
    return path, _image_metadata(image)


def _validate_file_binding(
    entry: Any, workspace: Path, label: str, findings: list[Finding]
) -> tuple[Path | None, str | None]:
    if not isinstance(entry, dict):
        findings.append(Finding("error", "AUTHORITY_BINDING_REQUIRED", f"{label} is required"))
        return None, None
    raw_path = entry.get("path")
    expected = entry.get("sha256")
    if not isinstance(raw_path, str) or not raw_path:
        findings.append(Finding("error", "AUTHORITY_PATH_REQUIRED", f"{label}.path is required"))
        return None, None
    try:
        path = resolve_path(raw_path, workspace)
    except GateError as exc:
        findings.append(Finding("error", "AUTHORITY_PATH_OUTSIDE_WORKSPACE", str(exc), raw_path))
        return None, None
    if not isinstance(expected, str) or not HEX64.fullmatch(expected):
        findings.append(
            Finding(
                "error",
                "AUTHORITY_SHA256_REQUIRED",
                f"{label}.sha256 requires a lowercase SHA-256",
                str(path),
            )
        )
        return path, None
    try:
        _, actual = read_bytes_snapshot(path, code="AUTHORITY_FILE_UNREADABLE")
    except GateError as exc:
        findings.append(Finding("error", exc.code, f"{label}: {exc}", str(path)))
        return path, None
    if actual != expected:
        findings.append(
            Finding(
                "error",
                "AUTHORITY_HASH_DRIFT",
                f"{label} SHA-256 drift: expected {expected}, got {actual}",
                str(path),
            )
        )
        return path, actual
    return path, actual


def _validate_source_authority(
    reference: dict[str, Any],
    workspace: Path,
    index: int,
    findings: list[Finding],
) -> dict[str, Any] | None:
    binding = reference.get("authority_receipt")
    receipt_path, _ = _validate_file_binding(
        binding, workspace, f"reference {index} authority_receipt", findings
    )
    if receipt_path is None or not isinstance(binding, dict):
        return None
    expected_receipt_sha = binding.get("sha256")
    try:
        receipt, receipt_sha = load_contract_snapshot(receipt_path)
    except GateError as exc:
        findings.append(
            Finding(
                "error",
                "AUTHORITY_RECEIPT_INVALID",
                f"reference {index} authority receipt: {exc}",
                str(receipt_path),
            )
        )
        return None
    if receipt_sha != expected_receipt_sha:
        findings.append(
            Finding(
                "error",
                "AUTHORITY_RECEIPT_HASH_DRIFT",
                f"reference {index} authority receipt changed while validating",
                str(receipt_path),
            )
        )
        return None

    exact_fields = {
        "schema": SOURCE_AUTHORITY_SCHEMA,
        "status": "VERIFIED",
        "sku": reference.get("sku"),
        "view": reference.get("view"),
        "role": reference.get("role"),
        "source_sha256": reference.get("sha256"),
    }
    for field, expected in exact_fields.items():
        if receipt.get(field) != expected:
            findings.append(
                Finding(
                    "error",
                    "AUTHORITY_RECEIPT_BINDING_MISMATCH",
                    f"reference {index} authority receipt {field} must equal {expected!r}",
                    str(receipt_path),
                )
            )
    authority_hashes: dict[str, str] = {}
    for field in ("catalog", "sot_manifest", "dossier"):
        _, actual = _validate_file_binding(
            receipt.get(field),
            workspace,
            f"reference {index} authority_receipt.{field}",
            findings,
        )
        if actual is not None:
            authority_hashes[field] = actual
    resolver = receipt.get("resolver")
    if not (
        isinstance(resolver, dict)
        and isinstance(resolver.get("id"), str)
        and resolver["id"].strip()
        and isinstance(resolver.get("version"), str)
        and resolver["version"].strip()
    ):
        findings.append(
            Finding(
                "error",
                "AUTHORITY_RESOLVER_REQUIRED",
                f"reference {index} authority receipt needs resolver id and version",
                str(receipt_path),
            )
        )
    return {
        "path": str(receipt_path),
        "sha256": receipt_sha,
        "schema": receipt.get("schema"),
        "status": receipt.get("status"),
        "resolver": resolver,
        "bound_authority_hashes": authority_hashes,
        "trust_state": "LOCAL_HASH_BOUND_AUTHORITY_RECEIPT",
    }


def _filename_view_conflict(path: Path, view: str) -> bool:
    tokens = {token for token in re.split(r"[^a-z0-9]+", path.stem.lower()) if token}
    declared = view.removeprefix("on_model_")
    for candidate in {"front", "side", "back"}:
        if candidate in tokens and candidate != declared:
            return True
    return False


def _validate_edit_region(region: Any, size: tuple[int, int], findings: list[Finding]) -> None:
    if not isinstance(region, dict):
        findings.append(
            Finding("error", "EDIT_REGION_REQUIRED", "Localized patches require edit_region")
        )
        return
    if region.get("coordinate_space", "pixels") != "pixels":
        findings.append(
            Finding(
                "error",
                "COORDINATE_SPACE_UNSUPPORTED",
                "Only pixel coordinates are supported",
            )
        )
    width, height = size
    feather = region.get("feather_px", 0)
    if not is_strict_int(feather) or not 0 <= feather <= min(width, height) // 4:
        findings.append(
            Finding(
                "error",
                "EDIT_REGION_FEATHER_INVALID",
                "feather_px must be a non-negative integer no larger than one quarter of the image",
            )
        )
    kind = region.get("type")
    if kind == "bbox":
        bbox = region.get("bbox")
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(is_strict_int(value) for value in bbox)
        ):
            findings.append(Finding("error", "BBOX_INVALID", "bbox must contain four integers"))
            return
        left, top, right, bottom = bbox
        if not (0 <= left < right <= width and 0 <= top < bottom <= height):
            findings.append(Finding("error", "BBOX_OUT_OF_BOUNDS", "bbox exceeds target geometry"))
    elif kind == "polygon":
        points = region.get("points")
        if not (
            isinstance(points, list)
            and len(points) >= 3
            and all(
                isinstance(point, list)
                and len(point) == 2
                and all(is_strict_int(value) for value in point)
                for point in points
            )
        ):
            findings.append(
                Finding(
                    "error",
                    "POLYGON_INVALID",
                    "polygon requires at least three integer points",
                )
            )
            return
        if any(not (0 <= x < width and 0 <= y < height) for x, y in points):
            findings.append(
                Finding("error", "POLYGON_OUT_OF_BOUNDS", "polygon exceeds target geometry")
            )
    else:
        findings.append(Finding("error", "EDIT_REGION_TYPE_INVALID", "Use polygon or bbox"))


def _valid_box(value: Any, width: int, height: int) -> bool:
    if not (
        isinstance(value, list) and len(value) == 4 and all(is_strict_int(item) for item in value)
    ):
        return False
    left, top, right, bottom = value
    return 0 <= left < right <= width and 0 <= top < bottom <= height


def _validate_collection_scene(value: Any, findings: list[Finding]) -> set[str]:
    if not isinstance(value, dict):
        findings.append(
            Finding(
                "error",
                "COLLECTION_SCENE_REQUIRED",
                "Native scenes require collection_scene",
            )
        )
        return set()
    for field in ("collection_id", "scene_id", "story_role", "scene_variation"):
        if not isinstance(value.get(field), str) or not value[field].strip():
            findings.append(
                Finding(
                    "error",
                    "COLLECTION_SCENE_FIELD_REQUIRED",
                    f"collection_scene.{field} is required",
                )
            )
    products = value.get("products")
    if not isinstance(products, list) or not products:
        findings.append(
            Finding(
                "error",
                "COLLECTION_PRODUCTS_REQUIRED",
                "Native scenes require an exact product/CTA cast",
            )
        )
        return set()
    skus: list[str] = []
    for index, product in enumerate(products):
        if not isinstance(product, dict):
            findings.append(
                Finding(
                    "error",
                    "COLLECTION_PRODUCT_INVALID",
                    f"collection product {index} must be an object",
                )
            )
            continue
        sku = product.get("sku")
        cta = product.get("cta")
        if not isinstance(sku, str) or not sku.strip():
            findings.append(
                Finding(
                    "error",
                    "COLLECTION_PRODUCT_SKU_REQUIRED",
                    f"collection product {index} needs sku",
                )
            )
        else:
            skus.append(sku.strip())
        if not isinstance(cta, str) or not cta.strip():
            findings.append(
                Finding(
                    "error",
                    "COLLECTION_PRODUCT_CTA_REQUIRED",
                    f"collection product {index} needs cta",
                )
            )
    if len(skus) != len(set(skus)):
        findings.append(
            Finding(
                "error",
                "COLLECTION_PRODUCT_DUPLICATE",
                "Each SKU may appear only once in a scene cast",
            )
        )
    return set(skus)


def _validate_optical_contract(value: Any, findings: list[Finding]) -> tuple[int, int] | None:
    if not isinstance(value, dict):
        findings.append(
            Finding(
                "error",
                "OPTICAL_CONTRACT_REQUIRED",
                "Native scenes require a measured optical_contract",
            )
        )
        return None
    dimensions = value.get("output_dimensions")
    if not (
        isinstance(dimensions, list)
        and len(dimensions) == 2
        and all(is_strict_int(item) and item > 0 for item in dimensions)
    ):
        findings.append(
            Finding(
                "error",
                "OUTPUT_DIMENSIONS_INVALID",
                "optical_contract.output_dimensions needs two positive integers",
            )
        )
        return None
    width = cast(int, dimensions[0])
    height = cast(int, dimensions[1])
    if not _valid_box(value.get("subject_bbox"), width, height):
        findings.append(
            Finding(
                "error",
                "SUBJECT_BBOX_INVALID",
                "subject_bbox must be inside output geometry",
            )
        )
    horizon = value.get("horizon_y")
    if not is_strict_int(horizon) or not 0 <= horizon < height:
        findings.append(
            Finding("error", "HORIZON_INVALID", "horizon_y must be inside output geometry")
        )
    contacts = value.get("foot_contact_points")
    if not (
        isinstance(contacts, list)
        and len(contacts) >= 2
        and all(
            isinstance(point, list)
            and len(point) == 2
            and all(is_strict_int(item) for item in point)
            and 0 <= point[0] < width
            and 0 <= point[1] < height
            for point in contacts
        )
    ):
        findings.append(
            Finding(
                "error",
                "FOOT_CONTACTS_INVALID",
                "At least two in-bounds foot/contact points are required",
            )
        )
    if value.get("lens_class") not in {"wide", "normal", "telephoto"}:
        findings.append(Finding("error", "LENS_CLASS_INVALID", "Use wide, normal, or telephoto"))
    if value.get("camera_height") not in {"low", "eye_level", "high"}:
        findings.append(Finding("error", "CAMERA_HEIGHT_INVALID", "Use low, eye_level, or high"))
    for field in ("key_light", "shadow"):
        record = value.get(field)
        if not isinstance(record, dict):
            findings.append(
                Finding(
                    "error",
                    "LIGHT_CONTRACT_REQUIRED",
                    f"optical_contract.{field} is required",
                )
            )
            continue
        if not isinstance(record.get("direction"), str) or not record["direction"].strip():
            findings.append(
                Finding(
                    "error",
                    "LIGHT_DIRECTION_REQUIRED",
                    f"{field}.direction is required",
                )
            )
        if record.get("softness") not in {"hard", "medium", "soft"}:
            findings.append(
                Finding("error", "LIGHT_SOFTNESS_INVALID", f"{field}.softness is invalid")
            )
        if field == "key_light":
            temperature = record.get("temperature_k")
            if not is_strict_int(temperature) or not 1500 <= temperature <= 15000:
                findings.append(
                    Finding(
                        "error",
                        "COLOR_TEMPERATURE_INVALID",
                        "key_light.temperature_k must be 1500..15000",
                    )
                )
    if value.get("floor_response") not in {
        "matte",
        "satin",
        "polished",
        "wet",
        "textured",
    }:
        findings.append(Finding("error", "FLOOR_RESPONSE_INVALID", "floor_response is invalid"))
    if value.get("reflection_mode") not in {"none", "diffuse", "subtle", "mirror"}:
        findings.append(Finding("error", "REFLECTION_MODE_INVALID", "reflection_mode is invalid"))
    for field in (
        "contact_shadow_required",
        "edge_spill_required",
        "depth_occlusion_required",
        "atmospheric_depth_required",
        "scene_grain_match_required",
    ):
        if value.get(field) is not True:
            findings.append(
                Finding(
                    "error",
                    "NATIVE_INTEGRATION_REQUIREMENT_MISSING",
                    f"optical_contract.{field} must be true",
                )
            )
    return width, height


def _validate_review_gate(value: Any, findings: list[Finding]) -> None:
    if not isinstance(value, dict):
        findings.append(
            Finding("error", "REVIEW_GATE_REQUIRED", "Native scenes require review_gate")
        )
        return
    if not isinstance(value.get("candidate_author"), str) or not value["candidate_author"].strip():
        findings.append(
            Finding(
                "error",
                "CANDIDATE_AUTHOR_REQUIRED",
                "review_gate.candidate_author is required",
            )
        )
    if value.get("independent_reviewer_required") is not True:
        findings.append(
            Finding(
                "error",
                "INDEPENDENT_REVIEW_REQUIRED",
                "An independent reviewer is mandatory",
            )
        )
    scores = value.get("minimum_scores")
    if not isinstance(scores, dict):
        findings.append(
            Finding(
                "error",
                "MINIMUM_SCORES_REQUIRED",
                "review_gate.minimum_scores is required",
            )
        )
        return
    if set(scores) != NATIVE_SCORE_DIMENSIONS:
        findings.append(
            Finding(
                "error",
                "SCORE_DIMENSIONS_INVALID",
                f"minimum_scores must contain {sorted(NATIVE_SCORE_DIMENSIONS)}",
            )
        )
    for dimension in NATIVE_SCORE_DIMENSIONS:
        score = scores.get(dimension)
        if not is_strict_int(score) or not 0 <= score <= 100:
            findings.append(
                Finding(
                    "error",
                    "MINIMUM_SCORE_INVALID",
                    f"minimum_scores.{dimension} must be an integer 0..100",
                )
            )
        elif score < NATIVE_POLICY_SCORE_FLOORS[dimension]:
            findings.append(
                Finding(
                    "error",
                    "MINIMUM_SCORE_BELOW_POLICY_FLOOR",
                    f"minimum_scores.{dimension} must be at least {NATIVE_POLICY_SCORE_FLOORS[dimension]}",
                )
            )


def _validate_promotion(value: Any, findings: list[Finding]) -> None:
    if not isinstance(value, dict):
        findings.append(
            Finding(
                "error",
                "PROMOTION_BOUNDARY_REQUIRED",
                "Native scenes require promotion",
            )
        )
        return
    if value.get("founder_approval_required") is not True:
        findings.append(
            Finding(
                "error",
                "FOUNDER_APPROVAL_REQUIRED",
                "Founder approval must remain required",
            )
        )
    if value.get("wiring_allowed") is not False:
        findings.append(
            Finding(
                "error",
                "PREMATURE_WIRING_FORBIDDEN",
                "Pre-generation wiring_allowed must be false",
            )
        )
    if value.get("deployment_allowed") is not False:
        findings.append(
            Finding(
                "error",
                "PREMATURE_DEPLOYMENT_FORBIDDEN",
                "Pre-generation deployment_allowed must be false",
            )
        )


def _validate_model_capability(
    generator: dict[str, Any], operation: Any, findings: list[Finding]
) -> dict[str, Any]:
    receipt: dict[str, Any] = {
        "status": "BLOCKED",
        "generation_authorized": False,
        "live_availability_verified": False,
    }
    model_id = generator.get("model_id")
    if not isinstance(model_id, str) or not model_id.strip():
        findings.append(
            Finding(
                "error",
                "MODEL_ID_REQUIRED",
                "generator.model_id must name a reviewed registry entry",
            )
        )
        receipt["code"] = "MODEL_ID_REQUIRED"
        return receipt
    if operation not in MODEL_OPERATION_MAP:
        receipt["code"] = "OPERATION_INVALID"
        return receipt
    try:
        registry = model_registry.load_registry(model_registry.DEFAULT_REGISTRY)
        registry_sha = sha256_file(model_registry.DEFAULT_REGISTRY)
        verified_on = date.fromisoformat(registry["verified_on"])
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        findings.append(
            Finding("error", "MODEL_REGISTRY_INVALID", f"Model registry is invalid: {exc}")
        )
        receipt["code"] = "MODEL_REGISTRY_INVALID"
        return receipt

    age_days = (date.today() - verified_on).days
    receipt.update(
        {
            "registry_path": str(model_registry.DEFAULT_REGISTRY),
            "registry_sha256": registry_sha,
            "registry_verified_on": registry["verified_on"],
            "registry_age_days": age_days,
            "model_requested": model_id,
        }
    )
    if age_days < 0 or age_days > MAX_REGISTRY_AGE_DAYS:
        findings.append(
            Finding(
                "error",
                "MODEL_REGISTRY_STALE",
                f"Model registry age {age_days} days is outside 0..{MAX_REGISTRY_AGE_DAYS}",
            )
        )
        receipt["code"] = "MODEL_REGISTRY_STALE"
        return receipt

    operation_id = MODEL_OPERATION_MAP[operation]
    registry_result = model_registry.validate_request(registry, model_id, operation_id)
    receipt["registry_result"] = registry_result
    model = model_registry.index_registry(registry).get(model_id.casefold())
    required_capability = MODEL_CAPABILITY_MAP[operation]
    if (
        registry_result.get("status") != "PASS"
        or model is None
        or required_capability not in model["capabilities"]
        or model["status"] == "deprecated_or_platform_specific"
    ):
        findings.append(
            Finding(
                "error",
                "MODEL_CAPABILITY_BLOCKED",
                f"Registry does not authorize {model_id} for {operation_id} with {required_capability}",
            )
        )
        receipt["code"] = "MODEL_CAPABILITY_BLOCKED"
        return receipt

    receipt.update(
        {
            "status": "PASS",
            "code": "REGISTERED_CAPABILITY_MATCH",
            "model": model["id"],
            "provider": model["provider"],
            "required_capability": required_capability,
            "registry_status": model["status"],
            "required_next_step": "Bind a fresh live model-discovery receipt and obtain explicit generation approval",
        }
    )
    return receipt


def _validate_protected_placements(
    value: Any,
    protected_entries: list[tuple[dict[str, Any], dict[str, Any]]],
    workspace: Path,
    output_size: tuple[int, int] | None,
    findings: list[Finding],
) -> None:
    if not isinstance(value, list) or not value:
        findings.append(
            Finding(
                "error",
                "PROTECTED_PLACEMENTS_REQUIRED",
                "Every protected layer requires an exact placement",
            )
        )
        return

    allowed: dict[Path, dict[str, Any]] = {}
    for reference, metadata in protected_entries:
        raw_path = reference.get("path")
        if not isinstance(raw_path, str):
            continue
        try:
            allowed[resolve_path(raw_path, workspace)] = metadata
        except GateError:
            continue

    seen: set[Path] = set()
    for index, placement in enumerate(value):
        if not isinstance(placement, dict):
            findings.append(
                Finding(
                    "error",
                    "PROTECTED_PLACEMENT_INVALID",
                    f"protected placement {index} must be an object",
                )
            )
            continue
        raw_path = placement.get("path")
        if not isinstance(raw_path, str) or not raw_path:
            findings.append(
                Finding(
                    "error",
                    "PROTECTED_PLACEMENT_PATH_REQUIRED",
                    f"protected placement {index} path is required",
                )
            )
            continue
        try:
            resolved = resolve_path(raw_path, workspace)
        except GateError as exc:
            findings.append(
                Finding(
                    "error",
                    "PATH_OUTSIDE_WORKSPACE",
                    str(exc),
                    raw_path,
                )
            )
            continue
        if resolved not in allowed:
            findings.append(
                Finding(
                    "error",
                    "PROTECTED_PLACEMENT_REFERENCE_MISMATCH",
                    "Placement path must name a hash-verified protected reference",
                    str(resolved),
                )
            )
            continue
        if resolved in seen:
            findings.append(
                Finding(
                    "error",
                    "PROTECTED_PLACEMENT_DUPLICATE",
                    "Each protected reference may be placed only once",
                    str(resolved),
                )
            )
        seen.add(resolved)
        x = placement.get("x")
        y = placement.get("y")
        if (
            not isinstance(x, int)
            or isinstance(x, bool)
            or not isinstance(y, int)
            or isinstance(y, bool)
            or x < 0
            or y < 0
        ):
            findings.append(
                Finding(
                    "error",
                    "PROTECTED_PLACEMENT_COORDINATES_INVALID",
                    "Placement x/y must be non-negative integers",
                    str(resolved),
                )
            )
            continue
        if output_size is not None:
            metadata = allowed[resolved]
            output_width, output_height = output_size
            if x + metadata["width"] > output_width or y + metadata["height"] > output_height:
                findings.append(
                    Finding(
                        "error",
                        "PROTECTED_PLACEMENT_OUT_OF_BOUNDS",
                        "Protected placement exceeds declared output geometry",
                        str(resolved),
                    )
                )

    missing = sorted(str(path) for path in set(allowed) - seen)
    if missing:
        findings.append(
            Finding(
                "error",
                "PROTECTED_PLACEMENT_MISSING",
                "Missing placements for: " + ", ".join(missing),
            )
        )


def validate_contract(contract: dict[str, Any], workspace: Path) -> dict[str, Any]:
    findings: list[Finding] = []
    if contract.get("schema") != SCHEMA:
        if contract.get("schema") in LEGACY_SCHEMAS:
            findings.append(
                Finding(
                    "error",
                    "SCHEMA_MIGRATION_REQUIRED",
                    f"Legacy {contract.get('schema')} is blocked; migrate explicitly to {SCHEMA}",
                )
            )
        else:
            findings.append(Finding("error", "SCHEMA_INVALID", f"schema must be {SCHEMA}"))
    operation = contract.get("operation")
    if operation not in OPERATIONS:
        findings.append(
            Finding(
                "error",
                "OPERATION_INVALID",
                f"operation must be one of {sorted(OPERATIONS)}",
            )
        )

    target = contract.get("target")
    target_path: Path | None = None
    target_meta: dict[str, Any] | None = None
    if isinstance(target, dict):
        target_path, target_meta = _validate_hash_entry(target, workspace, "target", findings)
    else:
        findings.append(Finding("error", "TARGET_REQUIRED", "target object is required"))

    generator = contract.get("generator")
    if not isinstance(generator, dict):
        findings.append(Finding("error", "GENERATOR_REQUIRED", "generator object is required"))
        generator = {}
    model_capability = _validate_model_capability(generator, operation, findings)
    canvas_px = contract.get("reference_canvas_px", 1536)
    if not is_strict_int(canvas_px) or not 512 <= canvas_px <= 4096:
        findings.append(
            Finding(
                "error",
                "REFERENCE_CANVAS_SIZE_INVALID",
                "reference_canvas_px must be an integer from 512 to 4096",
            )
        )

    references = contract.get("references")
    if not isinstance(references, list) or not references:
        findings.append(
            Finding("error", "REFERENCES_REQUIRED", "At least one reference is required")
        )
        references = []

    reference_report: list[dict[str, Any]] = []
    physical_authorities = 0
    protected_layers: list[tuple[dict[str, Any], dict[str, Any]]] = []
    garment_mattes: list[tuple[dict[str, Any], dict[str, Any]]] = []
    authoritative_skus: set[str] = set()
    truth_source_skus: set[str] = set()
    pose_authorities: list[dict[str, Any]] = []
    for index, reference in enumerate(references):
        if not isinstance(reference, dict):
            findings.append(
                Finding("error", "REFERENCE_INVALID", f"reference {index} is not an object")
            )
            continue
        role = reference.get("role")
        view = reference.get("view")
        sku = reference.get("sku")
        if role not in REFERENCE_ROLES:
            findings.append(
                Finding(
                    "error",
                    "REFERENCE_ROLE_INVALID",
                    f"reference {index} role is invalid",
                )
            )
        if view not in VALID_VIEWS:
            findings.append(
                Finding(
                    "error",
                    "REFERENCE_VIEW_INVALID",
                    f"reference {index} view is invalid",
                )
            )
        if not isinstance(sku, str) or not sku.strip():
            findings.append(
                Finding(
                    "error",
                    "REFERENCE_SKU_REQUIRED",
                    f"reference {index} sku is required",
                )
            )
        verbatim_text = reference.get("verbatim_text")
        if role in PRODUCT_ROLES and not (
            isinstance(verbatim_text, list)
            and all(isinstance(item, str) and item.strip() for item in verbatim_text)
        ):
            findings.append(
                Finding(
                    "error",
                    "VERBATIM_TEXT_DECLARATION_REQUIRED",
                    f"reference {index} must declare verbatim_text explicitly; use [] only after confirming no wording",
                )
            )
        path, metadata = _validate_hash_entry(reference, workspace, f"reference {index}", findings)
        authority_report = (
            _validate_source_authority(reference, workspace, index, findings)
            if role in PRODUCT_ROLES
            else None
        )
        if path is not None and isinstance(view, str) and _filename_view_conflict(path, view):
            findings.append(
                Finding(
                    "error",
                    "FILENAME_VIEW_CONFLICT",
                    f"Filename conflicts with declared {view} view; visually classify and rename the source before preflight",
                    str(path),
                )
            )
        contains_skus = reference.get("contains_skus")
        if role == "physical_product_authority":
            physical_authorities += 1
            if isinstance(sku, str) and sku.strip():
                authoritative_skus.add(sku.strip())
                truth_source_skus.add(sku.strip())
            pose_authorities.append(reference)
            if isinstance(contains_skus, list) and len(contains_skus) > 1:
                findings.append(
                    Finding(
                        "error",
                        "MULTI_SKU_AUTHORITY_FORBIDDEN",
                        "A multi-SKU board cannot be the physical authority for an exact patch",
                        str(path) if path else None,
                    )
                )
        if role == "protected_product_layer" and metadata is not None:
            protected_layers.append((reference, metadata))
            if not metadata["has_alpha"] or metadata["transparent_fraction"] <= 0:
                findings.append(
                    Finding(
                        "error",
                        "PROTECTED_LAYER_REAL_ALPHA_REQUIRED",
                        "Protected layers require genuine transparent pixels",
                        str(path) if path else None,
                    )
                )
            if metadata["opaque_fraction"] <= 0:
                findings.append(
                    Finding(
                        "error",
                        "PROTECTED_LAYER_OPAQUE_PIXELS_REQUIRED",
                        "Protected layers require at least one fully opaque pixel",
                        str(path) if path else None,
                    )
                )
        if role == "protected_garment_matte" and metadata is not None:
            garment_mattes.append((reference, metadata))
            if isinstance(sku, str) and sku.strip():
                authoritative_skus.add(sku.strip())
            pose_authorities.append(reference)
            if not metadata["has_alpha"] or metadata["transparent_fraction"] <= 0:
                findings.append(
                    Finding(
                        "error",
                        "GARMENT_MATTE_REAL_ALPHA_REQUIRED",
                        "Protected garment mattes require genuine transparent pixels",
                        str(path) if path else None,
                    )
                )
            if metadata["opaque_fraction"] <= 0:
                findings.append(
                    Finding(
                        "error",
                        "GARMENT_MATTE_OPAQUE_PIXELS_REQUIRED",
                        "Protected garment mattes require at least one fully opaque pixel",
                        str(path) if path else None,
                    )
                )
        if role in {"approved_techflat", "exact_logo_or_patch_authority"}:
            if isinstance(sku, str) and sku.strip():
                authoritative_skus.add(sku.strip())
                if role == "approved_techflat":
                    truth_source_skus.add(sku.strip())
        reference_report.append(
            {
                "index": index,
                "path": str(path) if path else reference.get("path"),
                "role": role,
                "view": view,
                "sku": sku,
                "authority_state": reference.get("authority_state"),
                "source_authority": authority_report,
                "metadata": metadata,
            }
        )

    if operation in {
        "protected_scene_composite",
        "native_collection_scene",
    } and not isinstance(contract.get("pose_change"), bool):
        findings.append(
            Finding(
                "error",
                "POSE_CHANGE_FLAG_INVALID",
                "pose_change must be an explicit JSON boolean",
            )
        )

    if operation == "localized_product_patch" and target_meta is not None:
        if generator.get("route") not in {"masked_edit", "deterministic_composite"}:
            findings.append(
                Finding(
                    "error",
                    "MASKLESS_GENERATOR_FOR_LOCAL_EDIT",
                    "Localized product patches require masked_edit or deterministic_composite",
                )
            )
        if generator.get("supports_explicit_mask") is not True:
            findings.append(
                Finding(
                    "error",
                    "EXPLICIT_MASK_CAPABILITY_REQUIRED",
                    "Selected generator must consume an explicit mask",
                )
            )
        _validate_edit_region(
            contract.get("edit_region"),
            (target_meta["width"], target_meta["height"]),
            findings,
        )
        if physical_authorities == 0:
            findings.append(
                Finding(
                    "error",
                    "PHYSICAL_AUTHORITY_REQUIRED",
                    "Localized product patches require a physical_product_authority",
                )
            )
        verification = contract.get("verification", {})
        if not isinstance(verification, dict):
            findings.append(
                Finding("error", "VERIFICATION_INVALID", "verification must be an object")
            )
            verification = {}
        max_outside = verification.get("max_outside_changed_pixels", 0)
        if not isinstance(max_outside, int) or isinstance(max_outside, bool) or max_outside < 0:
            findings.append(
                Finding(
                    "error",
                    "MAX_OUTSIDE_CHANGED_PIXELS_INVALID",
                    "verification.max_outside_changed_pixels must be a non-negative integer",
                )
            )
        elif max_outside != 0:
            findings.append(
                Finding(
                    "error",
                    "OUTSIDE_CHANGE_TOLERANCE_FORBIDDEN",
                    "Exact product patches require zero outside-mask changes",
                )
            )
        per_channel_tolerance = verification.get("per_channel_tolerance", 0)
        if (
            not isinstance(per_channel_tolerance, int)
            or isinstance(per_channel_tolerance, bool)
            or not 0 <= per_channel_tolerance <= 255
        ):
            findings.append(
                Finding(
                    "error",
                    "PER_CHANNEL_TOLERANCE_INVALID",
                    "verification.per_channel_tolerance must be an integer from 0 to 255",
                )
            )
        elif per_channel_tolerance != 0:
            findings.append(
                Finding(
                    "error",
                    "PER_CHANNEL_TOLERANCE_FORBIDDEN",
                    "Exact product patches require byte-exact channel comparison",
                )
            )
        min_inside = verification.get("min_inside_changed_pixels", 1)
        if not isinstance(min_inside, int) or isinstance(min_inside, bool) or min_inside < 1:
            findings.append(
                Finding(
                    "error",
                    "MIN_INSIDE_CHANGED_PIXELS_INVALID",
                    "verification.min_inside_changed_pixels must be a positive integer",
                )
            )
        review = contract.get("semantic_review")
        if not isinstance(review, dict) or review.get("required") is not True:
            findings.append(
                Finding(
                    "error",
                    "SEMANTIC_REVIEW_REQUIRED",
                    "A hash-bound logo/construction review receipt is required after pixel verification",
                )
            )
        else:
            if not isinstance(review.get("scope"), str) or not review["scope"].strip():
                findings.append(
                    Finding(
                        "error",
                        "SEMANTIC_REVIEW_SCOPE_REQUIRED",
                        "semantic_review.scope is required",
                    )
                )
            if (
                not isinstance(review.get("candidate_author"), str)
                or not review["candidate_author"].strip()
            ):
                findings.append(
                    Finding(
                        "error",
                        "CANDIDATE_AUTHOR_REQUIRED",
                        "semantic_review.candidate_author is required for independent review",
                    )
                )
            required_checks = review.get("required_checks")
            checks_valid = (
                isinstance(required_checks, list)
                and bool(required_checks)
                and all(isinstance(item, str) for item in required_checks)
                and len(required_checks) == len(set(required_checks))
                and set(required_checks).issubset(SEMANTIC_ALLOWED_CHECKS)
                and SEMANTIC_BASE_CHECKS.issubset(required_checks)
            )
            expected_text: list[str] = []
            for reference in references:
                if not isinstance(reference, dict):
                    continue
                verbatim = reference.get("verbatim_text")
                if verbatim is None:
                    continue
                if not (
                    isinstance(verbatim, list)
                    and all(isinstance(item, str) and item.strip() for item in verbatim)
                ):
                    findings.append(
                        Finding(
                            "error",
                            "VERBATIM_TEXT_INVALID",
                            "reference.verbatim_text must be a non-empty list of exact strings",
                        )
                    )
                else:
                    expected_text.extend(verbatim)
            if expected_text and (
                not isinstance(required_checks, list) or "verbatim_text" not in required_checks
            ):
                checks_valid = False
            if any(
                isinstance(reference, dict)
                and reference.get("role") == "exact_logo_or_patch_authority"
                for reference in references
            ) and (
                not isinstance(required_checks, list) or "logo_or_artwork" not in required_checks
            ):
                checks_valid = False
            if not checks_valid:
                findings.append(
                    Finding(
                        "error",
                        "SEMANTIC_REQUIRED_CHECKS_INVALID",
                        "semantic_review.required_checks must preserve policy checks and declared text/logo truth",
                    )
                )

    if operation == "protected_scene_composite":
        if generator.get("route") != "background_only_then_composite":
            findings.append(
                Finding(
                    "error",
                    "PROTECTED_COMPOSITE_ROUTE_REQUIRED",
                    "Protected scenes must generate the background first and composite protected pixels afterward",
                )
            )
        if not protected_layers:
            findings.append(
                Finding(
                    "error",
                    "PROTECTED_LAYER_REQUIRED",
                    "At least one protected_product_layer is required",
                )
            )
        _validate_protected_placements(
            contract.get("protected_placements"),
            protected_layers,
            workspace,
            ((target_meta["width"], target_meta["height"]) if target_meta is not None else None),
            findings,
        )
        verification = contract.get("verification")
        if (
            not isinstance(verification, dict)
            or verification.get("protected_rgb_must_match") is not True
        ):
            findings.append(
                Finding(
                    "error",
                    "PROTECTED_RGB_MATCH_REQUIRED",
                    "verification.protected_rgb_must_match must be true",
                )
            )
        if contract.get("pose_change") is True:
            missing_pose = [
                reference.get("path")
                for reference, _ in protected_layers
                if reference.get("approved_pose_source") is not True
            ]
            if missing_pose:
                findings.append(
                    Finding(
                        "error",
                        "POSE_CHANGE_WITHOUT_APPROVED_SOURCE",
                        "Exact-product pose changes require an approved product source already in the requested pose",
                    )
                )

    if operation == "native_collection_scene":
        route = generator.get("route")
        if route not in NATIVE_ROUTES:
            findings.append(
                Finding(
                    "error",
                    "NATIVE_ROUTE_INVALID",
                    f"Native scene route must be one of {sorted(NATIVE_ROUTES)}",
                )
            )
        if generator.get("supports_reference_images") is not True:
            findings.append(
                Finding(
                    "error",
                    "REFERENCE_IMAGE_CAPABILITY_REQUIRED",
                    "Native scenes require a reference-image-capable route",
                )
            )
        redraw_allowed = generator.get("product_redraw_allowed") is True
        if route == "full_native_redraw_candidate" and not redraw_allowed:
            findings.append(
                Finding(
                    "error",
                    "REDRAW_AUTHORIZATION_REQUIRED",
                    "full_native_redraw_candidate requires product_redraw_allowed: true",
                )
            )
        if route != "full_native_redraw_candidate" and redraw_allowed:
            findings.append(
                Finding(
                    "error",
                    "REDRAW_SCOPE_CONFLICT",
                    "Exact-product native routes must keep product_redraw_allowed false",
                )
            )
        cast_skus = _validate_collection_scene(contract.get("collection_scene"), findings)
        native_dimensions = _validate_optical_contract(contract.get("optical_contract"), findings)
        _validate_review_gate(contract.get("review_gate"), findings)
        _validate_promotion(contract.get("promotion"), findings)
        for reference in references:
            if not isinstance(reference, dict) or reference.get("role") == "context_only":
                continue
            if reference.get("authority_state") not in AUTHORITY_STATES:
                findings.append(
                    Finding(
                        "error",
                        "REFERENCE_AUTHORITY_STATE_REQUIRED",
                        "Native-scene product references must be founder_approved or sot_verified",
                        (reference.get("path") if isinstance(reference.get("path"), str) else None),
                    )
                )
        missing_skus = sorted(cast_skus - authoritative_skus)
        if missing_skus:
            findings.append(
                Finding(
                    "error",
                    "CAST_SKU_AUTHORITY_MISSING",
                    f"Scene cast lacks product authority for: {', '.join(missing_skus)}",
                )
            )
        missing_truth_sources = sorted(cast_skus - truth_source_skus)
        if missing_truth_sources:
            findings.append(
                Finding(
                    "error",
                    "CAST_SKU_PRODUCT_TRUTH_MISSING",
                    "Each scene SKU needs physical_product_authority or approved_techflat: "
                    + ", ".join(missing_truth_sources),
                )
            )
        target_is_approved_on_model = any(
            report.get("role") == "physical_product_authority"
            and report.get("path") == str(target_path)
            and report.get("view") in {"on_model_front", "on_model_side", "on_model_back"}
            and report.get("authority_state") in AUTHORITY_STATES
            for report in reference_report
        )
        if not target_is_approved_on_model:
            findings.append(
                Finding(
                    "error",
                    "APPROVED_ON_MODEL_TARGET_REQUIRED",
                    "Native scenes must rebuild around the hash-identical approved on-model target",
                    str(target_path) if target_path else None,
                )
            )
        if not redraw_allowed:
            matte_skus = {
                reference.get("sku")
                for reference, _ in garment_mattes
                if isinstance(reference.get("sku"), str)
            }
            missing_mattes = sorted(cast_skus - matte_skus)
            if missing_mattes:
                findings.append(
                    Finding(
                        "error",
                        "GARMENT_MATTE_REQUIRED",
                        f"Exact native scenes require a protected garment matte for: {', '.join(missing_mattes)}",
                    )
                )
            _validate_protected_placements(
                contract.get("protected_placements"),
                garment_mattes,
                workspace,
                native_dimensions,
                findings,
            )
            placements = contract.get("protected_placements")
            placement_paths = (
                {item.get("path") for item in placements if isinstance(item, dict)}
                if isinstance(placements, list)
                else set()
            )
            matte_paths = {reference.get("path") for reference, _ in garment_mattes}
            if not placements or not matte_paths.issubset(placement_paths):
                findings.append(
                    Finding(
                        "error",
                        "GARMENT_PLACEMENTS_REQUIRED",
                        "Every protected garment matte needs an exact protected_placements entry",
                    )
                )
        if contract.get("pose_change") is True:
            missing_pose = [
                reference.get("path")
                for reference in pose_authorities
                if reference.get("approved_pose_source") is not True
            ]
            if missing_pose or not pose_authorities:
                findings.append(
                    Finding(
                        "error",
                        "POSE_CHANGE_WITHOUT_APPROVED_SOURCE",
                        "Native pose changes require approved same-pose product authority for every protected garment",
                    )
                )

    errors = [finding for finding in findings if finding.level == "error"]
    return {
        "schema": "product-fidelity-preflight-receipt.v2",
        "status": "PASS" if not errors else "BLOCKED",
        "contract_sha256": sha256_json(contract),
        "operation": operation,
        "target": {
            "path": str(target_path) if target_path else None,
            "metadata": target_meta,
        },
        "references": reference_report,
        "model_capability": model_capability,
        "generation_authorized": False,
        "findings": [finding.as_dict() for finding in findings],
    }


def build_edit_mask(
    contract: dict[str, Any],
    target_size: tuple[int, int],
    *,
    include_feather: bool = True,
) -> Image.Image:
    region = contract["edit_region"]
    mask = Image.new("L", target_size, 0)
    draw = ImageDraw.Draw(mask)
    if region["type"] == "bbox":
        left, top, right, bottom = region["bbox"]
        draw.rectangle((left, top, right - 1, bottom - 1), fill=255)
    else:
        draw.polygon([tuple(point) for point in region["points"]], fill=255)
    feather = int(region.get("feather_px", 0))
    if include_feather and feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask


def _content_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    alpha_extrema = cast(tuple[int, int], alpha.getextrema())
    if alpha_extrema[0] < 255:
        bbox = alpha.getbbox()
        return bbox if bbox is not None else (0, 0, image.width, image.height)
    rgb = np.asarray(rgba.convert("RGB"), dtype=np.uint8)
    foreground = np.any(rgb < 247, axis=2)
    if not np.any(foreground):
        return (0, 0, image.width, image.height)
    ys, xs = np.nonzero(foreground)
    pad = 16
    return (
        max(0, int(xs.min()) - pad),
        max(0, int(ys.min()) - pad),
        min(image.width, int(xs.max()) + pad + 1),
        min(image.height, int(ys.max()) + pad + 1),
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "unknown"


def optimize_references(
    contract: dict[str, Any], contract_path: Path, workspace: Path, out_dir: Path
) -> dict[str, Any]:
    preflight = validate_contract(contract, workspace)
    if preflight["status"] != "PASS":
        raise GateError("Contract is BLOCKED; reference optimization cannot authorize generation")
    try:
        out_dir = resolve_path(str(out_dir), workspace)
    except GateError as exc:
        raise GateError(str(exc), "OUT_DIR_OUTSIDE_WORKSPACE") from exc
    if out_dir.exists():
        raise GateError(
            f"Reference-pack output already exists: {out_dir}",
            "OUT_DIR_ALREADY_EXISTS",
        )
    out_dir.mkdir(parents=True, exist_ok=False)
    canvas_px = contract.get("reference_canvas_px", 1536)

    derived: list[dict[str, Any]] = []
    for index, reference in enumerate(contract["references"]):
        source = resolve_path(reference["path"], workspace)
        image, source_sha = load_image_snapshot(source, mode="RGBA")
        if source_sha != reference["sha256"]:
            raise GateError(
                f"Reference changed after preflight: {source}",
                "SOURCE_HASH_DRIFT",
            )
        if reference["role"] in PRODUCT_ROLES:
            image = image.crop(_content_bbox(image))
        max_content = int(canvas_px * 0.9)
        scale = min(max_content / image.width, max_content / image.height, 1.0)
        if scale < 1:
            size = (
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            )
            image = image.resize(size, Image.Resampling.LANCZOS)
        alpha_extrema = cast(tuple[int, int], image.getchannel("A").getextrema())
        has_real_alpha = alpha_extrema[0] < 255
        background = (0, 0, 0, 0) if has_real_alpha else (255, 255, 255, 255)
        canvas = Image.new("RGBA", (canvas_px, canvas_px), background)
        offset = ((canvas_px - image.width) // 2, (canvas_px - image.height) // 2)
        canvas.alpha_composite(image, offset)
        canonical = (
            f"{_slug(reference['sku'])}-{_slug(reference['view'])}-"
            f"{_slug(reference['role'])}-{index + 1}.png"
        )
        destination = out_dir / canonical
        canvas.save(destination, format="PNG", optimize=True)
        derived.append(
            {
                "source_path": str(source),
                "source_sha256": source_sha,
                "canonical_path": str(destination),
                "canonical_sha256": sha256_file(destination),
                "sku": reference["sku"],
                "view": reference["view"],
                "role": reference["role"],
                "dimensions": [canvas_px, canvas_px],
                "source_was_upsampled": False,
            }
        )

    mask_record: dict[str, Any] | None = None
    if contract["operation"] == "localized_product_patch":
        target = resolve_path(contract["target"]["path"], workspace)
        target_image, target_sha = load_image_snapshot(target, mode="RGBA")
        if target_sha != contract["target"]["sha256"]:
            raise GateError(
                f"Target changed after preflight: {target}",
                "SOURCE_HASH_DRIFT",
            )
        mask = build_edit_mask(contract, target_image.size)
        mask_path = out_dir / "editable-region-mask-white-is-editable.png"
        mask.save(mask_path, format="PNG")
        overlay = target_image.copy()
        red = Image.new("RGBA", target_image.size, (255, 0, 0, 0))
        red.putalpha(mask.point(lambda value: round(value * 0.42)))
        overlay = Image.alpha_composite(overlay, red)
        overlay_path = out_dir / "editable-region-overlay-review.png"
        overlay.save(overlay_path, format="PNG")
        mask_record = {
            "mask_path": str(mask_path),
            "mask_sha256": sha256_file(mask_path),
            "overlay_path": str(overlay_path),
            "overlay_sha256": sha256_file(overlay_path),
            "polarity": "white_is_editable_black_is_locked",
        }

    persisted_contract, contract_file_sha = load_contract_snapshot(contract_path)
    if sha256_json(persisted_contract) != sha256_json(contract):
        raise GateError(
            "Contract changed after it was loaded; restart optimization",
            "CONTRACT_CHANGED_DURING_OPERATION",
        )
    manifest = {
        "schema": "product-fidelity-reference-pack.v2",
        "status": "PREPARED_NOT_GENERATION_APPROVED",
        "contract_path": str(contract_path),
        "contract_sha256": sha256_json(contract),
        "contract_file_sha256": contract_file_sha,
        "operation": contract["operation"],
        "references": derived,
        "mask": mask_record,
        "required_next_step": "Visually inspect optimized references and mask overlay, then rerun preflight using current hashes.",
    }
    manifest_path = out_dir / "reference-pack-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def _changed_bbox(changed: np.ndarray) -> list[int] | None:
    if not np.any(changed):
        return None
    ys, xs = np.nonzero(changed)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def verify_localized_output(
    contract: dict[str, Any],
    workspace: Path,
    output_path: Path,
    *,
    output_snapshot: tuple[Image.Image, str] | None = None,
) -> dict[str, Any]:
    verification = contract.get("verification")
    if (
        not isinstance(verification, dict)
        or verification.get("max_outside_changed_pixels") != 0
        or verification.get("per_channel_tolerance") != 0
        or not is_strict_int(verification.get("min_inside_changed_pixels"))
        or verification["min_inside_changed_pixels"] < 1
    ):
        return {
            "status": "BLOCKED",
            "code": "UNSAFE_VERIFICATION_POLICY",
            "contract_sha256": sha256_json(contract),
            "wiring_allowed": False,
        }
    target_path = resolve_path(contract["target"]["path"], workspace)
    baseline, baseline_sha = load_image_snapshot(target_path, mode="RGBA")
    if baseline_sha != contract.get("target", {}).get("sha256"):
        return {
            "status": "BLOCKED",
            "code": "SOURCE_HASH_DRIFT",
            "baseline_sha256": baseline_sha,
            "expected_baseline_sha256": contract.get("target", {}).get("sha256"),
            "wiring_allowed": False,
        }
    if output_snapshot is None:
        output, output_sha = load_image_snapshot(output_path, mode="RGBA")
    else:
        output, output_sha = output_snapshot
        output = output.convert("RGBA")
    if output.size != baseline.size:
        return {
            "status": "BLOCKED",
            "code": "DIMENSION_DRIFT",
            "baseline_dimensions": list(baseline.size),
            "output_dimensions": list(output.size),
        }
    mask = (
        np.asarray(
            build_edit_mask(contract, baseline.size, include_feather=False),
            dtype=np.uint8,
        )
        > 0
    )
    baseline_pixels = np.asarray(baseline, dtype=np.int16)
    output_pixels = np.asarray(output, dtype=np.int16)
    tolerance = int(contract.get("verification", {}).get("per_channel_tolerance", 0))
    changed = np.any(np.abs(output_pixels - baseline_pixels) > tolerance, axis=2)
    outside_changed = changed & ~mask
    inside_changed = changed & mask
    outside_count = int(np.count_nonzero(outside_changed))
    inside_count = int(np.count_nonzero(inside_changed))
    maximum = int(contract.get("verification", {}).get("max_outside_changed_pixels", 0))
    minimum_inside = int(contract.get("verification", {}).get("min_inside_changed_pixels", 1))
    passed = outside_count <= maximum and inside_count >= minimum_inside
    return {
        "status": "PASS" if passed else "BLOCKED",
        "code": (
            "OUTSIDE_MASK_PRESERVED"
            if passed
            else ("OUTSIDE_MASK_DRIFT" if outside_count > maximum else "NO_INTENDED_CHANGE")
        ),
        "baseline_sha256": baseline_sha,
        "output_sha256": output_sha,
        "dimensions": list(baseline.size),
        "outside_changed_pixels": outside_count,
        "maximum_outside_changed_pixels": maximum,
        "outside_changed_bbox": _changed_bbox(outside_changed),
        "inside_changed_pixels": inside_count,
        "minimum_inside_changed_pixels": minimum_inside,
        "per_channel_tolerance": tolerance,
    }


def verify_semantic_review(
    contract: dict[str, Any],
    workspace: Path,
    output_path: Path,
    review_path: Path | None,
    candidate_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate the independent, contract- and candidate-bound semantic review."""
    current_candidate_sha = read_bytes_snapshot(output_path)[1]
    if candidate_sha256 is not None and current_candidate_sha != candidate_sha256:
        return {
            "status": "BLOCKED",
            "code": "CANDIDATE_CHANGED_DURING_VERIFY",
            "candidate_sha256": current_candidate_sha,
            "expected_candidate_sha256": candidate_sha256,
            "findings": ["CANDIDATE_CHANGED_DURING_VERIFY"],
            "wiring_allowed": False,
        }
    candidate_sha = current_candidate_sha
    if review_path is None:
        return {
            "status": "BLOCKED",
            "code": "SEMANTIC_REVIEW_REQUIRED",
            "candidate_sha256": candidate_sha,
            "findings": ["SEMANTIC_REVIEW_REQUIRED"],
            "wiring_allowed": False,
        }
    try:
        resolved_review = resolve_path(str(review_path), workspace)
        review, review_sha = load_contract_snapshot(resolved_review)
    except GateError as exc:
        return {
            "status": "BLOCKED",
            "code": "SEMANTIC_REVIEW_INVALID",
            "candidate_sha256": candidate_sha,
            "findings": ["REVIEW_UNREADABLE_OR_OUTSIDE_WORKSPACE"],
            "message": str(exc),
            "wiring_allowed": False,
        }

    findings: list[str] = []
    if review.get("schema") != SEMANTIC_REVIEW_SCHEMA:
        findings.append("REVIEW_SCHEMA_INVALID")
    if review.get("contract_sha256") != sha256_json(contract):
        findings.append("REVIEW_CONTRACT_HASH_MISMATCH")
    if review.get("candidate_sha256") != candidate_sha:
        findings.append("REVIEW_CANDIDATE_HASH_MISMATCH")

    semantic_contract = contract.get("semantic_review", {})
    if not isinstance(semantic_contract, dict):
        semantic_contract = {}
    reviewer = review.get("reviewer")
    candidate_author = semantic_contract.get("candidate_author")
    if not isinstance(reviewer, str) or not reviewer.strip():
        findings.append("REVIEWER_REQUIRED")
    elif same_identity(reviewer, candidate_author):
        findings.append("BUILDER_SELF_REVIEW_FORBIDDEN")
    if review.get("scope") != semantic_contract.get("scope"):
        findings.append("REVIEW_SCOPE_MISMATCH")
    if review.get("verdict") != "PASS":
        findings.append("SEMANTIC_REVIEW_NOT_PASSING")
    required_checks = semantic_contract.get("required_checks")
    review_checks = review.get("checks")
    if not (
        isinstance(required_checks, list)
        and isinstance(review_checks, dict)
        and set(review_checks) == set(required_checks)
        and all(review_checks.get(check) == "PASS" for check in required_checks)
    ):
        findings.append("SEMANTIC_CHECKS_NOT_PASSING")
    expected_text = [
        text
        for reference in contract.get("references", [])
        if isinstance(reference, dict)
        for text in reference.get("verbatim_text", [])
        if isinstance(text, str)
    ]
    if review.get("expected_verbatim_text") != expected_text:
        findings.append("VERBATIM_TEXT_REVIEW_MISMATCH")
    review_findings = review.get("findings")
    if not isinstance(review_findings, list):
        findings.append("REVIEW_FINDINGS_LIST_REQUIRED")
        review_findings = []
    elif review_findings:
        findings.append("SEMANTIC_REVIEW_HAS_FINDINGS")

    passed = not findings
    return {
        "status": "PASS" if passed else "BLOCKED",
        "code": "SEMANTIC_REVIEW_PASS" if passed else "SEMANTIC_REVIEW_BLOCKED",
        "candidate_sha256": candidate_sha,
        "contract_sha256": sha256_json(contract),
        "review_path": str(resolved_review),
        "review_sha256": review_sha,
        "reviewer": reviewer,
        "review_identity_verified": False,
        "review_trust_state": "SELF_DECLARED_IDENTITY",
        "scope": review.get("scope"),
        "checks": review_checks,
        "expected_verbatim_text": expected_text,
        "verdict": review.get("verdict"),
        "review_findings": review_findings,
        "findings": findings,
        "wiring_allowed": False,
    }


def verify_protected_output(
    contract: dict[str, Any],
    workspace: Path,
    output_path: Path,
    *,
    output_snapshot: tuple[Image.Image, str] | None = None,
) -> dict[str, Any]:
    placements = contract.get("protected_placements")
    if not isinstance(placements, list) or not placements:
        return {
            "status": "BLOCKED",
            "code": "PROTECTED_PLACEMENTS_REQUIRED",
            "message": "Verification requires exact x/y placements for protected layers",
        }
    if output_snapshot is None:
        output_image, output_sha = load_image_snapshot(output_path, mode="RGB")
    else:
        output_image, output_sha = output_snapshot
        output_image = output_image.convert("RGB")
    output_dimensions = list(output_image.size)
    output = np.asarray(output_image, dtype=np.uint8)
    if contract.get("operation") == "native_collection_scene":
        expected_dimensions = contract.get("optical_contract", {}).get("output_dimensions")
    else:
        target_path = resolve_path(contract["target"]["path"], workspace)
        target_image, target_sha = load_image_snapshot(target_path)
        if target_sha != contract.get("target", {}).get("sha256"):
            return {
                "status": "BLOCKED",
                "code": "SOURCE_HASH_DRIFT",
                "output_sha256": output_sha,
                "target_sha256": target_sha,
                "expected_target_sha256": contract.get("target", {}).get("sha256"),
                "wiring_allowed": False,
            }
        expected_dimensions = list(target_image.size)
    if output_dimensions != expected_dimensions:
        return {
            "status": "BLOCKED",
            "code": "PROTECTED_OUTPUT_DIMENSION_DRIFT",
            "output_sha256": output_sha,
            "expected_dimensions": expected_dimensions,
            "output_dimensions": output_dimensions,
            "wiring_allowed": False,
        }
    checks: list[dict[str, Any]] = []
    passed = True
    protected_references: dict[Path, str] = {}
    for reference in contract.get("references", []):
        if (
            isinstance(reference, dict)
            and reference.get("role") in {"protected_product_layer", "protected_garment_matte"}
            and isinstance(reference.get("path"), str)
            and isinstance(reference.get("sha256"), str)
        ):
            protected_references[resolve_path(reference["path"], workspace)] = reference["sha256"]
    for placement in placements:
        layer_path = resolve_path(placement["path"], workspace)
        layer_image, layer_sha = load_image_snapshot(layer_path, mode="RGBA")
        expected_layer_sha = protected_references.get(layer_path)
        if expected_layer_sha is None or layer_sha != expected_layer_sha:
            checks.append(
                {
                    "path": str(layer_path),
                    "status": "BLOCKED",
                    "code": "PROTECTED_LAYER_HASH_DRIFT",
                    "expected_sha256": expected_layer_sha,
                    "actual_sha256": layer_sha,
                }
            )
            passed = False
            continue
        layer = np.asarray(layer_image, dtype=np.uint8)
        x = int(placement["x"])
        y = int(placement["y"])
        height, width = layer.shape[:2]
        if x < 0 or y < 0 or y + height > output.shape[0] or x + width > output.shape[1]:
            checks.append(
                {
                    "path": str(layer_path),
                    "status": "BLOCKED",
                    "code": "PLACEMENT_OUT_OF_BOUNDS",
                }
            )
            passed = False
            continue
        opaque = layer[..., 3] == 255
        if not np.any(opaque):
            return {
                "status": "BLOCKED",
                "code": "PROTECTED_LAYER_OPAQUE_PIXELS_REQUIRED",
                "output_sha256": output_sha,
                "path": str(layer_path),
                "wiring_allowed": False,
            }
        region = output[y : y + height, x : x + width]
        mismatch = np.any(region != layer[..., :3], axis=2) & opaque
        mismatch_count = int(np.count_nonzero(mismatch))
        checks.append(
            {
                "path": str(layer_path),
                "status": "PASS" if mismatch_count == 0 else "BLOCKED",
                "opaque_pixel_mismatches": mismatch_count,
            }
        )
        passed = passed and mismatch_count == 0
    return {
        "status": "PASS" if passed else "BLOCKED",
        "code": "PROTECTED_PIXELS_PRESERVED" if passed else "PROTECTED_PIXEL_DRIFT",
        "output_sha256": output_sha,
        "checks": checks,
        "promotion_state": "LAYOUT_PROOF_ONLY",
        "native_integration_verified": False,
        "wiring_allowed": False,
    }


def verify_native_output(
    contract: dict[str, Any],
    workspace: Path,
    output_path: Path,
    review_path: Path | None,
    *,
    output_snapshot: tuple[Image.Image, str] | None = None,
) -> dict[str, Any]:
    if output_snapshot is None:
        output, candidate_sha = load_image_snapshot(output_path)
    else:
        output, candidate_sha = output_snapshot
    output_dimensions = list(output.size)
    expected_dimensions = contract.get("optical_contract", {}).get("output_dimensions")
    if output_dimensions != expected_dimensions:
        return {
            "status": "BLOCKED",
            "code": "NATIVE_OUTPUT_DIMENSION_DRIFT",
            "candidate_sha256": candidate_sha,
            "expected_dimensions": expected_dimensions,
            "output_dimensions": output_dimensions,
            "wiring_allowed": False,
        }

    redraw_allowed = contract.get("generator", {}).get("product_redraw_allowed") is True
    preservation: dict[str, Any]
    if redraw_allowed:
        preservation = {
            "status": "NOT_EXACT",
            "code": "FOUNDER_AUTHORIZED_REDRAW_CANDIDATE",
            "message": "This route cannot claim protected product pixels.",
        }
    else:
        preservation = verify_protected_output(
            contract,
            workspace,
            output_path,
            output_snapshot=(output, candidate_sha),
        )
        if preservation.get("status") != "PASS":
            return {
                "status": "BLOCKED",
                "code": "GARMENT_PRESERVATION_FAILED",
                "candidate_sha256": candidate_sha,
                "preservation": preservation,
                "wiring_allowed": False,
            }

    current_candidate_sha = read_bytes_snapshot(output_path)[1]
    if current_candidate_sha != candidate_sha:
        return {
            "status": "BLOCKED",
            "code": "CANDIDATE_CHANGED_DURING_VERIFY",
            "candidate_sha256": current_candidate_sha,
            "expected_candidate_sha256": candidate_sha,
            "preservation": preservation,
            "wiring_allowed": False,
        }

    if review_path is None:
        return {
            "status": "BLOCKED",
            "code": "NATIVE_REVIEW_REQUIRED",
            "candidate_sha256": candidate_sha,
            "preservation": preservation,
            "wiring_allowed": False,
        }
    try:
        resolved_review = resolve_path(str(review_path), workspace)
    except GateError as exc:
        return {
            "status": "BLOCKED",
            "code": "REVIEW_OUTSIDE_WORKSPACE",
            "message": str(exc),
            "candidate_sha256": candidate_sha,
            "wiring_allowed": False,
        }
    try:
        review, review_sha = load_contract_snapshot(resolved_review)
    except GateError as exc:
        return {
            "status": "BLOCKED",
            "code": "NATIVE_REVIEW_INVALID",
            "message": str(exc),
            "candidate_sha256": candidate_sha,
            "wiring_allowed": False,
        }
    findings: list[str] = []
    if review.get("schema") != NATIVE_REVIEW_SCHEMA:
        findings.append("REVIEW_SCHEMA_INVALID")
    if review.get("contract_sha256") != sha256_json(contract):
        findings.append("REVIEW_CONTRACT_HASH_MISMATCH")
    if review.get("candidate_sha256") != candidate_sha:
        findings.append("REVIEW_CANDIDATE_HASH_MISMATCH")
    reviewer = review.get("reviewer")
    author = contract.get("review_gate", {}).get("candidate_author")
    if not isinstance(reviewer, str) or not reviewer.strip():
        findings.append("REVIEWER_REQUIRED")
    elif same_identity(reviewer, author):
        findings.append("BUILDER_SELF_REVIEW_FORBIDDEN")
    scores = review.get("scores")
    minimums = contract.get("review_gate", {}).get("minimum_scores", {})
    score_results: dict[str, dict[str, Any]] = {}
    if not isinstance(scores, dict):
        findings.append("REVIEW_SCORES_REQUIRED")
        scores = {}
    for dimension in NATIVE_SCORE_DIMENSIONS:
        score = scores.get(dimension)
        minimum = minimums.get(dimension)
        valid_score = isinstance(score, (int, float)) and not isinstance(score, bool)
        passed = False
        if valid_score and is_strict_int(minimum):
            numeric_score = float(cast(int | float, score))
            passed = 0 <= numeric_score <= 100 and numeric_score >= minimum
        score_results[dimension] = {
            "score": score,
            "minimum": minimum,
            "passed": passed,
        }
        if not passed:
            findings.append(f"SCORE_BELOW_MINIMUM:{dimension}")
    hard_fails = review.get("hard_fails")
    if not isinstance(hard_fails, list):
        findings.append("HARD_FAILS_LIST_REQUIRED")
        hard_fails = []
    elif hard_fails:
        findings.append("NATIVE_INTEGRATION_HARD_FAIL")
    founder_status = review.get("founder_status")
    if founder_status not in {"PENDING", "APPROVED", "REJECTED"}:
        findings.append("FOUNDER_STATUS_INVALID")
    elif founder_status == "REJECTED":
        findings.append("FOUNDER_REJECTED")

    passed = not findings
    if passed and founder_status == "APPROVED":
        code = "PASS_REVIEW_CANDIDATE_WITH_UNVERIFIED_FOUNDER_CLAIM"
    elif passed:
        code = "PASS_REVIEW_CANDIDATE"
    else:
        code = "NATIVE_INTEGRATION_REVIEW_BLOCKED"
    return {
        "status": "PASS" if passed else "BLOCKED",
        "code": code,
        "candidate_sha256": candidate_sha,
        "contract_sha256": sha256_json(contract),
        "output_dimensions": output_dimensions,
        "preservation": preservation,
        "review_path": str(resolved_review),
        "review_sha256": review_sha,
        "reviewer": reviewer,
        "review_identity_verified": False,
        "review_trust_state": "SELF_DECLARED_IDENTITY",
        "score_results": score_results,
        "hard_fails": hard_fails,
        "founder_status": founder_status,
        "findings": findings,
        "founder_approval_verified": False,
        "promotion_state": (
            "FOUNDER_APPROVAL_ATTESTATION_REQUIRED"
            if founder_status == "APPROVED" and passed
            else "FOUNDER_REVIEW_REQUIRED"
        ),
        "wiring_allowed": False,
        "deployment_allowed": False,
    }


def _validate_receipt_destination(
    receipt: Path | None,
    contract_path: Path,
    contract: dict[str, Any],
    workspace: Path,
    extra_inputs: list[Path],
) -> bool:
    if receipt is None:
        return False
    bound_inputs = {contract_path}
    records: list[Any] = [contract.get("target")]
    references = contract.get("references")
    if isinstance(references, list):
        records.extend(references)
    for record in records:
        if not isinstance(record, dict) or not isinstance(record.get("path"), str):
            continue
        try:
            bound_inputs.add(resolve_path(record["path"], workspace))
        except GateError:
            continue
        authority = record.get("authority_receipt")
        if isinstance(authority, dict) and isinstance(authority.get("path"), str):
            try:
                bound_inputs.add(resolve_path(authority["path"], workspace))
            except GateError:
                continue
    bound_inputs.update(extra_inputs)
    if receipt in bound_inputs:
        raise GateError(
            f"Receipt path collides with a bound input: {receipt}",
            "RECEIPT_INPUT_COLLISION",
        )
    if receipt.exists():
        raise GateError(
            f"Receipt path already exists; refusing to overwrite: {receipt}",
            "RECEIPT_ALREADY_EXISTS",
        )
    if not receipt.parent.is_dir():
        raise GateError(
            f"Receipt parent directory does not exist: {receipt.parent}",
            "RECEIPT_PARENT_MISSING",
        )
    return True


def write_receipt(data: dict[str, Any], receipt: Path | None) -> None:
    rendered = json.dumps(data, indent=2) + "\n"
    if receipt is not None:
        try:
            with receipt.open("x", encoding="utf-8") as handle:
                handle.write(rendered)
        except FileExistsError as exc:
            raise GateError(
                f"Receipt path already exists; refusing to overwrite: {receipt}",
                "RECEIPT_ALREADY_EXISTS",
            ) from exc
        except OSError as exc:
            raise GateError(
                f"Cannot create receipt {receipt}: {exc}",
                "RECEIPT_WRITE_FAILED",
            ) from exc
    print(rendered, end="")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common(subparser: argparse.ArgumentParser) -> None:
        subparser.add_argument("--contract", type=Path, required=True)
        subparser.add_argument("--workspace", type=Path, required=True)
        subparser.add_argument("--receipt", type=Path)

    preflight_parser = subparsers.add_parser("preflight")
    add_common(preflight_parser)

    optimize_parser = subparsers.add_parser("optimize")
    add_common(optimize_parser)
    optimize_parser.add_argument("--out-dir", type=Path, required=True)

    verify_parser = subparsers.add_parser("verify")
    add_common(verify_parser)
    verify_parser.add_argument("--output", type=Path, required=True)
    verify_parser.add_argument("--review", type=Path)

    args = parser.parse_args()
    workspace = args.workspace.expanduser().resolve()
    receipt_path: Path | None = None
    receipt_ready = False

    try:
        if args.receipt is not None:
            try:
                receipt_path = resolve_path(str(args.receipt), workspace)
            except GateError as exc:
                raise GateError(str(exc), "RECEIPT_OUTSIDE_WORKSPACE") from exc
        try:
            contract_path = resolve_path(str(args.contract), workspace)
        except GateError as exc:
            raise GateError(str(exc), "CONTRACT_OUTSIDE_WORKSPACE") from exc
        contract = load_contract(contract_path)
        extra_inputs: list[Path] = []
        for input_path in (
            getattr(args, "output", None),
            getattr(args, "review", None),
        ):
            if input_path is None:
                continue
            try:
                extra_inputs.append(resolve_path(str(input_path), workspace))
            except GateError:
                continue
        receipt_ready = _validate_receipt_destination(
            receipt_path, contract_path, contract, workspace, extra_inputs
        )
        if args.command == "preflight":
            result = validate_contract(contract, workspace)
        elif args.command == "optimize":
            result = optimize_references(
                contract,
                contract_path,
                workspace,
                args.out_dir.expanduser().resolve(),
            )
        else:
            preflight = validate_contract(contract, workspace)
            if preflight["status"] != "PASS":
                result = {
                    "schema": "product-fidelity-verification-receipt.v2",
                    "status": "BLOCKED",
                    "code": "PREFLIGHT_NOT_PASSING",
                    "preflight": preflight,
                }
            else:
                output_snapshot: tuple[Image.Image, str] | None = None
                try:
                    output_path = resolve_path(str(args.output), workspace)
                except GateError as exc:
                    result = {
                        "schema": "product-fidelity-verification-receipt.v2",
                        "status": "BLOCKED",
                        "code": "OUTPUT_OUTSIDE_WORKSPACE",
                        "message": str(exc),
                    }
                    output_path = None
                if output_path is not None and not output_path.is_file():
                    raise GateError(f"Output does not exist: {output_path}")
                if output_path is not None:
                    try:
                        output_snapshot = load_image_snapshot(output_path)
                    except GateError as exc:
                        result = {
                            "schema": "product-fidelity-verification-receipt.v2",
                            "status": "BLOCKED",
                            "code": "OUTPUT_UNREADABLE",
                            "message": str(exc),
                            "output_path": str(output_path),
                        }
                        output_path = None
                if output_path is None:
                    pass
                elif contract["operation"] == "localized_product_patch":
                    pixel_result = verify_localized_output(
                        contract,
                        workspace,
                        output_path,
                        output_snapshot=output_snapshot,
                    )
                    if pixel_result["status"] != "PASS":
                        result = pixel_result
                    else:
                        semantic_result = verify_semantic_review(
                            contract,
                            workspace,
                            output_path,
                            args.review.expanduser() if args.review else None,
                            pixel_result["output_sha256"],
                        )
                        if semantic_result["status"] != "PASS":
                            result = {
                                **semantic_result,
                                "pixel_verification": pixel_result,
                            }
                        else:
                            result = {
                                "status": "PASS",
                                "code": "PASS_PIXEL_AND_SEMANTIC_REVIEW",
                                "candidate_sha256": semantic_result["candidate_sha256"],
                                "pixel_verification": pixel_result,
                                "semantic_review": semantic_result,
                                "promotion_state": "FOUNDER_REVIEW_REQUIRED",
                                "wiring_allowed": False,
                                "deployment_allowed": False,
                            }
                elif contract["operation"] == "protected_scene_composite":
                    result = verify_protected_output(
                        contract,
                        workspace,
                        output_path,
                        output_snapshot=output_snapshot,
                    )
                else:
                    result = verify_native_output(
                        contract,
                        workspace,
                        output_path,
                        args.review.expanduser() if args.review else None,
                        output_snapshot=output_snapshot,
                    )
                result["schema"] = "product-fidelity-verification-receipt.v2"
        write_receipt(result, receipt_path)
        return 0 if result.get("status") in {"PASS", "PREPARED_NOT_GENERATION_APPROVED"} else 2
    except GateError as exc:
        result = {
            "schema": "product-fidelity-gate-error.v2",
            "status": "BLOCKED",
            "code": exc.code,
            "message": str(exc),
        }
        if receipt_ready and not exc.code.startswith("RECEIPT_"):
            try:
                write_receipt(result, receipt_path)
            except GateError as receipt_exc:
                result = {
                    "schema": "product-fidelity-gate-error.v2",
                    "status": "BLOCKED",
                    "code": receipt_exc.code,
                    "message": str(receipt_exc),
                }
                write_receipt(result, None)
        else:
            write_receipt(result, None)
        return 2


if __name__ == "__main__":
    sys.exit(main())
