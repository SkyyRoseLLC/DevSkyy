#!/usr/bin/env python3
"""Fail-closed preflight, reference packing, and output verification for product edits."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageOps

SCHEMA = "product-fidelity-edit.v1"
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
AUTHORITY_STATES = {"founder_approved", "sot_verified"}


class GateError(RuntimeError):
    """Raised for malformed inputs that prevent a trustworthy gate result."""


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


def load_contract(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GateError(f"Cannot read contract {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise GateError("Contract root must be a JSON object")
    return data


def image_metadata(path: Path) -> dict[str, Any]:
    with Image.open(path) as opened:
        image = ImageOps.exif_transpose(opened)
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
    actual = sha256_file(path)
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
    try:
        metadata = image_metadata(path)
    except Exception as exc:  # Pillow reports format-specific errors.
        findings.append(Finding("error", "IMAGE_UNREADABLE", f"{label}: {exc}", str(path)))
        return path, None
    return path, metadata


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
    kind = region.get("type")
    if kind == "bbox":
        bbox = region.get("bbox")
        if not (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, int) for value in bbox)
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
                and all(isinstance(value, int) for value in point)
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
        isinstance(value, list) and len(value) == 4 and all(isinstance(item, int) for item in value)
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
        and all(isinstance(item, int) and item > 0 for item in dimensions)
    ):
        findings.append(
            Finding(
                "error",
                "OUTPUT_DIMENSIONS_INVALID",
                "optical_contract.output_dimensions needs two positive integers",
            )
        )
        return None
    width, height = dimensions
    if not _valid_box(value.get("subject_bbox"), width, height):
        findings.append(
            Finding(
                "error",
                "SUBJECT_BBOX_INVALID",
                "subject_bbox must be inside output geometry",
            )
        )
    horizon = value.get("horizon_y")
    if not isinstance(horizon, int) or not 0 <= horizon < height:
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
            and all(isinstance(item, int) for item in point)
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
            if not isinstance(temperature, int) or not 1500 <= temperature <= 15000:
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
        if not isinstance(score, int) or not 0 <= score <= 100:
            findings.append(
                Finding(
                    "error",
                    "MINIMUM_SCORE_INVALID",
                    f"minimum_scores.{dimension} must be an integer 0..100",
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


def validate_contract(contract: dict[str, Any], workspace: Path) -> dict[str, Any]:
    findings: list[Finding] = []
    if contract.get("schema") != SCHEMA:
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
        path, metadata = _validate_hash_entry(reference, workspace, f"reference {index}", findings)
        if path is not None and isinstance(view, str) and _filename_view_conflict(path, view):
            findings.append(
                Finding(
                    "warning",
                    "FILENAME_VIEW_CONFLICT",
                    f"Filename conflicts with declared {view} view; optimize to a canonical pack before generation",
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
                "metadata": metadata,
            }
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
        if verification.get("max_outside_changed_pixels", 0) != 0:
            findings.append(
                Finding(
                    "warning",
                    "OUTSIDE_CHANGE_TOLERANCE_NONZERO",
                    "Exact product patches should normally allow zero outside-mask changes",
                )
            )
        if int(verification.get("min_inside_changed_pixels", 1)) < 1:
            findings.append(
                Finding(
                    "error",
                    "MINIMUM_INTENDED_CHANGE_REQUIRED",
                    "Localized patches must require at least one changed pixel inside the authorized region",
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
        _validate_optical_contract(contract.get("optical_contract"), findings)
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
        "schema": "product-fidelity-preflight-receipt.v1",
        "status": "PASS" if not errors else "BLOCKED",
        "operation": operation,
        "target": {
            "path": str(target_path) if target_path else None,
            "metadata": target_meta,
        },
        "references": reference_report,
        "findings": [finding.as_dict() for finding in findings],
    }


def build_edit_mask(contract: dict[str, Any], target_size: tuple[int, int]) -> Image.Image:
    region = contract["edit_region"]
    mask = Image.new("L", target_size, 0)
    draw = ImageDraw.Draw(mask)
    if region["type"] == "bbox":
        draw.rectangle(tuple(region["bbox"]), fill=255)
    else:
        draw.polygon([tuple(point) for point in region["points"]], fill=255)
    feather = int(region.get("feather_px", 0))
    if feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=feather))
    return mask


def _content_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    rgba = image.convert("RGBA")
    alpha = rgba.getchannel("A")
    if alpha.getextrema()[0] < 255:
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
    out_dir.mkdir(parents=True, exist_ok=True)
    canvas_px = int(contract.get("reference_canvas_px", 1536))
    if canvas_px < 512 or canvas_px > 4096:
        raise GateError("reference_canvas_px must be between 512 and 4096")

    derived: list[dict[str, Any]] = []
    for index, reference in enumerate(contract["references"]):
        source = resolve_path(reference["path"], workspace)
        with Image.open(source) as opened:
            image = ImageOps.exif_transpose(opened).convert("RGBA")
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
        has_real_alpha = image.getchannel("A").getextrema()[0] < 255
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
                "source_sha256": reference["sha256"],
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
        with Image.open(target) as opened:
            target_image = ImageOps.exif_transpose(opened).convert("RGBA")
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

    manifest = {
        "schema": "product-fidelity-reference-pack.v1",
        "status": "PREPARED_NOT_GENERATION_APPROVED",
        "contract_path": str(contract_path),
        "contract_sha256": sha256_file(contract_path),
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
    contract: dict[str, Any], workspace: Path, output_path: Path
) -> dict[str, Any]:
    target_path = resolve_path(contract["target"]["path"], workspace)
    with Image.open(target_path) as opened:
        baseline = ImageOps.exif_transpose(opened).convert("RGBA")
    with Image.open(output_path) as opened:
        output = ImageOps.exif_transpose(opened).convert("RGBA")
    if output.size != baseline.size:
        return {
            "status": "BLOCKED",
            "code": "DIMENSION_DRIFT",
            "baseline_dimensions": list(baseline.size),
            "output_dimensions": list(output.size),
        }
    mask = np.asarray(build_edit_mask(contract, baseline.size), dtype=np.uint8) > 0
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
        "baseline_sha256": sha256_file(target_path),
        "output_sha256": sha256_file(output_path),
        "dimensions": list(baseline.size),
        "outside_changed_pixels": outside_count,
        "maximum_outside_changed_pixels": maximum,
        "outside_changed_bbox": _changed_bbox(outside_changed),
        "inside_changed_pixels": inside_count,
        "minimum_inside_changed_pixels": minimum_inside,
        "per_channel_tolerance": tolerance,
    }


def verify_protected_output(
    contract: dict[str, Any], workspace: Path, output_path: Path
) -> dict[str, Any]:
    placements = contract.get("protected_placements")
    if not isinstance(placements, list) or not placements:
        return {
            "status": "BLOCKED",
            "code": "PROTECTED_PLACEMENTS_REQUIRED",
            "message": "Verification requires exact x/y placements for protected layers",
        }
    with Image.open(output_path) as opened:
        output = np.asarray(ImageOps.exif_transpose(opened).convert("RGB"), dtype=np.uint8)
    checks: list[dict[str, Any]] = []
    passed = True
    for placement in placements:
        layer_path = resolve_path(placement["path"], workspace)
        with Image.open(layer_path) as opened:
            layer = np.asarray(ImageOps.exif_transpose(opened).convert("RGBA"), dtype=np.uint8)
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
        "output_sha256": sha256_file(output_path),
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
) -> dict[str, Any]:
    with Image.open(output_path) as opened:
        output = ImageOps.exif_transpose(opened)
        output_dimensions = list(output.size)
    expected_dimensions = contract.get("optical_contract", {}).get("output_dimensions")
    candidate_sha = sha256_file(output_path)
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
        preservation = verify_protected_output(contract, workspace, output_path)
        if preservation.get("status") != "PASS":
            return {
                "status": "BLOCKED",
                "code": "GARMENT_PRESERVATION_FAILED",
                "candidate_sha256": candidate_sha,
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
    review = load_contract(resolved_review)
    findings: list[str] = []
    if review.get("schema") != "product-fidelity-native-review.v1":
        findings.append("REVIEW_SCHEMA_INVALID")
    if review.get("candidate_sha256") != candidate_sha:
        findings.append("REVIEW_CANDIDATE_HASH_MISMATCH")
    reviewer = review.get("reviewer")
    author = contract.get("review_gate", {}).get("candidate_author")
    if not isinstance(reviewer, str) or not reviewer.strip():
        findings.append("REVIEWER_REQUIRED")
    elif reviewer == author:
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
        passed = valid_score and isinstance(minimum, int) and 0 <= score <= 100 and score >= minimum
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
        code = "PASS_FOUNDER_APPROVED_CANDIDATE"
    elif passed:
        code = "PASS_REVIEW_CANDIDATE"
    else:
        code = "NATIVE_INTEGRATION_REVIEW_BLOCKED"
    return {
        "status": "PASS" if passed else "BLOCKED",
        "code": code,
        "candidate_sha256": candidate_sha,
        "output_dimensions": output_dimensions,
        "preservation": preservation,
        "review_path": str(resolved_review),
        "review_sha256": sha256_file(resolved_review),
        "reviewer": reviewer,
        "score_results": score_results,
        "hard_fails": hard_fails,
        "founder_status": founder_status,
        "findings": findings,
        "promotion_state": (
            "FOUNDER_APPROVED_NOT_WIRED"
            if founder_status == "APPROVED" and passed
            else "FOUNDER_REVIEW_REQUIRED"
        ),
        "wiring_allowed": False,
        "deployment_allowed": False,
    }


def write_receipt(data: dict[str, Any], receipt: Path | None) -> None:
    rendered = json.dumps(data, indent=2) + "\n"
    if receipt is not None:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(rendered, encoding="utf-8")
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
    contract_path = args.contract.expanduser().resolve()
    workspace = args.workspace.expanduser().resolve()
    contract = load_contract(contract_path)

    try:
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
                    "schema": "product-fidelity-verification-receipt.v1",
                    "status": "BLOCKED",
                    "code": "PREFLIGHT_NOT_PASSING",
                    "preflight": preflight,
                }
            else:
                output_path = args.output.expanduser().resolve()
                if not output_path.is_file():
                    raise GateError(f"Output does not exist: {output_path}")
                if contract["operation"] == "localized_product_patch":
                    result = verify_localized_output(contract, workspace, output_path)
                elif contract["operation"] == "protected_scene_composite":
                    result = verify_protected_output(contract, workspace, output_path)
                else:
                    result = verify_native_output(
                        contract,
                        workspace,
                        output_path,
                        args.review.expanduser() if args.review else None,
                    )
                result["schema"] = "product-fidelity-verification-receipt.v1"
        write_receipt(result, args.receipt.expanduser().resolve() if args.receipt else None)
        return 0 if result.get("status") in {"PASS", "PREPARED_NOT_GENERATION_APPROVED"} else 2
    except GateError as exc:
        result = {
            "schema": "product-fidelity-gate-error.v1",
            "status": "BLOCKED",
            "message": str(exc),
        }
        write_receipt(result, args.receipt.expanduser().resolve() if args.receipt else None)
        return 2


if __name__ == "__main__":
    sys.exit(main())
