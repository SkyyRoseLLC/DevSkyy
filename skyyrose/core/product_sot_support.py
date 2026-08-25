"""Catalog-wide support audit for the canonical hash-bound product SOT.

``product_sot`` establishes product identity and source authority.  This module
does not dilute that authority or write generated media.  It makes the support
needed by each downstream use explicit and reports a product as blocked until
the required evidence is actually registered.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from skyyrose.core import product_sot

REPO_ROOT = product_sot.REPO_ROOT
OPENING_MEDIA_PATH = (
    REPO_ROOT
    / "wordpress-theme"
    / "skyyrose-flagship-2"
    / "data"
    / "opening-product-media.json"
)

SUPPORT_AREAS = (
    "catalog_identity",
    "commerce_catalog",
    "storefront_media",
    "exact_product_authority",
    "reconstruction_capture",
    "measured_fit_and_fulfillment",
    "rights_and_promotion",
)

EXACT_AUTHORITY_KINDS = {
    "physical_product_photo",
    "physical_product_video",
    "approved_design_reference",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_opening_media(product_sot_bytes: bytes) -> dict[str, Any]:
    if not OPENING_MEDIA_PATH.is_file():
        raise ValueError(f"opening product-media manifest missing: {OPENING_MEDIA_PATH}")
    document = json.loads(OPENING_MEDIA_PATH.read_text(encoding="utf-8"))
    expected = hashlib.sha256(product_sot_bytes).hexdigest()
    if document.get("product_sot_sha256") != expected:
        raise ValueError("opening product-media manifest has stale product SOT hash")
    return document


def _artifact_errors(sku: str, product: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    artifacts = [*product.get("references", []), *product.get("media", {}).values()]
    for artifact in artifacts:
        path_value = artifact.get("path")
        if not isinstance(path_value, str) or not path_value:
            errors.append("ARTIFACT_PATH_MISSING")
            continue
        path = REPO_ROOT / path_value
        if not path.is_file():
            errors.append(f"ARTIFACT_MISSING:{path_value}")
            continue
        expected_hash = artifact.get("sha256")
        if not isinstance(expected_hash, str) or _sha256(path) != expected_hash:
            errors.append(f"ARTIFACT_HASH_DRIFT:{path_value}")
    source = product.get("source", {})
    dossier = source.get("dossier")
    if not isinstance(dossier, str) or not (REPO_ROOT / dossier).is_file():
        errors.append("DOSSIER_MISSING")
    elif _sha256(REPO_ROOT / dossier) != source.get("dossier_sha256"):
        errors.append("DOSSIER_HASH_DRIFT")
    return errors


def _is_ghost_media(media: dict[str, Any]) -> bool:
    value = media.get("on_model_front", {}).get("path", "")
    return isinstance(value, str) and "/ghost/" in value


def _support_entry(sku: str, product: dict[str, Any], media_record: dict[str, Any]) -> dict[str, Any]:
    artifacts = [*product.get("references", []), *product.get("media", {}).values()]
    authority_kinds = {artifact.get("kind") for artifact in artifacts}
    artifact_errors = _artifact_errors(sku, product)
    views = media_record.get("views", []) if isinstance(media_record, dict) else []
    opening_status = media_record.get("status") if isinstance(media_record, dict) else None
    approved_front = any(
        isinstance(view, dict) and view.get("role") == "on_model_front" for view in views
    )

    areas: dict[str, dict[str, Any]] = {
        "catalog_identity": {
            "state": "SUPPORTED" if not artifact_errors else "BLOCKED_ARTIFACT_INTEGRITY",
            "requirements": ["sku", "identity", "dossier_hash", "product_hash"],
            "blockers": artifact_errors,
        },
        "commerce_catalog": {
            "state": "SUPPORTED",
            "requirements": ["price", "sizes", "colors", "published", "is_preorder"],
            "runtime_authority": "WooCommerce owns live price, stock, variation availability, and permalink.",
        },
        "storefront_media": {
            "state": "SUPPORTED" if approved_front else "BLOCKED_MEDIA_APPROVAL",
            "approval_status": opening_status or "APPROVED_VIEW_SET_PRESENT",
            "approved_views": [view.get("role") for view in views if isinstance(view, dict)],
            "blockers": [] if approved_front else ["CURRENT_APPROVED_ON_MODEL_FRONT_MISSING"],
        },
        "exact_product_authority": {
            "state": "SUPPORTED" if authority_kinds & EXACT_AUTHORITY_KINDS else "BLOCKED_EXACT_AUTHORITY",
            "proof_level": product.get("verification", {}).get("proof_level"),
            "registered_authority_kinds": sorted(kind for kind in authority_kinds if isinstance(kind, str)),
            "blockers": [] if authority_kinds & EXACT_AUTHORITY_KINDS else ["VERIFIED_PHYSICAL_OR_DESIGN_AUTHORITY_MISSING"],
            "prohibited_substitutes": ["ghost_media", "generated_preview", "placeholder", "rejected_composite"],
        },
        "reconstruction_capture": {
            "state": "BLOCKED_CAPTURE_RECEIPT_REQUIRED",
            "requirements": [
                "same physical SKU/colorway/configuration/size",
                "front,left,right,back,top,bottom calibrated views",
                "scale and neutral colour target",
                "construction, artwork, material, trim, and hardware detail",
                "hash-bound local capture receipt",
                "pattern/CAD, verified scan, or founder-approved measured mesh for 3D release",
            ],
            "blockers": ["CATALOG_WIDE_CAPTURE_RECEIPT_NOT_REGISTERED"],
        },
        "measured_fit_and_fulfillment": {
            "state": "BLOCKED_CANONICAL_DETAILS_REQUIRED",
            "requirements": [
                "garment measurements and grading basis",
                "fit note and model height/worn size when on-model media is used",
                "materials and care",
                "shipping weight and package dimensions",
                "explicit SKU-level policy exception or inherited policy version",
            ],
            "blockers": ["CANONICAL_PER_SKU_MEASUREMENT_FIT_MATERIAL_CARE_FULFILLMENT_DATA_NOT_REGISTERED"],
        },
        "rights_and_promotion": {
            "state": "BLOCKED_EXPLICIT_AUTHORIZATION_REQUIRED",
            "requirements": [
                "source owner/capture operator",
                "usage rights and restriction",
                "founder approval identifier where promotion is requested",
                "separate product-fidelity, native-integration, and wiring decisions",
            ],
            "blockers": ["SOURCE_RIGHTS_AND_PROMOTION_AUTHORIZATION_NOT_REGISTERED"],
        },
    }
    if _is_ghost_media(product.get("media", {})):
        areas["storefront_media"]["ghost_source_warning"] = (
            "Catalog ghost media is not on-model, same-pose, reconstruction, or native-scene authority."
        )
    return {"sku": sku, "published": product["commerce"]["published"], "areas": areas}


def build_support_report() -> dict[str, Any]:
    """Return a deterministic support matrix without altering product authority."""
    current_bytes = product_sot.MANIFEST_PATH.read_bytes()
    expected = product_sot.serialize_manifest().encode("utf-8")
    if current_bytes != expected:
        raise ValueError("data/product-sot.json is stale; regenerate before support audit")
    manifest = json.loads(current_bytes)
    media = _load_opening_media(current_bytes).get("products", {})
    products = manifest.get("products", {})
    if set(products) != set(media):
        raise ValueError("opening product-media SKU set differs from product SOT")
    entries = [_support_entry(sku, products[sku], media[sku]) for sku in sorted(products)]
    counts = {
        area: dict(Counter(entry["areas"][area]["state"] for entry in entries))
        for area in SUPPORT_AREAS
    }
    return {
        "schema": "skyyrose.product-sot-support-report.v1",
        "candidate_only": True,
        "product_sot": {
            "path": "data/product-sot.json",
            "sha256": hashlib.sha256(current_bytes).hexdigest(),
            "sku_count": len(entries),
        },
        "support_areas": list(SUPPORT_AREAS),
        "summary": {"by_area_state": counts},
        "products": entries,
        "promotion_boundary": (
            "A supported SOT domain does not approve imagery, a native scene, a WooCommerce write, "
            "or deployment. Those require their separate review and authorization gates."
        ),
    }


def strict_blockers(report: dict[str, Any]) -> list[str]:
    """Return each published SKU/domain that has not met its full support contract."""
    blocked: list[str] = []
    for product in report["products"]:
        if not product["published"]:
            continue
        for area in SUPPORT_AREAS:
            if product["areas"][area]["state"] != "SUPPORTED":
                blocked.append(f"{product['sku']}:{area}:{product['areas'][area]['state']}")
    return blocked
