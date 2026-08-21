"""Canonical machine-facing SkyyRose product source of truth.

Human-approved facts remain authored in the catalog CSV, per-product dossiers,
and logo registry. This module compiles those inputs into one deterministic,
hash-bound contract consumed by imagery, video, website, ads, and WooCommerce
pipelines. Downstream code must read this contract instead of independently
rejoining the three authoring sources.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from skyyrose.core.catalog_loader import bool_col, read_catalog_rows
from skyyrose.core.dossier_loader import load_dossier
from skyyrose.core.dossier_schema import DossierSchema

REPO_ROOT = Path(__file__).resolve().parents[2]
THEME_ROOT = REPO_ROOT / "wordpress-theme" / "skyyrose-flagship"
DATA_ROOT = THEME_ROOT / "data"
CATALOG_PATH = DATA_ROOT / "skyyrose-catalog.csv"
LOGO_REGISTRY_PATH = DATA_ROOT / "logo-registry.json"
MANIFEST_PATH = REPO_ROOT / "data" / "product-sot.json"

SCHEMA = "skyyrose.product-sot.v1"
IMAGE_COLUMNS = (
    ("packshot_front", "image"),
    ("on_model_front", "front_model_image"),
    ("packshot_back", "back_image"),
    ("on_model_back", "back_model_image"),
)
CREATIVE_CONSUMERS = (
    "image_generation",
    "video_generation",
    "website",
    "advertising",
    "woocommerce",
)


class ProductSOTError(ValueError):
    """Raised when canonical product evidence cannot be resolved."""


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT.resolve()).as_posix()


def resolve_reference(value: str) -> Path:
    """Resolve a canonical reference without filename or SKU guessing."""
    reference = Path(value)
    candidates = (REPO_ROOT / reference, THEME_ROOT / reference, THEME_ROOT / "assets" / reference)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ProductSOTError(f"canonical product reference does not resolve: {value}")


def _evidence_kind(path: Path) -> str:
    normalized = path.as_posix().lower()
    if "source-photos" in normalized or "real-photo" in normalized or "real-" in path.name.lower():
        return "physical_product_photo"
    if "techflat" in normalized or "/split/" in normalized or "design-" in path.name.lower():
        return "approved_design_reference"
    if "brand-logos" in normalized or "logo" in path.name.lower() or "patch" in path.name.lower():
        return "approved_logo_or_patch_art"
    return "product_reference"


def _artifact(value: str, *, role: str, authority: int) -> dict[str, Any]:
    path = resolve_reference(value)
    return {
        "role": role,
        "authority": authority,
        "kind": _evidence_kind(path),
        "path": _relative(path),
        "sha256": _sha256_file(path),
        "bytes": path.stat().st_size,
    }


def _media(row: dict[str, str]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for role, column in IMAGE_COLUMNS:
        value = (row.get(column) or "").strip()
        if not value:
            continue
        result[role] = _artifact(value, role=role, authority=50)
    return result


def _logo_reference_artifacts(
    value: str,
    *,
    sku: str,
    role: str,
    authority: int,
) -> list[dict[str, Any]]:
    artifacts = [_artifact(value, role=role, authority=authority)]
    doc_path = resolve_reference(value)
    if doc_path.suffix.lower() != ".md":
        return artifacts
    text = doc_path.read_text(encoding="utf-8")
    end = text.find("\n---", 3)
    frontmatter = text[3:end] if text.startswith("---") and end != -1 else ""
    sku_match = re.search(rf"^[ \t]+{re.escape(sku)}:\s*(.+?)\s*$", frontmatter, re.MULTILINE)
    image_match = re.search(r"^image_path:\s*(.+?)\s*$", frontmatter, re.MULTILINE)
    image_path = sku_match.group(1) if sku_match else image_match.group(1) if image_match else ""
    if image_path:
        artifacts.append(
            _artifact(
                image_path.strip().strip('"').strip("'"),
                role=f"{role}_art",
                authority=authority + 1,
            )
        )
    return artifacts


def _reference_stack(dossier: Any, sku: str) -> list[dict[str, Any]]:
    stack: list[dict[str, Any]] = []
    if dossier.reference_image:
        stack.append(_artifact(dossier.reference_image, role="primary_garment_reference", authority=100))
    for index, value in enumerate(dossier.extra_references, start=1):
        stack.append(_artifact(value, role=f"garment_reference_{index}", authority=90 - index))
    if dossier.logo_reference:
        stack.extend(
            _logo_reference_artifacts(
                dossier.logo_reference,
                sku=sku,
                role="primary_logo_reference",
                authority=80,
            )
        )
    for index, value in enumerate(dossier.extra_logos, start=1):
        stack.extend(
            _logo_reference_artifacts(
                value,
                sku=sku,
                role=f"logo_reference_{index}",
                authority=70 - index,
            )
        )
    return stack


def _proof_level(reference_stack: list[dict[str, Any]], media: dict[str, Any]) -> str:
    garment_references = [
        item for item in reference_stack if "garment_reference" in item["role"]
    ]
    kinds = {item["kind"] for item in garment_references}
    if "physical_product_photo" in kinds:
        return "physical_product_photo"
    if "approved_design_reference" in kinds:
        return "approved_design_reference"
    if garment_references:
        return "canonical_product_reference"
    if media:
        return "catalog_media_only"
    return "blocked_missing_visual_proof"


def _logo_placements(registry: dict[str, Any], sku: str) -> list[dict[str, Any]]:
    product = registry.get("sku_logos", {}).get(sku)
    if not isinstance(product, dict):
        raise ProductSOTError(f"logo registry has no sku_logos record for {sku}")
    placements = product.get("placements")
    if not isinstance(placements, list):
        raise ProductSOTError(f"logo registry placements are invalid for {sku}")
    return placements


def _product_hash(record: dict[str, Any]) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return _sha256_bytes(encoded)


def build_product(row: dict[str, str], registry: dict[str, Any]) -> dict[str, Any]:
    sku = row["sku"].strip()
    dossier_slug = (row.get("dossier_slug") or "").strip()
    dossier = load_dossier(dossier_slug)
    schema = DossierSchema.from_raw(dossier)
    dossier_path = DATA_ROOT / "dossiers" / f"{dossier_slug}.md"
    references = _reference_stack(dossier, sku)
    media = _media(row)
    sizes = [item.strip() for item in (row.get("sizes") or "").split("|") if item.strip()]
    colors = [item.strip() for item in (row.get("color") or "").split("|") if item.strip()]

    record: dict[str, Any] = {
        "sku": sku,
        "identity": {
            "name": row["name"].strip(),
            "collection": row["collection"].strip(),
            "description": (row.get("description") or "").strip(),
            "badge": (row.get("badge") or "").strip(),
        },
        "commerce": {
            "price": (row.get("price") or "").strip(),
            "sizes": sizes,
            "colors": colors,
            "edition_size": (row.get("edition_size") or "").strip(),
            "published": bool_col(row, "published"),
            "is_preorder": bool_col(row, "is_preorder"),
        },
        "garment": {
            "type_lock": schema.garment_type_lock,
            "branding_summary": (row.get("branding_spec") or "").strip(),
            "branding_regions": [region.model_dump(mode="json") for region in schema.branding],
            "negative_constraints": schema.negative,
        },
        "scene_direction": {
            "pose": schema.scene_pose,
            "setting": schema.scene_setting,
            "collection_hero_is_inspiration_only": True,
            "hero_assets_must_not_be_replaced": True,
        },
        "rendering": {
            "output_slug": (row.get("render_output_slug") or "").strip(),
            "source_override": (row.get("render_source_override") or "").strip(),
            "back_source_override": (row.get("render_back_source_override") or "").strip(),
            "is_tech_flat": bool_col(row, "render_is_tech_flat"),
            "is_accessory": bool_col(row, "render_is_accessory"),
            "engine_override": (row.get("engine_override") or "").strip(),
        },
        "logo_placements": _logo_placements(registry, sku),
        "references": references,
        "media": media,
        "verification": {
            "record_complete": True,
            "proof_level": _proof_level(references, media),
            "on_model_front_present": "on_model_front" in media,
            "media_approval_is_separate": True,
            "no_visual_invention_allowed": True,
        },
        "source": {
            "dossier": _relative(dossier_path),
            "dossier_sha256": _sha256_file(dossier_path),
        },
        "consumer_contract": {
            "required_for": list(CREATIVE_CONSUMERS),
            "reference_order": "ascending authority number is weaker; highest authority wins",
            "reject_stale_product_hash": True,
        },
    }
    record["product_hash"] = _product_hash(record)
    return record


def build_manifest() -> dict[str, Any]:
    registry = json.loads(LOGO_REGISTRY_PATH.read_text(encoding="utf-8"))
    products = {
        row["sku"].strip(): build_product(row, registry)
        for row in read_catalog_rows(CATALOG_PATH)
    }
    return {
        "schema": SCHEMA,
        "generated_by": "skyyrose.core.product_sot.write_manifest — DO NOT EDIT",
        "authority": (
            "Single machine-facing product contract for image generation, video generation, "
            "website, advertising, and WooCommerce. Edit catalog/dossier/logo sources, then regenerate."
        ),
        "sources": {
            "catalog": _relative(CATALOG_PATH),
            "catalog_sha256": _sha256_file(CATALOG_PATH),
            "logo_registry": _relative(LOGO_REGISTRY_PATH),
            "logo_registry_sha256": _sha256_file(LOGO_REGISTRY_PATH),
        },
        "products": products,
    }


def serialize_manifest(manifest: dict[str, Any] | None = None) -> str:
    return json.dumps(manifest or build_manifest(), indent=2, ensure_ascii=False) + "\n"


def write_manifest(path: Path = MANIFEST_PATH) -> Path:
    if not str(path.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError(f"product SOT output must stay inside repository: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(serialize_manifest(), encoding="utf-8")
    return path


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def get_product(sku: str, path: Path = MANIFEST_PATH) -> dict[str, Any]:
    products = load_manifest(path).get("products", {})
    if sku not in products:
        raise KeyError(f"SKU {sku!r} not found in {path}")
    return products[sku]


if __name__ == "__main__":
    output = write_manifest()
    print(f"wrote {output} ({len(load_manifest(output)['products'])} products)")
