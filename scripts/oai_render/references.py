"""Product reference resolution backed exclusively by ``product-sot.json``.

No SKU filename guessing, directory globbing, catalog fallback, or legacy
hardcoded source map is allowed. A stale or incomplete product contract fails
closed before a paid image request can run.
"""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from skyyrose.core import product_sot

from . import config

log = logging.getLogger(__name__)
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".avif"}
JERSEY_PATCH_LOGOS = {
    "mlb-authentic-collection-card",
    "nfl-authentic-collection-card",
    "nba-authentic-collection-card",
    "hockey-championship-card",
}


class MissingReferenceError(RuntimeError):
    """Raised when product SOT cannot provide required visual proof."""


@dataclass(frozen=True)
class ReferenceImage:
    """One labeled, hash-bound reference fed to image generation."""

    label: str
    path: Path
    kind: str


@dataclass(frozen=True)
class Pair:
    """Two separately sold products approved for one coordinated look."""

    pair_id: str
    collection: str
    skus: tuple[str, str]
    label: str


PAIRS: tuple[Pair, ...] = (
    Pair("br-rose-set", "black-rose", ("br-001", "br-002"), "BLACK Rose Crewneck + Joggers"),
    Pair(
        "sg-baybridge-set",
        "signature",
        ("sg-001", "sg-005"),
        "Bridge Series 'The Bay Bridge' Shorts + Shirt",
    ),
    Pair(
        "sg-staygolden-set",
        "signature",
        ("sg-002", "sg-003"),
        "Bridge Series 'Stay Golden' Shirt + Shorts",
    ),
    Pair(
        "sg-mintlav-set",
        "signature",
        ("sg-013", "sg-014"),
        "Mint & Lavender Crewneck + Sweatpants",
    ),
    Pair(
        "lh-bomber-black",
        "love-hurts",
        ("lh-004", "lh-002"),
        "Love Hurts Bomber + Joggers (Black)",
    ),
    Pair(
        "lh-bomber-white",
        "love-hurts",
        ("lh-004", "lh-006"),
        "Love Hurts Bomber + Joggers (White)",
    ),
)


def _manifest() -> dict[str, Any]:
    manifest = product_sot.load_manifest()
    expected = product_sot.serialize_manifest()
    if product_sot.MANIFEST_PATH.read_text(encoding="utf-8") != expected:
        raise MissingReferenceError(
            "data/product-sot.json is stale; regenerate before dispatching creative work"
        )
    return manifest


def _record(sku: str) -> dict[str, Any]:
    products = _manifest()["products"]
    if sku not in products:
        raise MissingReferenceError(f"{sku}: absent from canonical product SOT")
    return products[sku]


def _path(artifact: dict[str, Any]) -> Path:
    path = product_sot.REPO_ROOT / artifact["path"]
    if not path.is_file():
        raise MissingReferenceError(f"canonical artifact is missing: {artifact['path']}")
    if artifact.get("sha256") != hashlib.sha256(path.read_bytes()).hexdigest():
        raise MissingReferenceError(f"canonical artifact hash drifted: {artifact['path']}")
    return path


def _is_logo(artifact: dict[str, Any]) -> bool:
    role = artifact.get("role", "").lower()
    return artifact.get("kind") == "approved_logo_or_patch_art" or "logo_reference" in role


def _is_back(artifact: dict[str, Any]) -> bool:
    role = artifact.get("role", "").lower()
    return "back" in role or "back" in Path(artifact["path"]).stem.lower()


def _image_artifacts(record: dict[str, Any]) -> list[dict[str, Any]]:
    artifacts = [*record["references"], *record["media"].values()]
    return sorted(
        artifacts,
        key=lambda item: (
            not _is_logo(item),
            item.get("authority", 50),
        ),
        reverse=True,
    )


def load_catalog() -> dict[str, dict[str, Any]]:
    """Return generation-facing product fields from product SOT only."""
    return {
        sku: {
            "name": record["identity"]["name"],
            "collection": record["identity"]["collection"],
            "is_preorder": record["commerce"]["is_preorder"],
            "output_slug": record["rendering"]["output_slug"] or sku,
            "product_sot_hash": record["product_hash"],
        }
        for sku, record in _manifest()["products"].items()
    }


def get_source_map() -> dict[str, dict[str, Path | None]]:
    """Project exact front/back garment references from product SOT."""
    source_map: dict[str, dict[str, Path | None]] = {}
    for sku, record in _manifest()["products"].items():
        front: Path | None = None
        back: Path | None = None
        for artifact in _image_artifacts(record):
            path = Path(artifact["path"])
            if path.suffix.lower() not in IMAGE_SUFFIXES or _is_logo(artifact):
                continue
            resolved = _path(artifact)
            if _is_back(artifact):
                back = back or resolved
            else:
                front = front or resolved
        source_map[sku] = {"front": front, "back": back}
    return source_map


def requires_patch(sku: str) -> bool:
    """Return whether the product requires an exact 3×4-inch jersey patch."""
    return any(
        placement.get("logo_id") in JERSEY_PATCH_LOGOS
        for placement in _record(sku)["logo_placements"]
    )


def has_back_source(sku: str) -> bool:
    return get_source_map()[sku]["back"] is not None


def get_pairs_for_sku(sku: str) -> list[Pair]:
    return [pair for pair in PAIRS if sku in pair.skus]


def get_logo_reference(sku: str, collection: str) -> Path | None:
    record = _record(sku)
    if record["identity"]["collection"] != collection:
        raise MissingReferenceError(f"{sku}: collection mismatch")
    for artifact in _image_artifacts(record):
        path = Path(artifact["path"])
        if path.suffix.lower() in IMAGE_SUFFIXES and _is_logo(artifact):
            return _path(artifact)
    return None


def find_flatlay_photo(sku: str) -> Path | None:
    """Return physical product proof when explicitly registered in product SOT."""
    for artifact in _record(sku)["references"]:
        if artifact.get("kind") == "physical_product_photo":
            path = Path(artifact["path"])
            if path.suffix.lower() in IMAGE_SUFFIXES:
                return _path(artifact)
    return None


def build_dossier_index() -> dict[str, Path]:
    return {
        sku: product_sot.REPO_ROOT / record["source"]["dossier"]
        for sku, record in _manifest()["products"].items()
    }


def _reference_kind(artifact: dict[str, Any]) -> str:
    if _is_logo(artifact):
        name = Path(artifact["path"]).name.lower()
        if "patch" in name or "collection-card" in name or "championship-card" in name:
            return "patch"
        return "logo"
    return "garment-back" if _is_back(artifact) else "garment"


def _label(kind: str) -> str:
    if kind == "garment":
        return (
            "REFERENCE IMAGE {n} — CANONICAL GARMENT PROOF: match the physical product or "
            "approved design exactly. Preserve fabric, color, construction, typography, "
            "logo placement, and scale; do not invent hidden details."
        )
    if kind == "garment-back":
        return (
            "REFERENCE IMAGE {n} — CANONICAL BACK GARMENT PROOF: preserve the exact rear "
            "panel layout and branding."
        )
    return (
        "REFERENCE IMAGE {n} — APPROVED LOGO/PATCH ART: reproduce this exact mark; do not "
        "redraw, resize, reposition, recolor, duplicate, or omit it."
    )


def build_references(
    sku: str,
    collection: str,
    *,
    include_back: bool = True,
    view: str = "front",
) -> list[ReferenceImage]:
    """Build the paid-generation reference set from exact SOT artifacts."""
    record = _record(sku)
    if record["identity"]["collection"] != collection:
        raise MissingReferenceError(
            f"{sku}: requested collection {collection!r} does not match product SOT"
        )

    refs: list[ReferenceImage] = []
    seen: set[Path] = set()
    for artifact in _image_artifacts(record):
        path = Path(artifact["path"])
        if path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        resolved = _path(artifact)
        if resolved in seen:
            continue
        kind = _reference_kind(artifact)
        if kind == "garment-back" and not include_back:
            continue
        seen.add(resolved)
        refs.append(ReferenceImage(label=_label(kind), path=resolved, kind=kind))

    if not any(reference.kind.startswith("garment") for reference in refs):
        raise MissingReferenceError(f"{sku}: product SOT has no usable garment proof")

    if requires_patch(sku):
        placements = [
            item for item in record["logo_placements"] if item.get("logo_id") in JERSEY_PATCH_LOGOS
        ]
        if not any(reference.kind == "patch" for reference in refs):
            raise MissingReferenceError(f"{sku}: required approved jersey patch art is missing")
        if any(
            item.get("width_inches") != 3 or item.get("height_inches") != 4
            for item in placements
        ):
            raise MissingReferenceError(f"{sku}: jersey patch must be exactly 3in wide × 4in tall")
        refs = [
            ReferenceImage(
                label=(
                    reference.label
                    + " Authentication patch physical size: EXACTLY 3 inches wide × 4 inches "
                    "tall; all text must remain readable."
                ),
                path=reference.path,
                kind=reference.kind,
            )
            if reference.kind == "patch"
            else reference
            for reference in refs
        ]

    if view == "back":
        refs.sort(key=lambda reference: 0 if reference.kind == "garment-back" else 1)
    capped = refs[: config.MAX_REFERENCE_IMAGES]
    return [
        ReferenceImage(
            label=reference.label.format(n=index + 1),
            path=reference.path,
            kind=reference.kind,
        )
        for index, reference in enumerate(capped)
    ]
