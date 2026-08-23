"""Approved product-imagery resolver backed by the canonical product SOT.

Product facts and media paths come from ``data/product-sot.json``. On-model or
mannequin media becomes storefront-eligible only when it is present in the
hash-bound V2 approval manifest. Rejected catalog candidates never win a
fallback merely because a filename exists.
"""

from __future__ import annotations

import functools
import hashlib
import json
from pathlib import Path
from typing import Literal

from skyyrose.core import product_sot
from skyyrose.core.paths import REPO_ROOT

APPROVAL_MANIFEST_PATH = (
    REPO_ROOT
    / "wordpress-theme"
    / "skyyrose-flagship-2"
    / "data"
    / "opening-product-media.json"
)
MANIFEST_PATH = REPO_ROOT / "data" / "sot-images.json"

Role = Literal["front", "back", "packshot", "back_packshot"]
_ROLE_KEYS: dict[str, tuple[str, ...]] = {
    "front": ("front_model_image", "image"),
    "back": ("back_model_image", "back_image"),
    "packshot": ("image",),
    "back_packshot": ("back_image",),
}
_APPROVED_FIELDS = {
    "on_model_front": "front_model_image",
    "on_model_back": "back_model_image",
    "mannequin_front": "mannequin_front_image",
    "mannequin_back": "mannequin_back_image",
    "render_3d": "render_3d_image",
}
_THEME_PREFIX = "wordpress-theme/skyyrose-flagship/"


def _validated_path(raw: str, sku: str) -> str:
    normalized = raw[len(_THEME_PREFIX) :] if raw.startswith(_THEME_PREFIX) else raw
    if (
        normalized.startswith("/")
        or ".." in Path(normalized).parts
        or not normalized.startswith("assets/")
    ):
        raise ValueError(f"product SOT image path escapes the assets tree for {sku!r}: {raw!r}")
    return normalized


def _entry(path: str, sku: str) -> dict[str, str]:
    validated = _validated_path(path, sku)
    return {"path": validated, "resolved": validated}


def _approval_manifest(product_sot_bytes: bytes) -> dict:
    approval = json.loads(APPROVAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(product_sot_bytes).hexdigest()
    if approval.get("product_sot_sha256") != expected_hash:
        raise ValueError("opening product-media approvals are stale relative to product-sot.json")
    product_manifest = json.loads(product_sot_bytes).get("products", {})
    approved_products = approval.get("products", {})
    approved_hashes = approval.get("product_hashes", {})
    if not isinstance(approved_products, dict) or set(approved_products) != set(product_manifest):
        raise ValueError("opening product-media approval SKU set differs from product-sot.json")
    if not isinstance(approved_hashes, dict) or set(approved_hashes) != set(product_manifest):
        raise ValueError("opening product-media hash SKU set differs from product-sot.json")
    for sku, product in product_manifest.items():
        if approved_hashes.get(sku) != product.get("product_hash"):
            raise ValueError(f"opening product-media approval is stale for {sku}")
    return approval


@functools.lru_cache(maxsize=1)
def _index() -> dict[str, dict]:
    product_sot_bytes = product_sot.MANIFEST_PATH.read_bytes()
    manifest = json.loads(product_sot_bytes)
    approvals = _approval_manifest(product_sot_bytes).get("products", {})
    result: dict[str, dict] = {}
    for sku, product in manifest.get("products", {}).items():
        images: dict[str, dict[str, str]] = {}
        media = product.get("media", {})
        if media.get("packshot_front", {}).get("path"):
            images["image"] = _entry(media["packshot_front"]["path"], sku)
        if media.get("packshot_back", {}).get("path"):
            images["back_image"] = _entry(media["packshot_back"]["path"], sku)

        for view in approvals.get(sku, {}).get("views", []):
            field = _APPROVED_FIELDS.get(view.get("role", ""))
            path = view.get("source")
            if field and isinstance(path, str) and path:
                images[field] = _entry(path, sku)

        result[sku] = {
            "sku": sku,
            "name": product.get("identity", {}).get("name", ""),
            "collection": product.get("identity", {}).get("collection", ""),
            "product_sot_hash": product.get("product_hash", ""),
            "images": images,
        }
    return result


def refresh() -> None:
    _index.cache_clear()


def all_skus() -> list[str]:
    return sorted(_index())


def _first_path(images: dict, keys: tuple[str, ...], sku: str) -> str | None:
    for key in keys:
        entry = images.get(key)
        if isinstance(entry, dict) and entry.get("path"):
            return _validated_path(entry["path"], sku)
    return None


def resolve_image(sku: str, role: Role = "front") -> str | None:
    if not sku:
        return None
    product = _index().get(sku)
    if not product:
        return None
    return _first_path(product.get("images", {}), _ROLE_KEYS.get(role, ()), sku)


def has_render(sku: str) -> bool:
    product = _index().get(sku)
    if not product:
        return False
    entry = product.get("images", {}).get("front_model_image")
    return isinstance(entry, dict) and bool(entry.get("path"))


def build_manifest() -> dict[str, dict[str, str]]:
    manifest: dict[str, dict[str, str]] = {}
    for sku, product in sorted(_index().items()):
        images = product.get("images", {})
        entry: dict[str, str] = {}
        for role, keys in _ROLE_KEYS.items():
            path = _first_path(images, keys, sku)
            if path:
                entry[role] = path
        if entry:
            manifest[sku] = entry
    return manifest


def serialize_manifest(manifest: dict | None = None) -> str:
    product_manifest = product_sot.build_manifest()
    product_manifest_bytes = product_sot.serialize_manifest(product_manifest).encode("utf-8")
    payload = {
        "_generated_by": "skyyrose.core.sot_images.write_manifest — DO NOT EDIT",
        "_authority": (
            "Product-SOT media with hash-bound approval gating. Approved on-model media wins; "
            "otherwise the canonical packshot remains the fail-closed fallback."
        ),
        "product_sot": {
            "path": "data/product-sot.json",
            "sha256": hashlib.sha256(product_manifest_bytes).hexdigest(),
            "product_hashes": {
                sku: product["product_hash"]
                for sku, product in product_manifest["products"].items()
            },
        },
        "approval_manifest": (
            "wordpress-theme/skyyrose-flagship-2/data/opening-product-media.json"
        ),
        "images": build_manifest() if manifest is None else manifest,
    }
    return json.dumps(payload, indent=2) + "\n"


def write_manifest(out_path: Path | None = None, manifest: dict | None = None) -> Path:
    output = out_path or MANIFEST_PATH
    if out_path is not None and not str(output.resolve()).startswith(str(REPO_ROOT.resolve())):
        raise ValueError(f"write_manifest: out_path must be within the repo: {out_path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(serialize_manifest(manifest), encoding="utf-8")
    return output


if __name__ == "__main__":
    images = build_manifest()
    path = write_manifest(manifest=images)
    print(f"wrote {path} ({len(images)} skus)")
