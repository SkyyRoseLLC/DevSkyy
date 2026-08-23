#!/usr/bin/env python3
"""Validate active SKU, Woo projection, and approved on-model card coverage."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
THEME = Path(__file__).resolve().parents[1]
PRODUCT_SOT = ROOT / "data/product-sot.json"
MEDIA_MANIFEST = THEME / "data/opening-product-media.json"
CATALOG = THEME / "data/skyyrose-catalog.csv"
WOO_SYNC = THEME / "data/woocommerce-product-sync.json"

ALLOWED_ROLES = {
    "on_model_front",
    "on_model_back",
    "mannequin_front",
    "mannequin_back",
    "render_3d",
    "packshot",
}
BLOCKED_STATUSES = {
    "MISSING_APPROVED_ON_MODEL_FRONT",
    "REJECTED_AUTHENTICITY",
    "FOUNDER_REVIEW_REQUIRED",
    "STALE_PRODUCT_HASH",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> int:
    product_bytes = PRODUCT_SOT.read_bytes()
    product_sot = json.loads(product_bytes)
    media = json.loads(MEDIA_MANIFEST.read_text(encoding="utf-8"))
    woo_sync = json.loads(WOO_SYNC.read_text(encoding="utf-8"))

    products = product_sot.get("products", {})
    active = {
        sku: product
        for sku, product in products.items()
        if product.get("commerce", {}).get("published") is True
    }
    if not active:
        raise ValueError("product SOT has no active products")
    if media.get("product_sot_sha256") != hashlib.sha256(product_bytes).hexdigest():
        raise ValueError("production media manifest is not bound to current product SOT")

    media_products = media.get("products", {})
    product_hashes = media.get("product_hashes", {})
    integrity = media.get("asset_integrity", {})
    if set(active) != set(media_products) or set(active) != set(product_hashes):
        raise ValueError("active SKU set differs from product media manifest")

    with CATALOG.open(newline="", encoding="utf-8-sig") as catalog_file:
        catalog_rows = {
            row["sku"]: row
            for row in csv.DictReader(catalog_file)
            if row.get("published") == "1"
        }
    if set(active) != set(catalog_rows):
        missing = sorted(set(active) - set(catalog_rows))
        extra = sorted(set(catalog_rows) - set(active))
        raise ValueError(f"catalog SKU drift; missing={missing}, extra={extra}")

    woo_rows = woo_sync.get("products", [])
    woo_products = {
        row.get("sku"): row
        for row in woo_rows
        if isinstance(row, dict) and isinstance(row.get("sku"), str)
    }
    if set(active) != set(woo_products):
        raise ValueError("WooCommerce projection SKU set differs from active product SOT")

    approved: list[str] = []
    blocked: list[str] = []
    collection_counts: dict[str, dict[str, int]] = {}
    for sku, product in active.items():
        collection = product.get("identity", {}).get("collection")
        record = media_products.get(sku, {})
        if record.get("collection") != collection:
            raise ValueError(f"collection mismatch for {sku}")
        if product_hashes.get(sku) != product.get("product_hash"):
            raise ValueError(f"stale product hash for {sku}")

        woo = woo_products[sku].get("product", {})
        if woo.get("status") != "publish" or woo.get("catalog_visibility") != "visible":
            raise ValueError(f"WooCommerce visibility drift for {sku}")

        views = record.get("views", [])
        counts = collection_counts.setdefault(collection, {"active": 0, "approved": 0, "blocked": 0})
        counts["active"] += 1
        if not views:
            if record.get("status") not in BLOCKED_STATUSES or not record.get("reason"):
                raise ValueError(f"blocked SKU lacks a fail-closed verdict: {sku}")
            if sku in integrity:
                raise ValueError(f"blocked SKU has approved integrity data: {sku}")
            blocked.append(sku)
            counts["blocked"] += 1
            continue

        roles = [view.get("role") for view in views]
        if roles[0] != "on_model_front" or any(role not in ALLOWED_ROLES for role in roles):
            raise ValueError(f"approved media roles are invalid for {sku}")
        if len(roles) != len(set(roles)) or set(integrity.get(sku, {})) != set(roles):
            raise ValueError(f"approved media integrity roles drift for {sku}")

        for view in views:
            source = view.get("source", "")
            source_path = (THEME / source).resolve()
            if (
                not isinstance(source, str)
                or Path(source).is_absolute()
                or ".." in Path(source).parts
                or not source_path.is_file()
            ):
                raise ValueError(f"approved source does not resolve for {sku}")
            proof = integrity[sku][view["role"]]
            if source_path.stat().st_size != proof.get("source_bytes") or sha256(source_path) != proof.get("source_sha256"):
                raise ValueError(f"approved source integrity drift for {sku}")

        approved.append(sku)
        counts["approved"] += 1

    for collection, counts in sorted(collection_counts.items()):
        print(
            f"{collection}: {counts['active']} active, "
            f"{counts['approved']} approved on-model, {counts['blocked']} blocked"
        )
    print(f"TOTAL: {len(active)} active, {len(approved)} approved on-model, {len(blocked)} blocked")
    if blocked:
        print("BLOCKED SKUS: " + ", ".join(sorted(blocked)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
