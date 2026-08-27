#!/usr/bin/env python3
"""Create a local, read-only WooCommerce authority receipt for the V2 catalog.

The script never mutates WordPress or WooCommerce. It reads the configured
store with existing credentials, redacts the endpoint from the receipt, and
records only fields needed to prove the V2 SKU-to-product binding.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from dotenv import load_dotenv

REPO_ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = REPO_ROOT / "wordpress-theme/skyyrose-flagship/data/skyyrose-catalog.csv"
DEFAULT_OUTPUT = REPO_ROOT / ".fashion-theme/woocommerce-runtime-capture-2026-08-26.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def configuration() -> tuple[str, str, str]:
    load_dotenv(REPO_ROOT / ".env.wordpress")
    load_dotenv(REPO_ROOT / "wordpress/.env", override=False)
    base = (
        os.getenv("WORDPRESS_URL") or os.getenv("WP_URL") or os.getenv("WP_SITE_URL") or ""
    ).rstrip("/")
    key = os.getenv("WC_CONSUMER_KEY") or os.getenv("WOOCOMMERCE_KEY") or ""
    secret = os.getenv("WC_CONSUMER_SECRET") or os.getenv("WOOCOMMERCE_SECRET") or ""
    if not (base and key and secret):
        raise RuntimeError("Missing WooCommerce URL or read credentials.")
    return base, key, secret


def catalog() -> dict[str, dict[str, str]]:
    with CATALOG_PATH.open(newline="", encoding="utf-8") as handle:
        return {row["sku"]: row for row in csv.DictReader(handle)}


def product_record(product: dict[str, Any]) -> dict[str, Any]:
    return {
        "product_id": product["id"],
        "type": product.get("type"),
        "status": product.get("status"),
        "purchasable": product.get("purchasable"),
        "stock_status": product.get("stock_status"),
        "categories": sorted(
            category.get("slug", "") for category in product.get("categories", [])
        ),
        "variation_ids": product.get("variations", []),
        "image_attachment_ids": [
            image.get("id") for image in product.get("images", []) if image.get("id")
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    base, key, secret = configuration()
    response = requests.get(
        f"{base}/wp-json/wc/v3/products",
        params={"per_page": 100, "status": "any", "orderby": "id", "order": "asc"},
        auth=(key, secret),
        timeout=30,
    )
    response.raise_for_status()
    products = response.json()
    if not isinstance(products, list):
        raise RuntimeError("WooCommerce products response was not a list.")

    expected = catalog()
    by_sku = {product.get("sku"): product for product in products if product.get("sku")}
    missing = sorted(set(expected) - set(by_sku))
    bindings = {sku: product_record(by_sku[sku]) for sku in sorted(expected) if sku in by_sku}
    unpublished = sorted(sku for sku, item in bindings.items() if item["status"] != "publish")
    zero_image_skus = sorted(
        sku for sku, item in bindings.items() if not item["image_attachment_ids"]
    )
    extra = {
        sku: {"product_id": product["id"], "status": product.get("status")}
        for sku, product in sorted(by_sku.items())
        if sku not in expected
    }
    receipt = {
        "schema_version": "1.0",
        "kind": "skyyrose-v2-woocommerce-read-authority",
        "captured_at_utc": datetime.now(UTC)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z"),
        "method": "read-only authenticated WooCommerce REST GET /wp-json/wc/v3/products",
        "source_commit": git_head(),
        "catalog": {
            "path": str(CATALOG_PATH.relative_to(REPO_ROOT)),
            "sha256": sha256(CATALOG_PATH),
            "expected_sku_count": len(expected),
        },
        "result": {
            "received_product_count": len(products),
            "exact_catalog_sku_bindings": len(bindings),
            "missing_catalog_skus": missing,
            "unpublished_catalog_skus": unpublished,
            "zero_image_skus": zero_image_skus,
            "extra_store_skus": extra,
        },
        "products": bindings,
        "limits": [
            "This receipt proves only the read-time WooCommerce SKU/product state.",
            "It does not prove that a WooCommerce attachment byte matches a local SOT source or that a product scene depicts the bound SKU.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {args.output.relative_to(REPO_ROOT)}")
    print(json.dumps(receipt["result"], indent=2, sort_keys=True))
    return 0 if not missing and not unpublished else 1


if __name__ == "__main__":
    raise SystemExit(main())
