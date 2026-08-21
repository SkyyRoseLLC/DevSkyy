#!/usr/bin/env python3
"""Fail closed when approved V2 card media drifts from the product SOT."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
THEME = Path(__file__).resolve().parents[1]
SOURCE_THEME = ROOT / "wordpress-theme/skyyrose-flagship"
PRODUCT_SOT = ROOT / "data/product-sot.json"
MEDIA_MANIFEST = THEME / "data/opening-product-media.json"


def main() -> int:
    product_sot_bytes = PRODUCT_SOT.read_bytes()
    product_sot = json.loads(product_sot_bytes)
    media = json.loads(MEDIA_MANIFEST.read_text(encoding="utf-8"))
    expected_sha = hashlib.sha256(product_sot_bytes).hexdigest()
    if media.get("product_sot_sha256") != expected_sha:
        raise ValueError("opening-product-media.json is not bound to the current product SOT")

    products = product_sot.get("products", {})
    media_products = media.get("products", {})
    if set(products) != set(media_products):
        raise ValueError("opening product-media SKU set differs from the product SOT")

    for sku, record in media_products.items():
        for view in record.get("views", []):
            for field in ("source", "derivative"):
                value = view.get(field)
                root = SOURCE_THEME if field == "source" else THEME
                if value and not (root / value).is_file():
                    raise ValueError(f"opening product media path does not resolve: {sku} {value}")
    print(f"Opening product media is bound to the current product SOT ({len(products)} SKUs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
