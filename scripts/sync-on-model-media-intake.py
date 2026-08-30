#!/usr/bin/env python3
"""Bind the V2 on-model registry to every current product without inventing approval.

This synchronizer repairs registry shape and freshness only. It preserves existing
view records for audit, adds explicit missing/pending states, and never upgrades a
record to approved. Approval, integrity hashes, dual-cast authority, and founder
references must be supplied by their separate review workflows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-sot", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    sot_bytes = args.product_sot.read_bytes()
    sot = _load(args.product_sot)
    current = _load(args.registry) if args.registry.is_file() else {}
    sot_products = sot.get("products")
    if not isinstance(sot_products, dict):
        raise ValueError("product SOT requires a products object")

    prior_products = current.get("products") if isinstance(current.get("products"), dict) else {}
    products: dict[str, Any] = {}
    product_hashes: dict[str, str] = {}
    for sku, sot_product in sorted(sot_products.items()):
        if not isinstance(sot_product, dict) or not isinstance(sot_product.get("product_hash"), str):
            raise ValueError(f"invalid SOT product record: {sku}")
        previous = prior_products.get(sku) if isinstance(prior_products, dict) else None
        previous = previous if isinstance(previous, dict) else {}
        views = previous.get("views") if isinstance(previous.get("views"), list) else []
        status = previous.get("status")
        if status == "APPROVED_CURRENT_STOREFRONT_ON_MODEL_FRONT":
            # A source mutation invalidates approval. A separate review must restore it.
            status = "PENDING_HASH_BOUND_REVALIDATION"
        elif not isinstance(status, str) or not status:
            status = "LEGACY_RECORD_PENDING_REVALIDATION" if views else "MISSING_EXACT_ON_MODEL_AUTHORITY"
        identity = sot_product.get("identity") if isinstance(sot_product.get("identity"), dict) else {}
        entry: dict[str, Any] = {
            "collection": identity.get("collection"),
            "status": status,
            "views": views,
        }
        for key in ("approval_reference", "reviewed_at", "notes"):
            if key in previous:
                entry[key] = previous[key]
        products[sku] = entry
        product_hashes[sku] = sot_product["product_hash"]

    desired = {
        "schema": "skyyrose.sot.opening-product-media.v2",
        "purpose": (
            "Fail-closed V2 on-model intake for all current products. Registry presence is not approval; "
            "only exact hash-bound, explicitly reviewed on-model sources may receive the approved status."
        ),
        "source_of_truth": "data/product-sot.json",
        "product_sot_sha256": _sha256_bytes(sot_bytes),
        "product_hashes": product_hashes,
        "delivery": current.get("delivery", {}),
        "products": products,
    }
    rendered = json.dumps(desired, indent=2, ensure_ascii=False) + "\n"
    current_text = args.registry.read_text(encoding="utf-8") if args.registry.is_file() else ""
    if args.check:
        if rendered != current_text:
            print("BLOCKED on-model intake registry is not synchronized")
            return 1
        print(f"PASS on-model intake registry synchronized for {len(products)} products")
        return 0

    args.registry.parent.mkdir(parents=True, exist_ok=True)
    args.registry.write_text(rendered, encoding="utf-8")
    print(f"wrote {args.registry} ({len(products)} products; no approvals invented)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
