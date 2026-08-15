#!/usr/bin/env python3
"""Generate the V2 presentation registry from the canonical catalog CSV.

This is a build-time adapter. WooCommerce remains responsible for product IDs,
prices, stock, variations, and visibility; SOT media remains responsible for
approved product visuals. The only supplemental classification is the explicit
Jersey Series SKU set, kept here once so it cannot drift across PHP templates.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CATALOG = ROOT / "wordpress-theme/skyyrose-flagship/data/skyyrose-catalog.csv"
OUTPUT = Path(__file__).resolve().parents[1] / "data/product-presentation-registry.json"
JERSEY_CHAPTERS = {
    "br-003": ("00 / Baseball Classic (Black)", 0),
    "br-008": ("01 / SF Inspired", 3.2),
    "br-009": ("02 / Last Oakland", 6.4),
    "br-010": ("03 / The Bay", 9.6),
    "br-011": ("04 / The Rose", 12.8),
    "br-012": ("05 / Baseball Classic (Last Oakland)", 6.4),
    "br-014": ("00 / Baseball Classic (Giants)", 3.2),
    "br-015": ("00 / Baseball Classic (White)", 0),
}
ALLOWED_COLLECTIONS = {"black-rose", "kids-capsule", "love-hurts", "signature"}


def build_registry() -> dict[str, object]:
    products: dict[str, dict[str, object]] = {}
    with CATALOG.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            sku = row["sku"].strip().lower()
            collection = row["collection"].strip()
            if not sku or sku in products:
                raise ValueError(f"Catalog contains an empty or duplicate SKU: {sku!r}")
            if collection not in ALLOWED_COLLECTIONS:
                raise ValueError(f"Unknown collection for {sku}: {collection!r}")
            presentation = "jersey-series" if sku in JERSEY_CHAPTERS else collection
            record: dict[str, object] = {
                "collection": collection,
                "presentation": presentation,
                # Jersey Series is a dedicated Black Rose release chapter, not
                # a fifth collection route. Keep its visual presentation
                # isolated while routing discovery through the parent world.
                "route": (
                    "/collections/black-rose/#jersey-series"
                    if presentation == "jersey-series"
                    else f"/collections/{presentation}/"
                ),
                "is_preorder": row["is_preorder"].strip().lower() in {"1", "true", "yes"},
            }
            if sku in JERSEY_CHAPTERS:
                record["jersey_chapter"], record["film_start"] = JERSEY_CHAPTERS[sku]
            products[sku] = record
    missing_jerseys = sorted(set(JERSEY_CHAPTERS) - set(products))
    if missing_jerseys:
        raise ValueError(f"Jersey supplement references unknown SKUs: {missing_jerseys}")
    return {
        "schema_version": "1.0.0",
        "kind": "skyyrose-v2-product-presentation-registry",
        "generated_from": "wordpress-theme/skyyrose-flagship/data/skyyrose-catalog.csv",
        "supplements": {"jersey_series_skus": sorted(JERSEY_CHAPTERS)},
        "products": dict(sorted(products.items())),
    }


def main() -> int:
    artifact = build_registry()
    rendered = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
    if "--check" in sys.argv:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            print(
                "Product presentation registry is stale. Run this script without --check.",
                file=sys.stderr,
            )
            return 1
        print(f"Product presentation registry is current ({len(artifact['products'])} SKUs).")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(artifact['products'])} SKUs).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
