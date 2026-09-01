#!/usr/bin/env python3
"""Verify every selected SKU can enter the product-creation pipeline safely.

This is the pre-provider gate for render, on-model, and derived-product work.
It proves that each SKU has a mandatory dossier, its binding founder
corrections, and a content-hash-pinned garment source bundle. It makes no API
calls and writes no assets.

Usage:
    python scripts/verify_product_asset_contract.py
    python scripts/verify_product_asset_contract.py --sku br-008 --sku br-004
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyyrose.core.catalog_loader import read_catalog_rows  # noqa: E402
from skyyrose.core.product_asset_contract import (  # noqa: E402
    ProductAssetContractError,
    load_product_asset_contract,
)


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sku", action="append", dest="skus", help="SKU to verify; repeatable")
    args = parser.parse_args(argv)

    catalog_skus = [(row.get("sku") or "").strip() for row in read_catalog_rows()]
    catalog_skus = [sku for sku in catalog_skus if sku]
    selected = args.skus or catalog_skus
    unknown = sorted(set(selected) - set(catalog_skus))
    if unknown:
        parser.error("Unknown SKU(s): " + ", ".join(unknown))

    failures: list[tuple[str, str]] = []
    for sku in selected:
        try:
            contract = load_product_asset_contract(sku)
        except (ProductAssetContractError, FileNotFoundError, KeyError, ValueError) as exc:
            failures.append((sku, str(exc)))
            print(f"BLOCKED {sku}: {exc}")
            continue
        roles = sorted({asset.role for asset in contract.assets.assets if asset.sha256})
        print(f"PASS {sku}: dossier + {len(contract.render.founder_corrections)} amendment(s) + {', '.join(roles)}")

    print(f"\nsummary: {len(selected) - len(failures)}/{len(selected)} product asset contracts verified")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
