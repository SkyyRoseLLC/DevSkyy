"""Regression coverage for the generated V2 product-presentation adapter."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "wordpress-theme/skyyrose-flagship/data/skyyrose-catalog.csv"
REGISTRY = ROOT / "wordpress-theme/skyyrose-flagship-2/data/product-presentation-registry.json"
CERTIFICATION = ROOT / "tools/v2-source-certification"


def _catalog() -> dict[str, dict[str, str]]:
    with CATALOG.open(newline="", encoding="utf-8") as source:
        return {row["sku"].lower(): row for row in csv.DictReader(source)}


def _registry() -> dict:
    return json.loads(REGISTRY.read_text(encoding="utf-8"))


def test_registry_covers_the_exact_canonical_catalog_skus() -> None:
    assert set(_registry()["products"]) == set(_catalog())


def test_registry_preorder_state_is_derived_from_the_catalog() -> None:
    catalog = _catalog()
    for sku, presentation in _registry()["products"].items():
        assert presentation["is_preorder"] is (
            catalog[sku]["is_preorder"].strip().lower() in {"1", "true", "yes"}
        )


def test_jersey_membership_and_routes_have_one_registry_authority() -> None:
    registry = _registry()
    jerseys = registry["supplements"]["jersey_series_skus"]
    assert sorted(jerseys) == [
        "br-003",
        "br-008",
        "br-009",
        "br-010",
        "br-011",
        "br-012",
        "br-014",
        "br-015",
    ]
    # Membership is fixed; presentation order and regions come from the certified
    # upstream SOT, not SKU sorting or the retired film-chapter adapter.
    source_bytes = (CERTIFICATION / "inputs/product-sot.json").read_bytes()
    contract = json.loads((CERTIFICATION / "build-inputs.json").read_text(encoding="utf-8"))
    source_hash = hashlib.sha256(source_bytes).hexdigest()
    assert source_hash == contract["product_sot_sha256"]
    assert registry["product_sot_sha256"] == source_hash
    assert registry["schema_version"] == "2.0.0"
    source_products = json.loads(source_bytes)["products"]
    source_jerseys = {
        sku: product["merchandising"]
        for sku, product in source_products.items()
        if product.get("merchandising", {}).get("series_slug") == "jersey-series"
    }
    assert set(jerseys) == set(source_jerseys)
    assert jerseys == sorted(source_jerseys, key=lambda sku: source_jerseys[sku]["series_order"])
    for sku in jerseys:
        record = registry["products"][sku]
        assert record["collection"] == "black-rose"
        assert record["presentation"] == "jersey-series"
        assert record["route"] == "/jersey-series/"
        assert record["series_region"] == source_jerseys[sku]["series_region"]
        assert record["series_order"] == source_jerseys[sku]["series_order"]
        assert "jersey_chapter" not in record
        assert "film_start" not in record
