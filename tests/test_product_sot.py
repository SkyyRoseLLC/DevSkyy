from __future__ import annotations

import json

from skyyrose.core import product_sot


def test_product_sot_covers_all_catalog_products() -> None:
    manifest = product_sot.build_manifest()
    assert len(manifest["products"]) == 33
    assert set(manifest["products"]) == {
        row["sku"] for row in product_sot.read_catalog_rows(product_sot.CATALOG_PATH)
    }


def test_every_product_has_hash_bound_evidence_and_consumers() -> None:
    for sku, product in product_sot.build_manifest()["products"].items():
        assert product["product_hash"], sku
        assert product["source"]["dossier_sha256"], sku
        assert product["garment"]["branding_regions"], sku
        assert product["garment"]["negative_constraints"], sku
        assert isinstance(product["logo_placements"], list), sku
        assert product["media"], sku
        assert set(product["consumer_contract"]["required_for"]) == set(
            product_sot.CREATIVE_CONSUMERS
        )


def test_all_jersey_patches_are_exactly_three_by_four_inches() -> None:
    jersey_skus = {"br-003", "br-008", "br-009", "br-010", "br-011", "br-012", "br-014", "br-015"}
    products = product_sot.build_manifest()["products"]
    for sku in jersey_skus:
        patches = [
            item
            for item in products[sku]["logo_placements"]
            if "patch" in item.get("position", "") or "card" in item.get("logo_id", "")
        ]
        assert patches, sku
        assert all(item.get("width_inches") == 3 for item in patches), sku
        assert all(item.get("height_inches") == 4 for item in patches), sku


def test_jersey_series_membership_and_order_are_explicit() -> None:
    products = product_sot.build_manifest()["products"]
    jerseys = {
        sku: product["merchandising"]
        for sku, product in products.items()
        if product["merchandising"]["series_slug"] == "jersey-series"
    }
    assert [
        sku
        for sku, _record in sorted(
            jerseys.items(), key=lambda item: item[1]["series_order"]
        )
    ] == [
        "br-003",
        "br-015",
        "br-009",
        "br-012",
        "br-008",
        "br-014",
        "br-010",
        "br-011",
    ]
    assert {record["series_region"] for record in jerseys.values()} == {
        "oakland",
        "san-francisco",
        "san-jose",
    }
    assert all(
        product["merchandising"]
        == {"series_slug": "", "series_region": "", "series_order": 0}
        for sku, product in products.items()
        if sku not in jerseys
    )


def test_br006_founder_footage_is_hash_bound_physical_product_evidence() -> None:
    references = product_sot.build_manifest()["products"]["br-006"]["references"]
    footage = [item for item in references if item["kind"] == "physical_product_video"]
    assert len(footage) == 1
    assert footage[0]["role"] == "garment_reference_3"
    assert footage[0]["path"].endswith("black-rose-sherpa-founder-footage-2024.mp4")
    assert footage[0]["sha256"]


def test_checked_in_manifest_is_current() -> None:
    expected = product_sot.serialize_manifest()
    assert product_sot.MANIFEST_PATH.read_text(encoding="utf-8") == expected
    assert json.loads(expected)["schema"] == product_sot.SCHEMA
