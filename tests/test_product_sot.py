from __future__ import annotations

import json

from skyyrose.core import product_sot


def test_product_sot_covers_the_root_catalog() -> None:
    manifest = product_sot.build_manifest()
    assert len(manifest["products"]) == 33
    assert set(manifest["products"]) == {
        row["sku"] for row in product_sot.read_catalog_rows(product_sot.CATALOG_PATH)
    }
    assert manifest["sources"]["catalog"] == "data/skyyrose-catalog.csv"
    assert manifest["sources"]["logo_registry"] == "data/logo-registry.json"


def test_every_product_is_hash_bound_to_root_owned_sources() -> None:
    for sku, product in product_sot.build_manifest()["products"].items():
        assert product["product_hash"], sku
        assert product["source"]["dossier"].startswith("data/dossiers/"), sku
        assert product["source"]["dossier_sha256"], sku
        assert product["garment"]["branding_regions"], sku
        assert product["garment"]["negative_constraints"], sku
        assert set(product["consumer_contract"]["required_for"]) == set(
            product_sot.CREATIVE_CONSUMERS
        )


def test_br005_material_locks_are_generated_and_hash_bound() -> None:
    product = product_sot.build_manifest()["products"]["br-005"]
    assert product["garment"]["material_lock_version"] == "v1"
    regions = {item["region"]: item for item in product["garment"]["branding_regions"]}
    assert "silicone" in regions["front-right-chest"]["material_lock"]["material_family"]
    assert "sleeve" in regions[
        "wearer's-left side body / viewer-right torso"
    ]["material_lock"]["reject_cues"]
    assert "zero raised edge" in regions[
        "hood-inside / inner-hood-lining"
    ]["material_lock"]["surface_response"]


def test_root_physical_references_are_not_downgraded_to_generic_or_techflat_evidence() -> None:
    manifest = product_sot.build_manifest()
    for sku in ("br-005", "br-007"):
        product = manifest["products"][sku]
        assert product["verification"]["proof_level"] == "physical_product_photo"

    br007 = manifest["products"]["br-007"]
    physical_views = {
        item["path"]: item["kind"]
        for item in br007["references"]
        if item["path"].startswith("assets/products/references/br-007-shorts-")
    }
    assert physical_views["assets/products/references/br-007-shorts-front-source.jpg"] == (
        "physical_product_photo"
    )
    assert physical_views["assets/products/references/br-007-shorts-left-hip-source.jpg"] == (
        "physical_product_photo"
    )
    assert physical_views["assets/products/references/br-007-shorts-back-source.jpg"] == (
        "physical_product_photo"
    )
    assert physical_views["assets/products/references/br-007-shorts-techflat.jpeg"] == (
        "approved_design_reference"
    )


def test_checked_in_manifest_is_current() -> None:
    expected = product_sot.serialize_manifest()
    assert product_sot.MANIFEST_PATH.read_text(encoding="utf-8") == expected
    assert json.loads(expected)["schema"] == product_sot.SCHEMA
