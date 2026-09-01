"""Regression contract for the founder-approved BR-009 football jersey sources.

The separate front/back files are product truth. This test deliberately checks
the source map, hash manifest, correction text, and deterministic patch record
together so a future catalog, render, or asset change cannot quietly restore
the older incorrect Oakland reference board.
"""

from __future__ import annotations

import json

from scripts.nano_banana.source_map import get_source_map as compatibility_source_map
from scripts.oai_render.references import get_source_map as canonical_source_map
from skyyrose.core.asset_manifest import AssetManifest
from skyyrose.core.dossier_loader import get_product_render_contract
from tests.sparse_guard import requires_tree


@requires_tree("assets/products")
def test_br009_founder_front_back_and_patch_contract_are_bound() -> None:
    expected_front = "assets/products/references/br-009-founder-white-football-front-sot.png"
    expected_back = "assets/products/references/br-009-founder-white-football-back-sot.png"

    for source_map in (canonical_source_map(), compatibility_source_map()):
        assert source_map["br-009"]["front"].as_posix().endswith(expected_front)
        assert source_map["br-009"]["back"].as_posix().endswith(expected_back)

    manifest = AssetManifest.load()
    assets = manifest.skus["br-009"].assets
    assert {asset.path for asset in assets if asset.role == "garment"} == {expected_front}
    assert {asset.path for asset in assets if asset.role == "garment-back"} == {expected_back}
    assert not manifest.verify(["br-009"])

    render = get_product_render_contract("br-009")
    correction = " ".join(render.founder_corrections)
    assert "FRONT 32: only the 3" in correction
    assert "BACK 32: only the 2" in correction
    assert "3 inches wide by 4 inches high" in correction

    contract_path = "assets/products/references/br-009-founder-white-football-patch-contract.json"
    contract = json.loads(open(contract_path, encoding="utf-8").read())
    assert contract["generator"]["route"] == "deterministic_composite"
    assert contract["references"][0]["role"] == "physical_product_authority"
    assert contract["references"][0]["view"] == "front"
    assert contract["references"][0]["path"] == expected_front
    assert contract["references"][1]["view"] == "front"
    assert contract["edit_region"]["physical_measurement"] == {
        "width_inches": 3,
        "height_inches": 4,
    }
    assert contract["edit_region"]["placement"] == "wearer-left front lower hem"
    assert contract["verification"]["max_outside_changed_pixels"] == 0
