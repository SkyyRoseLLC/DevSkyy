"""Fail-closed guardrails for the shared product-creation asset contract."""

from __future__ import annotations

import pytest

from skyyrose.core.asset_manifest import AssetManifest, AssetRecord, SkuAssets
from skyyrose.core.product_asset_contract import (
    ProductAssetContractError,
    load_product_asset_contract,
)


def test_real_contract_has_hash_pinned_dossier_and_garment() -> None:
    contract = load_product_asset_contract("br-001")
    roles = {asset.role for asset in contract.assets.assets if asset.sha256}
    assert {"dossier", "garment"} <= roles
    assert "PRODUCT:" in contract.prompt_text()
    assert "BRANDING — exactly what IS on this product:" in contract.prompt_text()


def test_missing_manifest_bundle_blocks_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    from skyyrose.core import product_asset_contract

    empty = AssetManifest()
    monkeypatch.setattr(product_asset_contract.AssetManifest, "load", lambda: empty)
    with pytest.raises(ProductAssetContractError, match="no content-hashed asset bundle"):
        load_product_asset_contract("br-001")


def test_bundle_without_a_hashed_garment_blocks_creation(monkeypatch: pytest.MonkeyPatch) -> None:
    from skyyrose.core import product_asset_contract

    manifest = AssetManifest(
        skus={
            "br-001": SkuAssets(
                sku="br-001",
                name="test",
                collection="test",
                garment_type="crewneck",
                assets=[
                    AssetRecord(
                        role="dossier",
                        path="wordpress-theme/skyyrose-flagship/data/dossiers/black-rose-crewneck.md",
                        sha256="sha256:placeholder",
                    )
                ],
            )
        }
    )
    monkeypatch.setattr(product_asset_contract.AssetManifest, "load", lambda: manifest)
    with pytest.raises(ProductAssetContractError, match="lacks a hash-pinned dossier"):
        load_product_asset_contract("br-001")


def test_catalog_hash_mismatch_blocks_even_when_old_assets_still_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A catalog/dossier re-route cannot reuse an old, otherwise valid manifest."""
    from skyyrose.core import product_asset_contract
    from skyyrose.core.asset_manifest import AssetManifest

    committed = AssetManifest.load()
    committed.catalog_sha = "sha256:stale-catalog"
    monkeypatch.setattr(product_asset_contract.AssetManifest, "load", lambda: committed)
    with pytest.raises(ProductAssetContractError, match="catalog changed after asset-manifest generation"):
        load_product_asset_contract("br-001")


def test_unreadable_founder_corrections_blocks_the_render_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from skyyrose.core import dossier_loader

    monkeypatch.setattr(dossier_loader, "RENDER_CORRECTIONS_PATH", tmp_path / "missing.json")
    with pytest.raises(dossier_loader.RenderCorrectionsError, match="Cannot read required founder corrections"):
        dossier_loader.get_product_render_contract("br-001")
