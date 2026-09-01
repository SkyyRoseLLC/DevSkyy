"""Fail-closed product-asset contract for every paid render or derivative.

The CSV identifies a sellable SKU; it cannot establish physical garment truth.
This module binds one required dossier, founder amendments, and the exact
content-hashed input bundle before provider work begins.
"""

from __future__ import annotations

from dataclasses import dataclass

from skyyrose.core.asset_manifest import AssetManifest, DriftFinding, SkuAssets, to_repo_relative
from skyyrose.core.catalog_loader import CATALOG_CSV
from skyyrose.core.dossier_loader import (
    DOSSIERS_DIR,
    RENDER_CORRECTIONS_PATH,
    ProductRenderContract,
    get_product_render_contract,
)
from skyyrose.core.hashing import sha256_of_file


class ProductAssetContractError(RuntimeError):
    """Raised when a SKU cannot prove current source provenance."""


@dataclass(frozen=True)
class ProductAssetContract:
    render: ProductRenderContract
    assets: SkuAssets

    @property
    def sku(self) -> str:
        return self.render.dossier.sku

    def prompt_text(self) -> str:
        return self.render.prompt_text()


def _required_roles_present(assets: SkuAssets) -> bool:
    hashed_roles = {asset.role for asset in assets.assets if asset.sha256}
    return {"dossier", "founder-corrections", "garment"} <= hashed_roles


def _assert_manifest_binds_current_contract(
    manifest: AssetManifest, render: ProductRenderContract, assets: SkuAssets
) -> None:
    if manifest.catalog_sha != sha256_of_file(CATALOG_CSV):
        raise ProductAssetContractError(
            "catalog changed after asset-manifest generation; run scripts/build_asset_manifest.py"
        )
    expected = {
        "dossier": to_repo_relative(DOSSIERS_DIR / f"{render.dossier.slug}.md"),
        "founder-corrections": to_repo_relative(RENDER_CORRECTIONS_PATH),
    }
    for role, path in expected.items():
        matches = [asset for asset in assets.assets if asset.role == role and asset.path == path]
        if len(matches) != 1 or not matches[0].sha256:
            raise ProductAssetContractError(
                f"{render.dossier.sku}: manifest does not pin current {role} asset {path}"
            )


def load_product_asset_contract(sku: str, *, verify: bool = True) -> ProductAssetContract:
    """Return current dossier-first facts and hash-pinned source assets.

    Production callers retain ``verify=True``. A missing or drifted asset is a
    hard stop, never a chance to render from a stale filename or catalog field.
    """
    render = get_product_render_contract(sku)
    manifest = AssetManifest.load()
    assets = manifest.skus.get(sku)
    if assets is None:
        raise ProductAssetContractError(
            f"{sku}: no content-hashed asset bundle; run scripts/build_asset_manifest.py"
        )
    if not _required_roles_present(assets):
        raise ProductAssetContractError(
            f"{sku}: asset bundle lacks a hash-pinned dossier, founder corrections, or garment source"
        )
    _assert_manifest_binds_current_contract(manifest, render, assets)
    if verify:
        drift: list[DriftFinding] = manifest.verify([sku])
        if drift:
            detail = "; ".join(f"{item.role}:{item.kind}:{item.path}" for item in drift)
            raise ProductAssetContractError(f"{sku}: source provenance drift — {detail}")
    return ProductAssetContract(render=render, assets=assets)


__all__ = ["ProductAssetContract", "ProductAssetContractError", "load_product_asset_contract"]
