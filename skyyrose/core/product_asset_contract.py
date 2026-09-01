"""Fail-closed production contract for every SKU render or product creation.

The commerce CSV is deliberately *not* a visual specification.  This module
is the one entry point a renderer may use when it needs product truth:

``commerce identity -> mandatory authored dossier -> founder corrections ->
content-hashed source bundle``.

It prevents three common forms of product drift before a provider is called:

* a SKU has no authored dossier;
* an old catalog description or cached vision analysis competes with the
  physical specification; and
* a source image, logo, patch, or dossier was replaced after its approved
  hash was recorded in ``assets/products/manifest.json``.

The returned contract intentionally exposes commerce fields only for identity
and routing.  ``prompt_text()`` is dossier-first and must be used for physical
render instructions.
"""

from __future__ import annotations

from dataclasses import dataclass

from skyyrose.core.asset_manifest import (
    AssetManifest,
    DriftFinding,
    SkuAssets,
    to_repo_relative,
)
from skyyrose.core.catalog_loader import CATALOG_CSV
from skyyrose.core.dossier_loader import (
    DOSSIERS_DIR,
    RENDER_CORRECTIONS_PATH,
    ProductRenderContract,
    get_product_render_contract,
)
from skyyrose.core.hashing import sha256_of_file


class ProductAssetContractError(RuntimeError):
    """Raised when a SKU cannot prove its physical-product source provenance."""


@dataclass(frozen=True)
class ProductAssetContract:
    """One SKU's physical render facts plus its hash-pinned input assets."""

    render: ProductRenderContract
    assets: SkuAssets

    @property
    def sku(self) -> str:
        return self.render.dossier.sku

    @property
    def product(self) -> dict[str, str]:
        """Commerce identity/routing metadata; never use as physical design truth."""
        return self.render.product

    def prompt_text(self) -> str:
        """Return the only physical-product prompt source for this SKU."""
        return self.render.prompt_text()


def _required_roles_present(assets: SkuAssets) -> bool:
    """Require a hashed dossier and at least one hashed garment input.

    Back views and patches are product/view-specific, so they are recorded and
    verified when present but are not globally required for every SKU.  A
    product with no physical garment source or no dossier cannot enter any
    creation pipeline.
    """
    hashed_roles = {asset.role for asset in assets.assets if asset.sha256}
    return {"dossier", "garment", "founder-corrections"} <= hashed_roles


def _assert_manifest_binds_current_contract(
    manifest: AssetManifest, render: ProductRenderContract, assets: SkuAssets
) -> None:
    """Prove the manifest pins the same catalog, dossier, and corrections we loaded."""
    current_catalog_sha = sha256_of_file(CATALOG_CSV)
    if manifest.catalog_sha != current_catalog_sha:
        raise ProductAssetContractError(
            "catalog changed after asset-manifest generation; run "
            "scripts/build_asset_manifest.py before product creation"
        )
    expected_dossier = to_repo_relative(DOSSIERS_DIR / f"{render.dossier.slug}.md")
    expected_corrections = to_repo_relative(RENDER_CORRECTIONS_PATH)
    expected = {"dossier": expected_dossier, "founder-corrections": expected_corrections}
    for role, path in expected.items():
        matching = [asset for asset in assets.assets if asset.role == role and asset.path == path]
        if len(matching) != 1 or not matching[0].sha256:
            raise ProductAssetContractError(
                f"{render.dossier.sku}: manifest does not pin current {role} asset {path}"
            )


def load_product_asset_contract(sku: str, *, verify: bool = True) -> ProductAssetContract:
    """Load an SKU's dossier-first contract and optionally prove source hashes.

    This is intentionally fail-closed.  A missing manifest record, a manifest
    record without a dossier or garment, or any drift finding blocks the caller
    before it can create a derivative.  Callers doing metadata-only inspection
    may pass ``verify=False``; production renderers must retain the default.
    """
    render = get_product_render_contract(sku)  # mandatory dossier / valid SKU
    manifest = AssetManifest.load()
    assets = manifest.skus.get(sku)
    if assets is None:
        raise ProductAssetContractError(
            f"{sku}: no content-hashed asset bundle in assets/products/manifest.json. "
            "Run scripts/build_asset_manifest.py after authoring verified sources."
        )
    if not _required_roles_present(assets):
        raise ProductAssetContractError(
            f"{sku}: asset bundle lacks a hash-pinned dossier, founder corrections, or garment source; "
            "do not create a derivative from an unproven product."
        )
    _assert_manifest_binds_current_contract(manifest, render, assets)
    if verify:
        drift: list[DriftFinding] = manifest.verify([sku])
        if drift:
            detail = "; ".join(f"{d.role}:{d.kind}:{d.path}" for d in drift)
            raise ProductAssetContractError(f"{sku}: source provenance drift — {detail}")
    return ProductAssetContract(render=render, assets=assets)


__all__ = [
    "ProductAssetContract",
    "ProductAssetContractError",
    "load_product_asset_contract",
]
