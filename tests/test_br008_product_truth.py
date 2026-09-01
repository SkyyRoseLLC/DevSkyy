"""Regression guards for the founder-approved BR-008 jersey construction."""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from pathlib import Path

from skyyrose.core.product_asset_contract import load_product_asset_contract


ROOT = Path(__file__).resolve().parents[1]
CATALOG = ROOT / "wordpress-theme/skyyrose-flagship/data/skyyrose-catalog.csv"
DOSSIER = ROOT / "wordpress-theme/skyyrose-flagship/data/dossiers/black-is-beautiful-jersey-series-1-sf-inspired-football.md"
LOGO_REGISTRY = ROOT / "wordpress-theme/skyyrose-flagship/data/logo-registry.json"
CORRECTIONS = ROOT / "wordpress-theme/skyyrose-flagship/data/render-corrections.json"
JERSEY_TRUTH = ROOT / ".fashion-theme/jersey-product-truth.json"
STAGE_ONE = ROOT / "renders/prompts-preview/black-is-beautiful-jersey-series-1-sf-inspired-football/stage1-base.txt"
STAGE_THREE = ROOT / "renders/prompts-preview/black-is-beautiful-jersey-series-1-sf-inspired-football/stage3-decoration.txt"


def _catalog_row(sku: str) -> dict[str, str]:
    with CATALOG.open(newline="", encoding="utf-8") as source:
        return next(row for row in csv.DictReader(source) if row["sku"] == sku)


def test_br008_canonical_contract_has_founder_80_on_both_sides_and_exact_trims() -> None:
    row = _catalog_row("br-008")
    dossier = DOSSIER.read_text(encoding="utf-8")
    truth = json.loads(JERSEY_TRUTH.read_text(encoding="utf-8"))["products"]["br-008"]
    registry = json.loads(LOGO_REGISTRY.read_text(encoding="utf-8"))["sku_logos"]["br-008"]

    assert "80 on BOTH sides" in row["branding_spec"]
    assert "rose-filled '8' plus plain-white '0'" in row["branding_spec"]
    assert "plain-white '8' plus rose-filled '0'" in row["branding_spec"]
    assert 'both front and back must read "80"' in dossier
    assert "3in wide × 4in long" in dossier
    assert "black twill with a white border" in dossier
    assert "each with black border edging" in dossier
    assert truth["required_patch"]["width_inches"] == 3
    assert truth["required_patch"]["height_inches"] == 4
    assert registry["front_number"] == "80"
    assert registry["back_number"] == "80"
    assert "jersey_number" not in registry


def test_br008_render_prompts_require_founder_80_construction() -> None:
    stage_one = STAGE_ONE.read_text(encoding="utf-8")
    stage_three = STAGE_THREE.read_text(encoding="utf-8")
    corrections = json.loads(CORRECTIONS.read_text(encoding="utf-8"))["corrections"]["br-008"]

    assert "each individually edged with black border stripes" in stage_one
    assert "Founder front SOT is **80**" in stage_three
    assert "Founder back SOT is **80**" in stage_three
    assert "black twill with a white border" in stage_three
    assert any("80 on BOTH sides" in correction for correction in corrections)


def test_br008_production_asset_contract_is_dossier_first_and_hash_pinned() -> None:
    contract = load_product_asset_contract("br-008")
    prompt = contract.prompt_text()

    # Customer catalog copy may describe the product, but it is not present as
    # a competing render input. The source contract is dossier + amendment.
    assert 'Founder front SOT is **80**' in prompt
    assert 'Founder back SOT is **80**' in prompt
    assert "newer than any conflicting cached analysis or catalog copy" in prompt
    assert "wearer-left front lower hem" in prompt
    assert "3 inches wide by 4 inches long" in prompt
    assert {asset.role for asset in contract.assets.assets} >= {"dossier", "garment"}


def test_photo_audit_ignores_negative_garment_disambiguators() -> None:
    script = ROOT / "scripts/audit_source_photos.py"
    spec = importlib.util.spec_from_file_location("audit_source_photos", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    # Dataclasses resolves postponed annotations through sys.modules.
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    finally:
        sys.modules.pop(spec.name, None)

    lock = "Authentic-style football jersey. NOT a hoodie. NOT a t-shirt."
    assert module._infer_garment_type(lock, "SF Inspired Football") == "jersey"
    assert module._infer_garment_type(
        "Black pullover hoodie — distinct from the matching Crewneck/Joggers set.",
        "BLACK Rose Hoodie",
    ) == "hoodie"
    assert module._infer_garment_type(
        "Black jogger sweatpants — lower body only.", "BLACK Rose Joggers"
    ) == "joggers"
