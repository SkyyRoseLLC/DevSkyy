"""Regression guards for root-owned SkyyRose source-of-truth files.

Themes may package generated views and runtime assets, but they must never become
an editable authority for product, collection, render-review, or visual identity
data.  This test deliberately checks locations rather than duplicated contents:
one physical owner is what prevents consumer drift.
"""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATA = ROOT / "data"
THEME_DATA = ROOT / "wordpress-theme" / "skyyrose-flagship" / "data"


def test_editable_product_and_collection_authorities_are_root_owned() -> None:
    expected_sources = {
        "skyyrose-catalog.csv",
        "dossiers",
        "brand-logos",
        "visual-manifest.json",
        "logo-registry.json",
        "render-keepers.json",
        "render-corrections.json",
        "brand/typography.json",
        "brand/typography.schema.json",
        "collections/identity.schema.json",
        "collections/black-rose/identity.json",
        "collections/love-hurts/identity.json",
        "collections/signature/identity.json",
        "collections/kids-capsule/identity.json",
        "collections/black-rose/copy.md",
        "collections/love-hurts/copy.md",
        "collections/signature/copy.md",
        "collections/kids-capsule/copy.md",
    }
    for relative_path in expected_sources:
        assert (SOURCE_DATA / relative_path).exists(), relative_path
        assert not (THEME_DATA / relative_path).exists(), relative_path


def test_product_sot_names_root_owned_catalog_and_registry() -> None:
    product_sot = json.loads((SOURCE_DATA / "product-sot.json").read_text(encoding="utf-8"))
    assert product_sot["sources"]["catalog"] == "data/skyyrose-catalog.csv"
    assert product_sot["sources"]["logo_registry"] == "data/logo-registry.json"


def test_theme_collection_directory_contains_generated_views_not_source_identity() -> None:
    for slug in ("black-rose", "love-hurts", "signature", "kids-capsule"):
        generated_dir = THEME_DATA / "collections" / slug
        assert (generated_dir / "sot.json").is_file()
        assert not (generated_dir / "identity.json").exists()
        assert not (generated_dir / "copy.md").exists()

