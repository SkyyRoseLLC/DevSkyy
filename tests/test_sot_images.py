"""Tests for the canonical SOT product-imagery resolver.

The resolver is THE single authority for "what image represents this SKU" across
every surface. These tests pin the approval-first contract: approved on-model media
wins, while rejected or pending candidates fall back to canonical packshots.
"""

from __future__ import annotations

import json

import pytest

from skyyrose.core import sot_images


@pytest.fixture(autouse=True)
def _clear_sot_cache():
    """Drop the lru_cached index around every test so monkeypatched/synthetic
    indices never leak into a later test (the cache is process-global)."""
    sot_images.refresh()
    yield
    sot_images.refresh()


def test_rejected_front_falls_back_to_packshot():
    path = sot_images.resolve_image("br-004", "front")
    assert path is not None
    assert path == sot_images.resolve_image("br-004", "packshot")
    assert "front_model_image" not in sot_images._index()["br-004"]["images"]


def test_approved_front_prefers_on_model_render():
    path = sot_images.resolve_image("br-006", "front")
    assert path == sot_images._index()["br-006"]["images"]["front_model_image"]["path"]
    assert path != sot_images.resolve_image("br-006", "packshot")


def test_resolve_back_uses_canonical_back_when_model_is_unapproved():
    path = sot_images.resolve_image("br-002", "back")
    assert path is not None
    assert "back" in path


def test_packshot_role_returns_flat_image():
    path = sot_images.resolve_image("br-004", "packshot")
    assert path is not None
    # The flat packshot is the catalog `image` column (jpeg/png studio shot).
    assert path.endswith((".jpeg", ".jpg", ".png"))


def test_back_packshot_role_returns_exact_back_image():
    # br-002 has a real back image in the SOT — back_packshot must return
    # exactly that value, with no render-fallback (mirrors the packshot role's
    # single-key contract, but for the back face).
    path = sot_images.resolve_image("br-002", "back_packshot")
    assert path is not None
    assert path == sot_images._index()["br-002"]["images"]["back_image"]["path"]


def test_unknown_sku_returns_none_not_a_guess():
    assert sot_images.resolve_image("zz-999", "front") is None
    assert sot_images.resolve_image("", "front") is None


def test_resolve_returns_theme_relative_assets_path():
    path = sot_images.resolve_image("br-010", "front")
    assert path is not None
    assert path.startswith("assets/images/products/")


def test_all_four_collections_indexed():
    # Every collection's SOT products are reachable through one index.
    skus = sot_images.all_skus()
    assert any(s.startswith("br-") for s in skus)
    assert any(s.startswith("lh-") for s in skus)
    assert any(s.startswith("sg-") for s in skus)
    assert any(s.startswith("kids-") for s in skus)


def test_has_render_true_for_real_sku_false_for_unknown():
    assert sot_images.has_render("br-006") is True
    assert sot_images.has_render("br-004") is False
    assert sot_images.has_render("zz-999") is False


def test_build_manifest_shape():
    manifest = sot_images.build_manifest()
    assert "br-004" in manifest
    entry = manifest["br-004"]
    assert set(entry).issubset({"front", "back", "packshot", "back_packshot"})
    assert "front" in entry  # every published SKU has at least a front
    # Manifest paths are the same theme-relative contract as resolve_image.
    assert entry["front"] == sot_images.resolve_image("br-004", "front")


def test_manifest_is_json_serializable():
    # The dashboard consumes this as data/sot-images.json — must round-trip.
    manifest = sot_images.build_manifest()
    assert json.loads(json.dumps(manifest)) == manifest


def test_back_falls_back_to_back_image_when_no_back_model(monkeypatch):
    # In live data every SKU with back_image also has back_model_image, so the
    # second fallback key is otherwise unexercised. Synthesize the gap.
    synthetic = {
        "xx-001": {"images": {"back_image": {"path": "assets/images/products/xx-001-back.webp"}}}
    }
    monkeypatch.setattr(sot_images, "_index", lambda: synthetic)
    assert sot_images.resolve_image("xx-001", "back") == "assets/images/products/xx-001-back.webp"


def test_resolve_rejects_path_traversal(monkeypatch):
    # Defense-in-depth: a poisoned SOT path that escapes the assets tree must raise,
    # never silently return a climbing path a consumer might open()/serve.
    synthetic = {"xx-002": {"images": {"front_model_image": {"path": "../../etc/passwd"}}}}
    monkeypatch.setattr(sot_images, "_index", lambda: synthetic)
    with pytest.raises(ValueError, match="escapes the assets tree"):
        sot_images.resolve_image("xx-002", "front")


def test_approval_manifest_rejects_per_sku_hash_drift(tmp_path, monkeypatch):
    product_sot_bytes = sot_images.product_sot.MANIFEST_PATH.read_bytes()
    approval = json.loads(sot_images.APPROVAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    approval["product_hashes"]["br-001"] = "0" * 64
    manifest_path = tmp_path / "opening-product-media.json"
    manifest_path.write_text(json.dumps(approval), encoding="utf-8")
    monkeypatch.setattr(sot_images, "APPROVAL_MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError, match="stale for br-001"):
        sot_images._approval_manifest(product_sot_bytes)


def test_approval_manifest_rejects_incomplete_sku_binding(tmp_path, monkeypatch):
    product_sot_bytes = sot_images.product_sot.MANIFEST_PATH.read_bytes()
    approval = json.loads(sot_images.APPROVAL_MANIFEST_PATH.read_text(encoding="utf-8"))
    del approval["product_hashes"]["br-001"]
    manifest_path = tmp_path / "opening-product-media.json"
    manifest_path.write_text(json.dumps(approval), encoding="utf-8")
    monkeypatch.setattr(sot_images, "APPROVAL_MANIFEST_PATH", manifest_path)

    with pytest.raises(ValueError, match="hash SKU set differs"):
        sot_images._approval_manifest(product_sot_bytes)
