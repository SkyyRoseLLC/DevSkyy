"""Regression gates for founder-declared Jersey Series source pixels."""

from __future__ import annotations

import json
from pathlib import Path

from scripts.nano_banana.source_map import get_source_map as legacy_source_map
from scripts.oai_render.references import build_references, get_source_map
from skyyrose.core.hashing import sha256_of_file


ROOT = Path(__file__).resolve().parent.parent
RECEIPT = ROOT / "assets/products/references/founder-jersey-sot.json"


def _receipt_by_sku() -> dict[str, dict[str, object]]:
    payload = json.loads(RECEIPT.read_text(encoding="utf-8"))
    return {entry["sku"]: entry for entry in payload["assets"]}


def _view_source(entry: dict[str, object], view: str) -> dict[str, str]:
    split_views = entry.get("split_views", {})
    if isinstance(split_views, dict) and view in split_views:
        return split_views[view]
    return entry["source"]


def test_founder_jersey_sources_are_content_hashed_and_present() -> None:
    entries = _receipt_by_sku()
    assert set(entries) == {"br-008", "br-009", "br-010", "br-011", "br-012", "br-014"}

    for sku, entry in entries.items():
        source_info = entry["source"]
        source = ROOT / str(source_info["path"])
        assert source.is_file(), f"{sku}: founder SOT source is missing"
        assert sha256_of_file(source) == source_info["sha256"], f"{sku}: founder SOT hash drift"
        for view in entry["views"]:
            view_info = _view_source(entry, view)
            view_path = ROOT / view_info["path"]
            assert view_path.is_file(), f"{sku}/{view}: SOT view is missing"
            assert sha256_of_file(view_path) == view_info["sha256"], f"{sku}/{view}: SOT hash drift"


def test_both_render_maps_resolve_the_same_founder_sources() -> None:
    entries = _receipt_by_sku()
    primary = get_source_map()
    legacy = legacy_source_map()

    for sku, entry in entries.items():
        front = ROOT / _view_source(entry, "front")["path"]
        assert primary[sku]["front"] == front
        assert legacy[sku]["front"] == front
        if entry["views"] == ["front"]:
            assert primary[sku]["back"] is None
            assert legacy[sku]["back"] is None
        else:
            back = ROOT / _view_source(entry, "back")["path"]
            assert primary[sku]["back"] == back
            assert legacy[sku]["back"] == back


def test_founder_sources_exclude_superseded_sku_prefix_matches() -> None:
    """A founder board must be the only garment source in a paid render set."""
    entries = _receipt_by_sku()
    for sku, entry in entries.items():
        refs = build_references(sku, "black-rose", view="front")
        garment_paths = {ref.path for ref in refs if ref.kind.startswith("garment")}
        expected = {ROOT / _view_source(entry, "front")["path"]}
        if "back" in entry["views"]:
            expected.add(ROOT / _view_source(entry, "back")["path"])
        assert garment_paths == expected
