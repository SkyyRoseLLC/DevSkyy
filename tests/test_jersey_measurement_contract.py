"""Founder measurement requirements must not drift from Jersey Series inputs."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DOSSIERS = ROOT / "wordpress-theme/skyyrose-flagship/data/dossiers"
REGISTRY = ROOT / "wordpress-theme/skyyrose-flagship/data/logo-registry.json"
BR009_PATCH_CONTRACT = ROOT / "assets/products/references/br-009-founder-white-football-patch-contract.json"


def test_all_baseball_wordmarks_and_patches_follow_founder_measurements() -> None:
    slugs = (
        "black-is-beautiful-jersey-series-0-baseball-classic",
        "black-is-beautiful-jersey-series-5-last-oakland-baseball",
        "black-is-beautiful-jersey-series-0-baseball-classic-giants",
        "black-is-beautiful-jersey-series-0-baseball-classic-white",
    )
    for slug in slugs:
        dossier = (DOSSIERS / f"{slug}.md").read_text(encoding="utf-8")
        assert "75–80% of the usable front-panel" in dossier
        assert "3in wide × 4in high" in dossier

    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))["sku_logos"]
    for sku in ("br-003", "br-012", "br-014", "br-015"):
        assert registry[sku]["front_text_coverage"] == "75–80% of usable front-panel width"
        patch = next(item for item in registry[sku]["placements"] if item["position"] == "front_patch")
        assert patch["size_inches"] == "3 × 4"


def test_every_jersey_series_sport_patch_has_the_shared_3x4_contract() -> None:
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))["sku_logos"]
    for sku in ("br-008", "br-009", "br-010", "br-011", "br-012", "br-014", "br-015", "br-003"):
        patch = next(
            item
            for item in registry[sku]["placements"]
            if item["position"] in {"front_patch", "bottom_left_corner"}
        )
        assert patch["size_inches"] == "3 × 4"


def test_br009_patch_composite_has_an_explicit_locked_pixel_boundary() -> None:
    contract = json.loads(BR009_PATCH_CONTRACT.read_text(encoding="utf-8"))
    assert contract["generator"]["route"] == "deterministic_alpha_composite"
    assert contract["edit_region"]["bbox"] == [278, 300, 305, 338]
    assert contract["edit_region"]["physical_measurement"] == {"width_inches": 3, "height_inches": 4}
    assert contract["verification"]["max_outside_changed_pixels"] == 0
