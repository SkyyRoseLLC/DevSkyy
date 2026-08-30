"""Regression guard for the two physically distinct Black Rose hoodies."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PRODUCT_SOT = ROOT / "data/product-sot.json"
ROOT_DOSSIERS = ROOT / "data/dossiers"
THEME_DATA = ROOT / "wordpress-theme/skyyrose-flagship/data"
LAUNCH_MAP = ROOT / "scripts/launch/sku_image_map.json"
MAPPER = ROOT / "scripts/launch/map_images_to_products.py"


def _mapper_module():
    spec = importlib.util.spec_from_file_location("launch_image_mapper", MAPPER)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_product_truth_lives_at_the_repository_root_not_the_theme() -> None:
    assert (ROOT / "data/skyyrose-catalog.csv").is_file()
    assert ROOT_DOSSIERS.is_dir()
    assert (ROOT / "data/brand-logos/three-rose-cluster.md").is_file()
    assert not (THEME_DATA / "skyyrose-catalog.csv").exists()
    assert not (THEME_DATA / "dossiers").exists()
    assert not (THEME_DATA / "brand-logos").exists()


def test_br004_and_br005_have_distinct_root_bound_product_locks() -> None:
    products = json.loads(PRODUCT_SOT.read_text(encoding="utf-8"))["products"]
    regular = products["br-004"]
    signature = products["br-005"]

    assert regular["source"]["dossier"] == "data/dossiers/black-rose-hoodie.md"
    assert signature["source"]["dossier"] == "data/dossiers/black-rose-hoodie-signature-edition.md"
    assert regular["source"]["dossier_sha256"] == _sha256(ROOT_DOSSIERS / "black-rose-hoodie.md")
    assert signature["source"]["dossier_sha256"] == _sha256(
        ROOT_DOSSIERS / "black-rose-hoodie-signature-edition.md"
    )

    regular_lock = regular["garment"]
    assert "longer regular" in regular_lock["type_lock"].lower()
    assert "centered" in regular_lock["branding_summary"].lower()
    assert "side-body" in " ".join(regular_lock["negative_constraints"]).lower()
    assert "silicone" in " ".join(regular_lock["negative_constraints"]).lower()
    assert "sublimated" in " ".join(regular_lock["negative_constraints"]).lower()

    signature_lock = signature["garment"]
    assert "right chest" in signature_lock["branding_summary"].lower()
    assert "side body" in signature_lock["branding_summary"].lower()
    regions = {region["region"] for region in signature_lock["branding_regions"]}
    assert "front-right-chest" in regions
    assert "wearer's-left side body / viewer-right torso" in regions
    assert "hood-inside / inner-hood-lining" in regions
    assert signature_lock["material_lock_version"] == "v1"

    region_records = {
        region["region"]: region for region in signature_lock["branding_regions"]
    }
    chest_material = region_records["front-right-chest"]["material_lock"]
    assert "silicone" in chest_material["material_family"].lower()
    assert "rubber-like" in chest_material["surface_response"].lower()
    assert "thread" in chest_material["reject_cues"].lower()

    side_material = region_records[
        "wearer's-left side body / viewer-right torso"
    ]["material_lock"]
    assert "embroidery thread" in side_material["material_family"].lower()
    assert "stitch texture" in side_material["surface_response"].lower()
    assert "sleeve" in side_material["reject_cues"].lower()

    lining_material = region_records["hood-inside / inner-hood-lining"]["material_lock"]
    assert "sublimation dye" in lining_material["material_family"].lower()
    assert "zero raised edge" in lining_material["surface_response"].lower()
    assert "patch" in lining_material["reject_cues"].lower()


def test_launch_mapping_cannot_swap_the_regular_and_signature_hoodies() -> None:
    mapper = _mapper_module()
    assert mapper.MANUAL_OVERRIDES["br-004"] == "BR_WOMENS_BLACK_ROSE_HOODED_DRESS_main"
    assert "br-005" not in mapper.MANUAL_OVERRIDES
    assert "br-005" in mapper.EXACT_SOURCE_REQUIRED_SKUS
    assert "br-015" in mapper.EXACT_SOURCE_REQUIRED_SKUS

    published = json.loads(LAUNCH_MAP.read_text(encoding="utf-8"))
    assert published["br-004"]["match_key"] == "BR_WOMENS_BLACK_ROSE_HOODED_DRESS_main"
    assert "br-005" not in published
    assert "br-015" not in published
