from __future__ import annotations

import hashlib
import json
from pathlib import Path

from skyyrose.core.on_model_media_intake import (
    validate_generation_receipt,
    validate_media_intake,
)


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    asset = tmp_path / "wordpress-theme/skyyrose-flagship/assets/images/products/br-006-onmodel.webp"
    asset.parent.mkdir(parents=True)
    asset_bytes = b"verified on-model source"
    asset.write_bytes(asset_bytes)

    product_sot = {
        "products": {
            "br-006": {
                "product_hash": "product-hash-br-006",
                "media": {
                    "on_model_front": {
                        "path": str(asset.relative_to(tmp_path)),
                        "sha256": _sha256(asset_bytes),
                        "bytes": len(asset_bytes),
                    }
                },
            }
        }
    }
    product_sot_path = tmp_path / "data/product-sot.json"
    product_sot_path.parent.mkdir()
    product_sot_path.write_text(json.dumps(product_sot), encoding="utf-8")
    product_sot_bytes = product_sot_path.read_bytes()

    registry = {
        "product_sot_sha256": _sha256(product_sot_bytes),
        "product_hashes": {"br-006": "product-hash-br-006"},
        "reviewed_at": "2026-08-21",
        "asset_integrity": {
            "br-006": {
                "on_model_front": {
                    "source_sha256": _sha256(asset_bytes),
                    "source_bytes": len(asset_bytes),
                }
            }
        },
        "products": {
            "br-006": {
                "views": [
                    {
                        "role": "on_model_front",
                        "source": "assets/images/products/br-006-onmodel.webp",
                    }
                ]
            }
        },
    }
    registry_path = tmp_path / "opening-product-media.json"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    return product_sot_path, registry_path


def test_registered_front_is_supported_only_when_all_bindings_match(tmp_path: Path) -> None:
    product_sot_path, registry_path = _write_fixture(tmp_path)

    report = validate_media_intake(product_sot_path, registry_path, repo_root=tmp_path)

    assert report["summary"] == {"supported": 1, "blocked": 0}
    assert report["products"][0]["state"] == "SUPPORTED"
    assert report["products"][0]["warnings"] == ["LEGACY_APPROVAL_REFERENCE_MISSING"]


def test_rejects_a_view_that_does_not_resolve_to_the_sku_sot_source(tmp_path: Path) -> None:
    product_sot_path, registry_path = _write_fixture(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["products"]["br-006"]["views"][0]["source"] = "assets/images/products/other.webp"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_media_intake(product_sot_path, registry_path, repo_root=tmp_path)

    assert report["summary"] == {"supported": 0, "blocked": 1}
    assert "SOURCE_PATH_MISMATCH" in report["products"][0]["blockers"]


def test_rejects_source_bytes_that_no_longer_match_the_bound_hash(tmp_path: Path) -> None:
    product_sot_path, registry_path = _write_fixture(tmp_path)
    asset = tmp_path / "wordpress-theme/skyyrose-flagship/assets/images/products/br-006-onmodel.webp"
    asset.write_bytes(b"tampered source")

    report = validate_media_intake(product_sot_path, registry_path, repo_root=tmp_path)

    assert report["summary"] == {"supported": 0, "blocked": 1}
    assert "SOURCE_HASH_MISMATCH" in report["products"][0]["blockers"]
    assert "SOURCE_BYTES_MISMATCH" in report["products"][0]["blockers"]


def test_rejects_a_stale_product_hash_even_when_the_media_bytes_match(tmp_path: Path) -> None:
    product_sot_path, registry_path = _write_fixture(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["product_hashes"]["br-006"] = "stale-product-hash"
    registry_path.write_text(json.dumps(registry), encoding="utf-8")

    report = validate_media_intake(product_sot_path, registry_path, repo_root=tmp_path)

    assert report["summary"] == {"supported": 0, "blocked": 1}
    assert "PRODUCT_HASH_MISMATCH" in report["products"][0]["blockers"]


def test_explicit_approval_mode_rejects_legacy_records_without_a_reference(tmp_path: Path) -> None:
    product_sot_path, registry_path = _write_fixture(tmp_path)

    report = validate_media_intake(
        product_sot_path,
        registry_path,
        repo_root=tmp_path,
        require_explicit_approval=True,
    )

    assert report["summary"] == {"supported": 0, "blocked": 1}
    assert "EXPLICIT_APPROVAL_REFERENCE_MISSING" in report["products"][0]["blockers"]


def test_generation_receipt_requires_a_hash_bound_local_original(tmp_path: Path) -> None:
    product_sot_path, _registry_path = _write_fixture(tmp_path)
    original = tmp_path / "assets/products/references/br-006-front.jpeg"
    original.parent.mkdir(parents=True)
    original_bytes = b"physical product original"
    original.write_bytes(original_bytes)
    receipt = {
        "sku": "br-006",
        "view": "left",
        "originals": [
            {
                "role": "physical_product_authority",
                "local_path": str(original.relative_to(tmp_path)),
                "sha256": _sha256(original_bytes),
                "bytes": len(original_bytes),
            }
        ],
        "product_sot_sha256": _sha256(product_sot_path.read_bytes()),
        "product_hash": "product-hash-br-006",
        "model": "gpt-image-2",
        "operation": "candidate_only",
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    report = validate_generation_receipt(product_sot_path, receipt_path, repo_root=tmp_path)

    assert report == {"valid": True, "blockers": []}


def test_generation_receipt_rejects_remote_or_hash_drifted_originals(tmp_path: Path) -> None:
    product_sot_path, _registry_path = _write_fixture(tmp_path)
    receipt = {
        "sku": "br-006",
        "view": "left",
        "originals": [
            {
                "role": "physical_product_authority",
                "local_path": "https://example.com/original.jpeg",
                "sha256": "not-a-local-hash",
                "bytes": 1,
            }
        ],
        "product_sot_sha256": _sha256(product_sot_path.read_bytes()),
        "product_hash": "product-hash-br-006",
        "model": "gpt-image-2",
        "operation": "candidate_only",
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    report = validate_generation_receipt(product_sot_path, receipt_path, repo_root=tmp_path)

    assert report["valid"] is False
    assert "ORIGINAL_URL_NOT_ALLOWED" in report["blockers"]
