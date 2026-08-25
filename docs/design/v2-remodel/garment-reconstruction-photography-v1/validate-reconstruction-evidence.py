#!/usr/bin/env python3
"""Fail-closed validation for the candidate-only reconstruction/photography intake."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[3]
PROGRAM_PATH = ROOT / "reconstruction-program.json"


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected object in {path}")
    return value


def validate_receipt(program: dict[str, Any], sku: str, errors: list[str]) -> bool:
    requirements = program["receipt_requirements"]
    receipt_path = ROOT / requirements["relative_path"].replace("<sku>", sku)
    if not receipt_path.is_file():
        errors.append(f"{sku}: CAPTURE_RECEIPT_MISSING ({receipt_path.relative_to(ROOT)})")
        return False
    receipt = load_json(receipt_path)
    for field in requirements["required_fields"]:
        if field not in receipt or receipt[field] in (None, "", [], {}):
            errors.append(f"{sku}: RECEIPT_FIELD_MISSING ({field})")
    if receipt.get("sku") != sku:
        errors.append(f"{sku}: RECEIPT_SKU_MISMATCH ({receipt.get('sku')!r})")
    if receipt.get("authority_tier") not in program["accepted_exact_authority_tiers"]:
        errors.append(f"{sku}: RECONSTRUCTION_AUTHORITY_TIER_REJECTED")
    required_views = set(requirements["required_geometry_views"])
    present_views = set(receipt.get("geometry_views", []))
    if not required_views.issubset(present_views):
        errors.append(f"{sku}: GEOMETRY_VIEWS_MISSING ({','.join(sorted(required_views - present_views))})")
    kinds = {file.get("kind") for file in receipt.get("files", []) if isinstance(file, dict)}
    for kind in requirements["required_file_kinds"]:
        if kind not in kinds:
            errors.append(f"{sku}: FILE_KIND_MISSING ({kind})")
    for file in receipt.get("files", []):
        if not isinstance(file, dict):
            errors.append(f"{sku}: INVALID_FILE_RECEIPT")
            continue
        for field in requirements["required_file_fields"]:
            if not file.get(field):
                errors.append(f"{sku}: FILE_FIELD_MISSING ({field})")
        relative_path = file.get("relative_path")
        expected_sha = file.get("sha256")
        if not relative_path or not expected_sha:
            continue
        asset_path = receipt_path.parent / relative_path
        if not asset_path.is_file():
            errors.append(f"{sku}: HASH_BOUND_SOURCE_MISSING ({asset_path.relative_to(ROOT)})")
        elif digest(asset_path) != expected_sha:
            errors.append(f"{sku}: HASH_MISMATCH ({asset_path.relative_to(ROOT)})")
    return not any(error.startswith(f"{sku}:") for error in errors)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--strict", action="store_true", help="Return non-zero while any SKU evidence is pending.")
    args = parser.parse_args()
    program = load_json(PROGRAM_PATH)
    errors: list[str] = []
    if program.get("schema") != "skyyrose.verified-garment-reconstruction-photography.v1":
        errors.append("PROGRAM_SCHEMA_INVALID")
    sot = REPO / program["source_lock"]["product_sot_path"]
    if not sot.is_file():
        errors.append("PRODUCT_SOT_MISSING")
    elif digest(sot) != program["source_lock"]["product_sot_sha256"]:
        errors.append("PRODUCT_SOT_HASH_DRIFT")
    sku_requirements = program.get("sku_requirements", [])
    skus = [entry.get("sku") for entry in sku_requirements]
    if len(skus) != 16 or len(set(skus)) != 16 or any(not sku for sku in skus):
        errors.append("SKU_REGISTER_INVALID")
    for sku in skus:
        validate_receipt(program, sku, errors)
    if errors:
        print("status PENDING" if all("CAPTURE_RECEIPT_MISSING" in error for error in errors) else "status BLOCKED")
        print(f"sku_count {len(skus)}")
        for error in errors:
            print(f"blocker {error}")
        return 1 if args.strict else 0
    print("status EVIDENCE_COMPLETE_PENDING_INDEPENDENT_FIDELITY_REVIEW")
    print(f"sku_count {len(skus)}")
    print("promotion NONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
