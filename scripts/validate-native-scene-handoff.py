#!/usr/bin/env python3
"""Validate the portable, candidate-only native-scene handoff."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HANDOFF = ROOT / "docs/design/v2-remodel/native-scene-regeneration-v2"


def sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load(path: Path) -> dict[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"invalid JSON: {path}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"expected an object: {path}")
    return value


def local_path(relative: object) -> Path:
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("reference path must be a non-empty repository-relative path")
    resolved = (ROOT / relative).resolve()
    try:
        resolved.relative_to(ROOT.resolve())
    except ValueError as error:
        raise ValueError(f"reference path escapes repository: {relative}") from error
    return resolved


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, default=DEFAULT_HANDOFF)
    args = parser.parse_args()
    handoff = args.handoff.resolve()
    blockers: list[str] = []
    warnings: list[str] = []

    try:
        product_sot = ROOT / "data/product-sot.json"
        product_sot_hash = sha256(product_sot)
        manifest = load(handoff / "handoff-manifest.json")
        ledger = load(handoff / "phase-ledger.json")
        sources = load(handoff / "source-ledger.json")
    except (OSError, ValueError) as error:
        print(f"BLOCKED {error}")
        return 2

    for name, record, hash_key in (
        ("manifest", manifest, "sha256"),
        ("phase ledger", ledger, "product_sot_sha256"),
        ("source ledger", sources, "sha256"),
    ):
        lock = record.get("source_freshness_lock") or record.get("baseline") or record.get("freshness_lock")
        if not isinstance(lock, dict) or lock.get(hash_key) != product_sot_hash:
            blockers.append(f"{name}: PRODUCT_SOT_HASH_MISMATCH")

    if manifest.get("candidate_images_present") is not False:
        blockers.append("manifest: CANDIDATE_IMAGES_STATE_INVALID")
    promotion = manifest.get("promotion")
    if not isinstance(promotion, dict) or promotion.get("wiring_allowed") is not False or promotion.get("deployment_allowed") is not False:
        blockers.append("manifest: PROMOTION_BOUNDARY_INVALID")

    contracts = sorted((handoff / "scene-contracts").glob("*.json"))
    if len(contracts) != 7:
        blockers.append(f"scene contracts: EXPECTED_7_GOT_{len(contracts)}")
    for contract_path in contracts:
        try:
            contract = load(contract_path)
        except ValueError as error:
            blockers.append(str(error))
            continue
        contract_promotion = contract.get("promotion")
        if not isinstance(contract_promotion, dict) or contract_promotion.get("wiring_allowed") is not False or contract_promotion.get("deployment_allowed") is not False:
            blockers.append(f"{contract_path.name}: PROMOTION_BOUNDARY_INVALID")
        for reference in contract.get("references", []):
            if not isinstance(reference, dict):
                blockers.append(f"{contract_path.name}: INVALID_REFERENCE")
                continue
            try:
                asset = local_path(reference.get("path"))
            except ValueError as error:
                blockers.append(f"{contract_path.name}: {error}")
                continue
            if not asset.is_file():
                blockers.append(f"{contract_path.name}: MISSING_REFERENCE {reference.get('path')}")
            elif sha256(asset) != reference.get("sha256"):
                blockers.append(f"{contract_path.name}: REFERENCE_HASH_MISMATCH {reference.get('path')}")

    for receipt_path in sorted((handoff / "preflight-receipts").glob("*.json")):
        try:
            receipt = load(receipt_path)
            if receipt.get("status") != "BLOCKED":
                blockers.append(f"{receipt_path.name}: PRELIGHT_STATUS_NOT_BLOCKED")
        except ValueError as error:
            blockers.append(str(error))

    if sources.get("workspace") != str(ROOT):
        warnings.append("source-ledger workspace is historical; contract paths are portable and separately hash-checked")

    for warning in warnings:
        print(f"WARNING {warning}")
    for blocker in blockers:
        print(f"BLOCKED {blocker}")
    if blockers:
        return 1
    print(f"PASS {len(contracts)} contracts, current product SOT, all contract authority bytes exact")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
