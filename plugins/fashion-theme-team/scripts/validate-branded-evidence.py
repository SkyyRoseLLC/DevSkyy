#!/usr/bin/env python3
"""Validate candidate-bound evidence beyond JSON Schema expressiveness."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def contract_sha(contract: dict[str, object]) -> str:
    material = copy.deepcopy(contract)
    material["provenance"].pop("contract_sha256", None)  # type: ignore[union-attr]
    return sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    )


def candidate_sha(artifacts: list[dict[str, str]]) -> str:
    material = "\n".join(
        f"{item['id']}:{item['path']}:{item['sha256']}"
        for item in sorted(artifacts, key=lambda item: item["id"])
    )
    return sha256(material.encode())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--schema", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path, required=True)
    args = parser.parse_args()
    evidence = json.loads(args.evidence.read_text())
    contract = json.loads(args.contract.read_text())
    schema = json.loads(args.schema.read_text())
    errors = [error.message for error in Draft202012Validator(schema).iter_errors(evidence)]
    if evidence.get("contract_id") != contract.get("contract_id"):
        errors.append("contract ID mismatch")
    actual_contract_sha = contract_sha(contract)
    if (
        contract["provenance"]["contract_sha256"] != actual_contract_sha
        or evidence.get("contract_sha256") != actual_contract_sha
    ):
        errors.append("contract hash mismatch")
    source = args.plugin_root / "vendor/branded-skills" / contract["source"]["relative_path"]
    actual_source_sha = sha256(source.read_bytes()) if source.is_file() else "MISSING"
    if (
        contract["source"]["sha256"] != actual_source_sha
        or evidence.get("source_sha256") != actual_source_sha
    ):
        errors.append("source hash mismatch")
    for artifact in evidence.get("artifacts", []):
        path = args.plugin_root / artifact["path"]
        if not path.is_file() or sha256(path.read_bytes()) != artifact["sha256"]:
            errors.append(f"artifact hash mismatch: {artifact['id']}")
    if evidence.get("candidate_sha256") != candidate_sha(evidence.get("artifacts", [])):
        errors.append("candidate hash mismatch")
    try:
        if datetime.fromisoformat(evidence["started_at"]) > datetime.fromisoformat(
            evidence["completed_at"]
        ):
            errors.append("evidence timestamps are reversed")
        if datetime.fromisoformat(evidence["completed_at"]) > datetime.fromisoformat(
            evidence["review"]["reviewed_at"]
        ):
            errors.append("review predates completed evidence")
    except (KeyError, ValueError):
        errors.append("invalid evidence timestamp")
    reviewer = evidence.get("review", {}).get("reviewer")
    disallowed = set(evidence.get("builder_ids", [])) | set(contract["provenance"]["owners"])
    if reviewer in disallowed:
        errors.append("reviewer is not independent")
    required_checks = {item["id"] for item in contract["verification"]["checks"]}
    observed_checks = {item.get("id") for item in evidence.get("checks", [])}
    if required_checks - observed_checks:
        errors.append("required contract checks are missing")
    if evidence.get("verdict") == "PASS" and any(
        item.get("status") in {"FAIL", "BLOCKED", "UNKNOWN"} for item in evidence.get("checks", [])
    ):
        errors.append("PASS contains a non-passing check")
    if errors:
        print("\n".join(errors))
        return 1
    print("PASS: evidence is schema-valid, hash-bound, ordered, and independently reviewed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
