#!/usr/bin/env python3
"""Validate and query the fail-closed product-fidelity model registry."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any

DEFAULT_REGISTRY = (
    Path(__file__).resolve().parent.parent / "references" / "model-capability-registry.json"
)
REQUIRED_FIELDS = {
    "id",
    "aliases",
    "provider",
    "modality",
    "status",
    "capabilities",
    "allowed_operations",
    "blocked_operations",
    "fidelity_tier",
    "notes",
    "source",
}
VALID_MODALITIES = {"image", "video", "3d"}


def load_registry(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema") != "product-fidelity-model-registry.v1":
        raise ValueError("unsupported registry schema")
    date.fromisoformat(data["verified_on"])
    models = data.get("models")
    if not isinstance(models, list) or not models:
        raise ValueError("registry models must be a non-empty list")

    keys: set[str] = set()
    for index, model in enumerate(models):
        missing = REQUIRED_FIELDS - set(model)
        if missing:
            raise ValueError(f"model {index} missing fields: {sorted(missing)}")
        if model["modality"] not in VALID_MODALITIES:
            raise ValueError(f"model {model['id']} has invalid modality")
        for field in (
            "aliases",
            "capabilities",
            "allowed_operations",
            "blocked_operations",
        ):
            if not isinstance(model[field], list) or not all(
                isinstance(item, str) for item in model[field]
            ):
                raise ValueError(f"model {model['id']} field {field} must be a string list")
        for key in [model["id"], *model["aliases"]]:
            normalized = key.casefold()
            if normalized in keys:
                raise ValueError(f"duplicate model id or alias: {key}")
            keys.add(normalized)
    return data


def index_registry(data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for model in data["models"]:
        for key in [model["id"], *model["aliases"]]:
            index[key.casefold()] = model
    return index


def validate_request(data: dict[str, Any], model_id: str, operation: str) -> dict[str, Any]:
    model = index_registry(data).get(model_id.casefold())
    if model is None:
        return {
            "status": "BLOCKED",
            "code": "UNKNOWN_MODEL",
            "model": model_id,
            "operation": operation,
            "message": "Unknown models fail closed until first-party capabilities are reviewed and registered.",
        }
    if operation not in model["allowed_operations"]:
        return {
            "status": "BLOCKED",
            "code": "OPERATION_NOT_ALLOWED",
            "model": model["id"],
            "operation": operation,
            "fidelity_tier": model["fidelity_tier"],
            "message": model["notes"],
        }
    return {
        "status": "PASS",
        "code": "REGISTERED_CAPABILITY_MATCH",
        "model": model["id"],
        "operation": operation,
        "fidelity_tier": model["fidelity_tier"],
        "live_availability_verified": False,
        "post_verification_required": True,
        "message": model["notes"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("check")
    list_parser = subparsers.add_parser("list")
    list_parser.add_argument("--modality", choices=sorted(VALID_MODALITIES))
    validate_parser = subparsers.add_parser("validate")
    validate_parser.add_argument("--model", required=True)
    validate_parser.add_argument("--operation", required=True)
    args = parser.parse_args()

    try:
        data = load_registry(args.registry.resolve())
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(
            json.dumps(
                {"status": "BLOCKED", "code": "INVALID_REGISTRY", "message": str(exc)},
                indent=2,
            )
        )
        return 2

    if args.command == "check":
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "models": len(data["models"]),
                    "verified_on": data["verified_on"],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "list":
        models = [
            model
            for model in data["models"]
            if args.modality is None or model["modality"] == args.modality
        ]
        print(json.dumps(models, indent=2))
        return 0

    receipt = validate_request(data, args.model, args.operation)
    print(json.dumps(receipt, indent=2))
    return 0 if receipt["status"] == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
