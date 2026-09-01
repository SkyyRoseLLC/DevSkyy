#!/usr/bin/env python3
"""Validate and query the fail-closed product-fidelity model registry."""

from __future__ import annotations

import argparse
import hashlib
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
API_CONTRACT_FIELDS = {
    "api_surface",
    "endpoint",
    "request_model",
    "required_generator_fields",
    "forbidden_parameters",
    "parameter_policy",
    "mask",
    "required_receipt_fields",
}
MASK_CONTRACT_FIELDS = {
    "format",
    "alpha_channel_required",
    "alpha_zero_is_editable",
    "same_dimensions_as_input",
    "max_input_bytes",
}
PARAMETER_POLICY_FIELDS = {"allowed", "required", "enums"}


def _validate_api_contract(model: dict[str, Any]) -> None:
    api_contract = model.get("api_contract")
    if api_contract is None:
        return
    if not isinstance(api_contract, dict) or set(api_contract) != API_CONTRACT_FIELDS:
        raise ValueError(f"model {model['id']} has invalid api_contract fields")
    for field in ("api_surface", "endpoint", "request_model"):
        if not isinstance(api_contract[field], str) or not api_contract[field].strip():
            raise ValueError(f"model {model['id']} api_contract.{field} must be non-empty")
    if not api_contract["endpoint"].startswith("/v1/"):
        raise ValueError(f"model {model['id']} api_contract.endpoint must start with /v1/")
    required_fields = api_contract["required_generator_fields"]
    if not isinstance(required_fields, dict) or not required_fields:
        raise ValueError(
            f"model {model['id']} api_contract.required_generator_fields must be an object"
        )
    for field in ("forbidden_parameters", "required_receipt_fields"):
        values = api_contract[field]
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise ValueError(f"model {model['id']} api_contract.{field} must be a string list")
        if len(values) != len(set(values)):
            raise ValueError(f"model {model['id']} api_contract.{field} contains duplicates")
    parameter_policy = api_contract["parameter_policy"]
    if not isinstance(parameter_policy, dict) or set(parameter_policy) != PARAMETER_POLICY_FIELDS:
        raise ValueError(f"model {model['id']} has invalid api_contract.parameter_policy fields")
    for field in ("allowed", "required"):
        values = parameter_policy[field]
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise ValueError(
                f"model {model['id']} api_contract.parameter_policy.{field} must be a string list"
            )
        if len(values) != len(set(values)):
            raise ValueError(
                f"model {model['id']} api_contract.parameter_policy.{field} contains duplicates"
            )
    if not set(parameter_policy["required"]).issubset(parameter_policy["allowed"]):
        raise ValueError(
            f"model {model['id']} api_contract.parameter_policy.required must be allowed"
        )
    enums = parameter_policy["enums"]
    if not isinstance(enums, dict) or not set(enums).issubset(parameter_policy["allowed"]):
        raise ValueError(f"model {model['id']} api_contract.parameter_policy.enums is invalid")
    for field, values in enums.items():
        if (
            not isinstance(values, list)
            or not values
            or not all(isinstance(item, str) and item for item in values)
        ):
            raise ValueError(
                f"model {model['id']} api_contract.parameter_policy.enums.{field} is invalid"
            )
    mask = api_contract["mask"]
    if not isinstance(mask, dict) or set(mask) != MASK_CONTRACT_FIELDS:
        raise ValueError(f"model {model['id']} has invalid api_contract.mask fields")
    if mask["format"] != "png":
        raise ValueError(f"model {model['id']} api_contract.mask.format must be png")
    for field in (
        "alpha_channel_required",
        "alpha_zero_is_editable",
        "same_dimensions_as_input",
    ):
        if mask[field] is not True:
            raise ValueError(f"model {model['id']} api_contract.mask.{field} must be true")
    max_input_bytes = mask["max_input_bytes"]
    if isinstance(max_input_bytes, bool) or not isinstance(max_input_bytes, int):
        raise ValueError(
            f"model {model['id']} api_contract.mask.max_input_bytes must be an integer"
        )
    if max_input_bytes <= 0:
        raise ValueError(f"model {model['id']} api_contract.mask.max_input_bytes must be positive")


def _validate_registry(data: dict[str, Any]) -> dict[str, Any]:
    if data.get("schema") != "product-fidelity-model-registry.v2":
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
        _validate_api_contract(model)
        for key in [model["id"], *model["aliases"]]:
            normalized = key.casefold()
            if normalized in keys:
                raise ValueError(f"duplicate model id or alias: {key}")
            keys.add(normalized)
    return data


def load_registry_snapshot(path: Path) -> tuple[dict[str, Any], str]:
    payload = path.read_bytes()
    data = json.loads(payload.decode("utf-8"))
    return _validate_registry(data), hashlib.sha256(payload).hexdigest()


def load_registry(path: Path) -> dict[str, Any]:
    return load_registry_snapshot(path)[0]


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
        "api_contract": model.get("api_contract"),
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
