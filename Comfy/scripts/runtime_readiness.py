#!/usr/bin/env python3
"""Read-only FASHN and Comfy readiness checks for governed scene execution."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

import httpx

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skyyrose.integrations.comfy_client import (  # noqa: E402
    DEFAULT_COMFY_BASE_URL,
    REQUIRED_BACKGROUND_MODEL,
    ComfyClient,
    ComfyRuntimeError,
    workflow_sha256,
)
from skyyrose.integrations.fashn_client import FashnClient, FashnError  # noqa: E402

BIREFNET_PATH = Path("/Users/theceo/ComfyUI-Shared/models/background_removal/birefnet.safetensors")
BIREFNET_SHA256 = "9ab37426bf4de0567af6b5d21b16151357149139362e6e8992021b8ce356a154"
WORKFLOWS = (
    (PROJECT_ROOT / "Comfy/workflows/lh-commerce-2-runway-environment-a1.api.json", None),
    (
        PROJECT_ROOT / "Comfy/workflows/lh-commerce-2-birefnet-segmentation-a1.api.json",
        "LH-COMMERCE-2",
    ),
)


def _sha256(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


async def check(*, base_url: str, minimum_fashn_credits: int) -> dict:
    if minimum_fashn_credits < 0:
        raise ValueError("minimum FASHN credits must be non-negative")
    result: dict = {
        "schema": "skyyrose.scene-runtime-readiness/1",
        "status": "BLOCKED",
        "fashn": {"status": "BLOCKED"},
        "comfy": {"status": "BLOCKED"},
        "birefnet": {"status": "BLOCKED"},
    }
    if BIREFNET_PATH.is_file():
        actual = _sha256(BIREFNET_PATH)
        result["birefnet"] = {
            "status": "PASS" if actual == BIREFNET_SHA256 else "BLOCKED",
            "path": str(BIREFNET_PATH),
            "sha256": actual,
            "expected_sha256": BIREFNET_SHA256,
        }
    else:
        result["birefnet"]["reason"] = "MODEL_FILE_MISSING"

    try:
        async with ComfyClient(base_url=base_url) as comfy:
            snapshot = await comfy.inspect_runtime()
            workflow_checks = []
            for workflow_path, governing_scene_id in WORKFLOWS:
                workflow = json.loads(workflow_path.read_text(encoding="utf-8"))
                sealed = await comfy.validate_workflow(
                    workflow, governing_scene_id=governing_scene_id
                )
                workflow_checks.append(
                    {
                        "path": str(workflow_path.relative_to(PROJECT_ROOT)),
                        "sha256": workflow_sha256(workflow),
                        "node_classes": list(sealed.node_classes),
                        "expected_output_node_ids": list(sealed.expected_output_node_ids),
                        "live_schema_sha256": sealed.live_schema_sha256,
                    }
                )
    except (ComfyRuntimeError, httpx.HTTPError, OSError, ValueError) as exc:
        result["comfy"] = {"status": "BLOCKED", "reason": str(exc)}
    else:
        result["comfy"] = {
            "status": "PASS",
            "base_url": snapshot.base_url,
            "comfyui_version": snapshot.comfyui_version,
            "frontend_version": snapshot.frontend_version,
            "input_directory": snapshot.input_directory,
            "output_directory": snapshot.output_directory,
            "required_nodes": list(snapshot.required_nodes),
            "background_models": list(snapshot.background_models),
            "required_background_model": REQUIRED_BACKGROUND_MODEL,
            "validated_workflows": workflow_checks,
        }

    try:
        async with FashnClient.from_default() as fashn:
            balance = await fashn.get_credits()
    except (FashnError, KeyError, OSError, ValueError) as exc:
        result["fashn"] = {
            "status": "BLOCKED",
            "minimum_required": minimum_fashn_credits,
            "reason": str(exc),
        }
    else:
        result["fashn"] = {
            "status": "PASS" if balance.total >= minimum_fashn_credits else "BLOCKED",
            "minimum_required": minimum_fashn_credits,
            "credits": {
                "total": balance.total,
                "subscription": balance.subscription,
                "on_demand": balance.on_demand,
            },
        }
    result["status"] = (
        "PASS"
        if all(result[key]["status"] == "PASS" for key in ("fashn", "comfy", "birefnet"))
        else "BLOCKED"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default=DEFAULT_COMFY_BASE_URL)
    parser.add_argument("--minimum-fashn-credits", type=int, default=8)
    args = parser.parse_args()
    try:
        result = asyncio.run(
            check(base_url=args.base_url, minimum_fashn_credits=args.minimum_fashn_credits)
        )
    except ValueError as exc:
        result = {
            "schema": "skyyrose.scene-runtime-readiness/1",
            "status": "BLOCKED",
            "reason": str(exc),
        }
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "PASS" else 2


if __name__ == "__main__":
    sys.exit(main())
