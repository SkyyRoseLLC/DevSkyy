#!/usr/bin/env python3
"""Run the approved, receipt-bound GPT-Image-2 geometry candidate pilot.

This runner intentionally produces candidate-only assets. It validates every
preflight receipt before a provider call, refuses to overwrite any output, and
adds a local output SHA-256 and byte count to its original receipt immediately
after a successful response. It does not wire, promote, or deploy media.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from scripts.oai_render.client import OAIImageClient
from skyyrose.core.on_model_media_intake import validate_generation_receipt


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RECEIPTS = REPO_ROOT / "data/candidates/geometry-pilot-2026-08-26/receipts"
PRODUCT_SOT = REPO_ROOT / "data/product-sot.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_receipt(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def require_preflight(path: Path) -> dict[str, object]:
    result = validate_generation_receipt(PRODUCT_SOT, path, repo_root=REPO_ROOT)
    if not result["valid"]:
        raise RuntimeError(f"Preflight receipt failed for {path}: {', '.join(result['blockers'])}")
    receipt = load_receipt(path)
    if receipt.get("receipt_stage") != "preflight":
        raise RuntimeError(f"Receipt is not an unused preflight receipt: {path}")
    if "output" in receipt:
        raise RuntimeError(f"Receipt already has an output binding: {path}")
    return receipt


def relative_output(receipt: dict[str, object]) -> str:
    planned = receipt.get("planned_output")
    if not isinstance(planned, str) or not planned:
        raise RuntimeError("Receipt requires a non-empty planned_output path")
    output = (REPO_ROOT / planned).resolve()
    try:
        output.relative_to(REPO_ROOT.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Output path escapes repository: {planned}") from exc
    if output.exists():
        raise RuntimeError(f"Refusing to overwrite existing candidate output: {planned}")
    return planned


def reference_paths(receipt: dict[str, object]) -> list[Path]:
    originals = receipt.get("originals")
    if not isinstance(originals, list):
        raise RuntimeError("Receipt originals are missing")
    paths: list[Path] = []
    for original in originals:
        if not isinstance(original, dict) or not isinstance(original.get("local_path"), str):
            raise RuntimeError("Receipt has an invalid local original")
        paths.append(REPO_ROOT / original["local_path"])
    return paths


def write_and_bind_output(path: Path, receipt: dict[str, object], image_bytes: bytes) -> None:
    planned = relative_output(receipt)
    output = REPO_ROOT / planned
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    temporary.write_bytes(image_bytes)
    os.replace(temporary, output)

    prior_text = path.read_text(encoding="utf-8")
    receipt["receipt_stage"] = "generated_candidate"
    receipt["output"] = {
        "local_path": planned,
        "sha256": sha256_bytes(image_bytes),
        "bytes": len(image_bytes),
    }
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")

    result = validate_generation_receipt(PRODUCT_SOT, path, repo_root=REPO_ROOT, require_output=True)
    if not result["valid"]:
        path.write_text(prior_text, encoding="utf-8")
        raise RuntimeError(f"Output receipt failed for {path}: {', '.join(result['blockers'])}")


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument("--receipts-dir", type=Path, default=DEFAULT_RECEIPTS)
    command.add_argument("--max-images", type=int, default=18)
    command.add_argument("--execute", action="store_true", help="permit provider calls after preflight validation")
    return command


def main() -> int:
    args = parser().parse_args()
    receipts = sorted(args.receipts_dir.glob("*.json"))
    if not receipts:
        raise RuntimeError(f"No receipts found in {args.receipts_dir}")
    if len(receipts) > args.max_images:
        raise RuntimeError(f"Refusing {len(receipts)} images: max-images is {args.max_images}")

    plans: list[tuple[Path, dict[str, object]]] = []
    for receipt_path in receipts:
        receipt = require_preflight(receipt_path)
        relative_output(receipt)
        plans.append((receipt_path, receipt))
        print(f"PRELIGHT PASS {receipt['sku']} {receipt['view']} -> {receipt['planned_output']}")

    if not args.execute:
        print(f"DRY RUN: {len(plans)} receipt-bound candidate images; no provider call made.")
        return 0

    client = OAIImageClient()
    for number, (receipt_path, receipt) in enumerate(plans, start=1):
        prompt = receipt.get("prompt")
        if not isinstance(prompt, str) or not prompt:
            raise RuntimeError(f"Receipt prompt is missing: {receipt_path}")
        print(f"RENDER {number}/{len(plans)} {receipt['sku']} {receipt['view']}", flush=True)
        image_bytes = client.edit(prompt=prompt, image_paths=reference_paths(receipt))
        write_and_bind_output(receipt_path, receipt, image_bytes)
        print(f"RECEIPT PASS {number}/{len(plans)} {receipt['sku']} {receipt['view']}", flush=True)

    print(f"COMPLETE: {len(plans)} candidate images remain quarantined pending founder review.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, json.JSONDecodeError) as exc:
        print(f"BLOCKED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
