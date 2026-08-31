#!/usr/bin/env python3
"""Run the contracted GPT Image 2 SG-001 logo correction once."""

from __future__ import annotations

import base64
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np
import openai
from dotenv import dotenv_values
from openai import OpenAI

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "data/candidates/sig-commerce-3-sg001-logo-mask-contract-v5-2026-08-31.json"
TARGET_PATH = (
    ROOT
    / "renders/scroll-world/SIG-COMMERCE-3/protected-input/sg-005-bay-bridge-look-protected-v1.png"
)
REFERENCE_PATH = (
    ROOT / "assets/products/source-photos/signature/sg-001-bay-bridge-shorts-front-authentic.png"
)
MASK_PATH = (
    ROOT
    / "renders/scroll-world/SIG-COMMERCE-3/correction-v5/sg001-reference-pack-v3/openai-edit-mask-alpha-zero-is-editable.png"
)
OUTPUT_DIR = ROOT / "renders/scroll-world/SIG-COMMERCE-3/correction-v5"
RAW_OUTPUT_PATH = OUTPUT_DIR / "sg001-logo-edit-provider-raw-v1.png"
OUTPUT_PATH = OUTPUT_DIR / "sg-005-sg-001-exact-blue-rose-protected-v2.png"
RECEIPT_PATH = OUTPUT_DIR / "sg001-logo-edit-generation-receipt-v1.json"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_path(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


def canonical_contract_hash(contract: dict[str, object]) -> str:
    payload = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return sha256_bytes(payload)


def load_api_key() -> str:
    candidates = (ROOT / ".env", Path("/Users/theceo/DevSkyy/.env"))
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        return key
    for path in candidates:
        if path.is_file():
            value = str(dotenv_values(path).get("OPENAI_API_KEY") or "").strip()
            if value:
                return value
    raise RuntimeError("OPENAI_API_KEY is unavailable")


def main() -> None:
    contract = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
    generator = contract["generator"]
    parameters = generator["parameters"]
    prompt = generator["prompt"]

    expected_hashes = [
        contract["target"]["sha256"],
        contract["references"][0]["sha256"],
    ]
    actual_hashes = [sha256_path(TARGET_PATH), sha256_path(REFERENCE_PATH)]
    if actual_hashes != expected_hashes:
        raise RuntimeError("Contracted request image bytes drifted")

    client = OpenAI(api_key=load_api_key(), timeout=180, max_retries=0)
    with (
        TARGET_PATH.open("rb") as target_file,
        REFERENCE_PATH.open("rb") as reference_file,
        MASK_PATH.open("rb") as mask_file,
    ):
        raw_response = client.images.with_raw_response.edit(
            model=generator["model_id"],
            image=[target_file, reference_file],
            mask=mask_file,
            prompt=prompt,
            size=parameters["size"],
            quality=parameters["quality"],
            output_format=parameters["output_format"],
            background=parameters["background"],
        )
    response = raw_response.parse()
    if not response.data or not response.data[0].b64_json:
        raise RuntimeError("OpenAI image edit returned no image bytes")
    provider_bytes = base64.b64decode(response.data[0].b64_json)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_OUTPUT_PATH.write_bytes(provider_bytes)
    target = cv2.imread(str(TARGET_PATH), cv2.IMREAD_UNCHANGED)
    provider = cv2.imdecode(np.frombuffer(provider_bytes, np.uint8), cv2.IMREAD_UNCHANGED)
    mask = cv2.imread(str(MASK_PATH), cv2.IMREAD_UNCHANGED)
    if target is None or provider is None or mask is None:
        raise RuntimeError("Could not decode target, provider output, or mask")
    if target.shape != provider.shape or target.shape[:2] != mask.shape[:2]:
        raise RuntimeError("Provider output geometry drifted")

    editable = mask[:, :, 3] == 0
    provider_changed = np.any(provider != target, axis=2)
    provider_outside_changed = int(np.count_nonzero(provider_changed & ~editable))
    final = provider.copy()
    final[~editable] = target[~editable]
    final_changed = np.any(final != target, axis=2)
    final_outside_changed = int(np.count_nonzero(final_changed & ~editable))
    final_inside_changed = int(np.count_nonzero(final_changed & editable))
    if final_outside_changed != 0 or final_inside_changed < 1:
        raise RuntimeError("Final masked pixel-integrity contract failed")
    if not cv2.imwrite(str(OUTPUT_PATH), final):
        raise RuntimeError(f"Could not write {OUTPUT_PATH}")

    request_id = raw_response.headers.get("x-request-id", "").strip()
    if not request_id:
        raise RuntimeError("OpenAI response omitted x-request-id")
    receipt = {
        "schema": "product-fidelity-generation-receipt.v1",
        "contract_sha256": canonical_contract_hash(contract),
        "provider": "openai",
        "api_surface": "images",
        "endpoint": "/v1/images/edits",
        "requested_model": generator["model_id"],
        "input_sha256": actual_hashes[0],
        "request_image_sha256s": actual_hashes,
        "mask_path": str(MASK_PATH.relative_to(ROOT)),
        "mask_sha256": sha256_path(MASK_PATH),
        "output_sha256": sha256_path(OUTPUT_PATH),
        "prompt_sha256": sha256_bytes(prompt.encode("utf-8")),
        "request_parameters": parameters,
        "x_request_id": request_id,
        "sdk_version": f"openai-python/{openai.__version__}",
        "provider_output": {
            "path": str(RAW_OUTPUT_PATH.relative_to(ROOT)),
            "sha256": sha256_path(RAW_OUTPUT_PATH),
            "outside_mask_changed_pixels_before_lock_restore": provider_outside_changed,
        },
        "pixel_integrity": {
            "inside_mask_changed_pixels": final_inside_changed,
            "outside_mask_changed_pixels": final_outside_changed,
            "outside_mask_byte_identical": final_outside_changed == 0,
        },
    }
    RECEIPT_PATH.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {k: receipt[k] for k in ("output_sha256", "x_request_id", "pixel_integrity")}, indent=2
        )
    )


if __name__ == "__main__":
    main()
