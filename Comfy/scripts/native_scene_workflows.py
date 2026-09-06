"""Deterministic Comfy API workflows for protected SkyyRose scene integration."""

from __future__ import annotations

from typing import Any

from skyyrose.integrations.comfy_client import (
    verify_protected_foreground_rgb as _verify_protected_foreground_rgb,
)

MASTER_WIDTH = 2048
MASTER_HEIGHT = 1152
BIREFNET_MODEL = "birefnet.safetensors"


def verify_protected_foreground_rgb(
    *,
    source_bytes: bytes,
    protected_rgba_bytes: bytes,
    alpha_mask_bytes: bytes,
    governing_scene_id: str,
    expected_scene_contract_sha256: str,
) -> dict[str, Any]:
    """Prove segmentation changed alpha only using scene-bound coverage authority."""
    return _verify_protected_foreground_rgb(
        source_bytes=source_bytes,
        protected_rgba_bytes=protected_rgba_bytes,
        alpha_mask_bytes=alpha_mask_bytes,
        governing_scene_id=governing_scene_id,
        expected_scene_contract_sha256=expected_scene_contract_sha256,
    )


def _require_protected_rgb_receipt(
    receipt: dict[str, Any],
    *,
    protected_foreground_filename: str,
    source_bytes: bytes,
    protected_foreground_bytes: bytes,
    alpha_mask_bytes: bytes,
    governing_scene_id: str,
    expected_scene_contract_sha256: str,
) -> None:
    asset = receipt.get("protected_asset")
    verification = asset.get("verification") if isinstance(asset, dict) else None
    required_hashes = ("source_sha256", "protected_output_sha256", "alpha_mask_sha256")
    recomputed = verify_protected_foreground_rgb(
        source_bytes=source_bytes,
        protected_rgba_bytes=protected_foreground_bytes,
        alpha_mask_bytes=alpha_mask_bytes,
        governing_scene_id=governing_scene_id,
        expected_scene_contract_sha256=expected_scene_contract_sha256,
    )
    if (
        receipt.get("schema") != "skyyrose.comfy-execution-receipt/1"
        or "JoinImageWithAlpha" not in receipt.get("node_classes", [])
        or not isinstance(asset, dict)
        or asset.get("foreground_output", {}).get("filename") != protected_foreground_filename
        or not isinstance(verification, dict)
        or verification.get("schema") != "skyyrose.protected-rgb-verification/1"
        or asset.get("source_upload", {}).get("sha256") != verification.get("source_sha256")
        or asset.get("foreground_output", {}).get("sha256")
        != verification.get("protected_output_sha256")
        or asset.get("mask_output", {}).get("sha256") != verification.get("alpha_mask_sha256")
        or verification != recomputed
        or verification.get("status") != "PASS"
        or verification.get("changed_pixel_count") != 0
        or verification.get("alpha_mismatch_count") != 0
        or verification.get("coverage_pass") is not True
        or not isinstance(verification.get("protected_pixel_count"), int)
        or verification["protected_pixel_count"] <= 0
        or any(
            not isinstance(verification.get(key), str) or len(verification[key]) != 64
            for key in required_hashes
        )
    ):
        raise ValueError("protected foreground requires a bound passing Comfy RGB receipt")


def _resize(input_ref: list[Any], *, width: int, height: int, method: str = "area") -> dict:
    return {
        "class_type": "ResizeImageMaskNode",
        "inputs": {
            "input": input_ref,
            "resize_type": {
                "resize_type": "scale dimensions",
                "width": width,
                "height": height,
                "crop": "disabled",
            },
            "scale_method": method,
        },
    }


def build_birefnet_segmentation_workflow(
    *,
    candidate_filename: str,
    filename_prefix: str = "SkyyRose/LH-COMMERCE-2/protected-foreground-a1",
) -> dict[str, Any]:
    """Extract alpha while keeping the accepted candidate RGB as the source."""
    if not candidate_filename.strip():
        raise ValueError("candidate_filename is required")
    return {
        "1": {"class_type": "LoadImage", "inputs": {"image": candidate_filename}},
        "2": {
            "class_type": "LoadBackgroundRemovalModel",
            "inputs": {"bg_removal_name": BIREFNET_MODEL},
        },
        "3": {
            "class_type": "RemoveBackground",
            "inputs": {"bg_removal_model": ["2", 0], "image": ["1", 0]},
        },
        "4": {
            "class_type": "JoinImageWithAlpha",
            "inputs": {"image": ["1", 0], "alpha": ["3", 0]},
        },
        "5": {
            "class_type": "SaveImage",
            "inputs": {"images": ["4", 0], "filename_prefix": filename_prefix},
        },
        "6": {"class_type": "MaskToImage", "inputs": {"mask": ["3", 0]}},
        "7": {
            "class_type": "SaveImage",
            "inputs": {"images": ["6", 0], "filename_prefix": f"{filename_prefix}-mask"},
        },
    }


def build_lh_native_composite_workflow(
    *,
    background_filename: str,
    protected_foreground_filename: str,
    protected_rose_filename: str,
    protected_asset_receipt: dict[str, Any],
    protected_source_bytes: bytes,
    protected_foreground_bytes: bytes,
    protected_mask_bytes: bytes,
    governing_scene_id: str,
    expected_scene_contract_sha256: str,
    filename_prefix: str = "SkyyRose/LH-COMMERCE-2/master-a1",
) -> dict[str, Any]:
    """Composite protected customer/product and rose layers with grounded shadows.

    Inputs are already approved alpha-bearing PNGs. Resizing is explicit and
    paired for RGB and alpha so the source transforms remain auditable.
    """
    for label, value in {
        "background_filename": background_filename,
        "protected_foreground_filename": protected_foreground_filename,
        "protected_rose_filename": protected_rose_filename,
    }.items():
        if not value.strip():
            raise ValueError(f"{label} is required")
    _require_protected_rgb_receipt(
        protected_asset_receipt,
        protected_foreground_filename=protected_foreground_filename,
        source_bytes=protected_source_bytes,
        protected_foreground_bytes=protected_foreground_bytes,
        alpha_mask_bytes=protected_mask_bytes,
        governing_scene_id=governing_scene_id,
        expected_scene_contract_sha256=expected_scene_contract_sha256,
    )

    workflow: dict[str, Any] = {
        "1": {"class_type": "LoadImage", "inputs": {"image": background_filename}},
        "2": _resize(["1", 0], width=MASTER_WIDTH, height=MASTER_HEIGHT, method="lanczos"),
        "3": {
            "class_type": "LoadImage",
            "inputs": {"image": protected_foreground_filename},
            "_meta": {"title": "Accepted FASHN RGB plus BiRefNet alpha"},
        },
        "4": _resize(["3", 0], width=610, height=1080),
        "5": _resize(["3", 1], width=610, height=1080),
        "6": {
            "class_type": "GrowMask",
            "inputs": {"mask": ["5", 0], "expand": 10, "tapered_corners": True},
        },
        "7": {"class_type": "MaskToImage", "inputs": {"mask": ["6", 0]}},
        "8": {
            "class_type": "ImageBlur",
            "inputs": {"image": ["7", 0], "blur_radius": 18, "sigma": 5.0},
        },
        "9": {"class_type": "ImageToMask", "inputs": {"image": ["8", 0], "channel": "red"}},
        "10": {
            "class_type": "EmptyImage",
            "inputs": {"width": 610, "height": 1080, "batch_size": 1, "color": 0},
        },
        "11": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["2", 0],
                "source": ["10", 0],
                "x": 174,
                "y": 58,
                "resize_source": False,
                "mask": ["9", 0],
            },
            "_meta": {"title": "Soft customer contact and cast shadow"},
        },
        "12": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["11", 0],
                "source": ["4", 0],
                "x": 160,
                "y": 44,
                "resize_source": False,
                "mask": ["5", 0],
            },
            "_meta": {"title": "Protected LH-003 customer placement"},
        },
        "13": {
            "class_type": "LoadImage",
            "inputs": {"image": protected_rose_filename},
            "_meta": {"title": "Exact protected enchanted rose dome and plinth"},
        },
        "14": _resize(["13", 0], width=540, height=790),
        "15": _resize(["13", 1], width=540, height=790),
        "16": {
            "class_type": "GrowMask",
            "inputs": {"mask": ["15", 0], "expand": 8, "tapered_corners": True},
        },
        "17": {"class_type": "MaskToImage", "inputs": {"mask": ["16", 0]}},
        "18": {
            "class_type": "ImageBlur",
            "inputs": {"image": ["17", 0], "blur_radius": 16, "sigma": 4.0},
        },
        "19": {"class_type": "ImageToMask", "inputs": {"image": ["18", 0], "channel": "red"}},
        "20": {
            "class_type": "EmptyImage",
            "inputs": {"width": 540, "height": 790, "batch_size": 1, "color": 0},
        },
        "21": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["12", 0],
                "source": ["20", 0],
                "x": 1360,
                "y": 328,
                "resize_source": False,
                "mask": ["19", 0],
            },
            "_meta": {"title": "Soft rose plinth contact shadow"},
        },
        "22": {
            "class_type": "ImageCompositeMasked",
            "inputs": {
                "destination": ["21", 0],
                "source": ["14", 0],
                "x": 1344,
                "y": 312,
                "resize_source": False,
                "mask": ["15", 0],
            },
            "_meta": {"title": "Protected rose monument placement"},
        },
        "23": {
            "class_type": "SaveImage",
            "inputs": {"images": ["22", 0], "filename_prefix": filename_prefix},
        },
    }
    return workflow
