from __future__ import annotations

import hashlib
import io
from pathlib import Path

import pytest
from PIL import Image

from Comfy.scripts.birefnet_edge_confidence import (
    BIREFNET_MODEL_SHA256,
    EdgeConfidenceError,
    analyze_birefnet_mask,
    write_receipt_exclusive,
)

SOURCE_SHA256 = hashlib.sha256(b"protected-source").hexdigest()


def _png_bytes(values: list[int], size: tuple[int, int]) -> bytes:
    image = Image.new("L", size)
    image.putdata(values)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def _rgba_png_bytes(alpha_values: list[int], size: tuple[int, int]) -> bytes:
    image = Image.new("RGBA", size, (255, 255, 255, 255))
    alpha = Image.new("L", size)
    alpha.putdata(alpha_values)
    image.putalpha(alpha)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def test_receipt_binds_source_model_mask_and_descriptive_metrics() -> None:
    mask = _png_bytes([0, 32, 128, 255], (2, 2))

    receipt = analyze_birefnet_mask(mask, source_sha256=SOURCE_SHA256)

    assert receipt["status"] == "EVIDENCE_RECORDED_REVIEW_REQUIRED"
    assert receipt["source_sha256"] == SOURCE_SHA256
    assert receipt["model"]["sha256"] == BIREFNET_MODEL_SHA256
    assert receipt["mask"] == {
        "sha256": hashlib.sha256(mask).hexdigest(),
        "width": 2,
        "height": 2,
        "mode_analyzed": "L",
    }
    assert receipt["metrics"]["foreground_ratio"] == 0.75
    assert receipt["metrics"]["transparent_ratio"] == 0.25
    assert receipt["metrics"]["opaque_ratio"] == 0.25
    assert receipt["metrics"]["transition_ratio"] == 0.5
    assert receipt["metrics"]["ambiguous_transition_ratio"] == 0.25
    assert receipt["protected_rgb_modified"] is False
    assert receipt["generative_use_allowed"] is False
    assert receipt["quality_decision"] == "UNDETERMINED"
    assert receipt["independent_visual_review_required"] is True


@pytest.mark.parametrize("value", [0, 255])
def test_degenerate_mask_is_blocked(value: int) -> None:
    mask = _png_bytes([value] * 4, (2, 2))

    receipt = analyze_birefnet_mask(mask, source_sha256=SOURCE_SHA256)

    assert receipt["status"] == "BLOCKED_DEGENERATE_MASK"
    assert receipt["quality_decision"] == "UNDETERMINED"


def test_rgba_mask_uses_alpha_instead_of_rgb_luminance() -> None:
    mask = _rgba_png_bytes([0, 64, 192, 255], (2, 2))

    receipt = analyze_birefnet_mask(mask, source_sha256=SOURCE_SHA256)

    assert receipt["mask"]["mode_analyzed"] == "A"
    assert receipt["metrics"]["transparent_ratio"] == 0.25
    assert receipt["metrics"]["opaque_ratio"] == 0.25


def test_invalid_hash_or_image_is_rejected() -> None:
    with pytest.raises(EdgeConfidenceError, match="source_sha256"):
        analyze_birefnet_mask(b"not-an-image", source_sha256="bad")
    with pytest.raises(EdgeConfidenceError, match="readable image"):
        analyze_birefnet_mask(b"not-an-image", source_sha256=SOURCE_SHA256)


def test_receipt_is_create_only(tmp_path: Path) -> None:
    receipt = analyze_birefnet_mask(
        _png_bytes([0, 64, 192, 255], (2, 2)),
        source_sha256=SOURCE_SHA256,
    )
    destination = tmp_path / "edge-confidence.json"

    write_receipt_exclusive(destination, receipt)

    assert destination.is_file()
    with pytest.raises(EdgeConfidenceError, match="already exists"):
        write_receipt_exclusive(destination, receipt)
