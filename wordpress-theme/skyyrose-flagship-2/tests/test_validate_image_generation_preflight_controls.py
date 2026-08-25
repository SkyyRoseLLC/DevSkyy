from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest
from PIL import Image


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "validate-image-generation-preflight.py"
SPEC = importlib.util.spec_from_file_location("image_generation_preflight", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def job(target: Path, mask: Path) -> dict[str, object]:
    return {
        "input_images": [{"role": "edit_target", "path": target.name}],
        "localized_edit_controls": {
            "provider_operation": "explicit_mask_edit",
            "mask_polarity": "white_is_editable_black_is_locked",
            "outside_mask_changed_pixels": 0,
            "semantic_review_required": True,
            "mask": {"path": mask.name, "sha256": sha256(mask)},
        },
    }


def test_localized_edit_requires_bound_canonical_mask(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "target.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (20, 10), "black").save(target)
    Image.new("L", (20, 10), 0).save(mask)
    with Image.open(mask) as opened:
        editable = opened.copy()
    editable.putpixel((5, 5), 255)
    editable.save(mask)
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)

    audit: dict[str, dict[str, object]] = {}
    MODULE.validate_localized_edit_controls(audit, "job-1", job(target, mask))
    assert str(mask.relative_to(tmp_path)) in audit


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("provider_operation", "semantic_edit"),
        ("mask_polarity", "transparent_is_editable"),
        ("outside_mask_changed_pixels", 1),
        ("semantic_review_required", False),
    ],
)
def test_localized_edit_rejects_unsafe_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: object
) -> None:
    target = tmp_path / "target.png"
    mask = tmp_path / "mask.png"
    Image.new("RGB", (20, 10), "black").save(target)
    Image.new("L", (20, 10), 255).save(mask)
    payload = job(target, mask)
    controls = payload["localized_edit_controls"]
    assert isinstance(controls, dict)
    controls[field] = value
    monkeypatch.setattr(MODULE, "ROOT", tmp_path)

    with pytest.raises(SystemExit):
        MODULE.validate_localized_edit_controls({}, "job-1", payload)
