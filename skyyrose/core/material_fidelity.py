"""Prompt compiler for physical embellishment material locks.

The product dossier owns the facts. This module converts its structured
per-region material locks into a high-salience prompt block shared by image
generation and visual-QA paths. It deliberately describes observable physics,
not inferred manufacturing details.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


MATERIAL_FIDELITY_GUARDRAIL = (
    "EMBELLISHMENT MATERIAL FIDELITY: technique names are physical manufacturing facts, "
    "not interchangeable style words. Molded silicone must read as clean-edged rubber-like "
    "relief with no thread texture; embroidery must resolve as directional thread and stitches; "
    "sublimation must remain inside the textile fibers with no raised edge, pile, patch border, "
    "or rubber relief. Never convert one technique into another, and never move a locked element "
    "to a more visible body region."
)


class MaterialFidelityError(ValueError):
    """Raised before generation when an opted-in material contract is incomplete."""


_FIELDS = (
    ("material_family", "MATERIAL"),
    ("physical_construction", "CONSTRUCTION"),
    ("surface_response", "SURFACE / LIGHT RESPONSE"),
    ("attachment_method", "ATTACHMENT"),
    ("verification_cues", "VERIFY"),
    ("reject_cues", "REJECT"),
)


def _value(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def compile_material_lock_prompt(regions: Iterable[Any]) -> str:
    """Render structured material locks as a deterministic prompt section.

    Regions without a material lock are ignored for backward compatibility.
    Dossiers that declare ``material_lock_version: v1`` are validated upstream
    and therefore cannot reach this function with a partial lock.
    """

    blocks: list[str] = []
    for region in regions:
        lock = _value(region, "material_lock")
        if not lock:
            continue
        region_name = str(_value(region, "region") or "unspecified-region").strip()
        technique = str(_value(region, "technique") or "unspecified-technique").strip()
        lines = [f"REGION: {region_name}", f"TECHNIQUE: {technique}"]
        for key, label in _FIELDS:
            value = _value(lock, key)
            if value:
                lines.append(f"{label}: {str(value).strip()}")
        blocks.append("\n".join(lines))

    if not blocks:
        return ""
    return (
        "MATERIAL LOCKS — PHYSICAL PRODUCT TRUTH (all six facts per region are binding):\n"
        f"{MATERIAL_FIDELITY_GUARDRAIL}\n\n"
        + "\n\n".join(blocks)
    )


def require_complete_material_locks(
    *,
    version: str,
    regions: Iterable[Any],
    context: str,
) -> None:
    """Fail closed before a paid render when an opted-in contract is incomplete."""
    normalized_version = str(version or "").strip()
    if not normalized_version:
        return
    if normalized_version != "v1":
        raise MaterialFidelityError(
            f"{context}: unsupported material_lock_version {normalized_version!r}"
        )
    region_list = list(regions)
    if not region_list:
        raise MaterialFidelityError(
            f"{context}: material_lock_version v1 has no parsed branding regions"
        )
    missing = [
        str(_value(region, "region") or "unspecified-region")
        for region in region_list
        if not _value(region, "material_lock")
    ]
    if missing:
        raise MaterialFidelityError(
            f"{context}: material_lock_version v1 is incomplete for: {', '.join(missing)}"
        )


__all__ = [
    "MATERIAL_FIDELITY_GUARDRAIL",
    "MaterialFidelityError",
    "compile_material_lock_prompt",
    "require_complete_material_locks",
]
