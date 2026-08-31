#!/usr/bin/env python3
"""Build the BR-COMMERCE-2 v6 protected cast and review composite.

This is deliberately deterministic. It patches only the two BR-007 side panels
from independent wearer-left and wearer-right physical-source photographs, then
derives a real alpha matte from the baked Playground checkerboard. No generative
model is used and pixels outside the explicit panel mask remain byte-identical.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

REPO = Path(__file__).resolve().parents[1]
TARGET = (
    REPO
    / "renders/scroll-world/BR-COMMERCE-2/founder-supplied-source/br-commerce-2-playground-dual-cast-target-2026-08-31.png"
)
WEARER_LEFT = REPO / "assets/products/source-photos/black-rose/br-007-shorts-wearer-left.jpeg"
WEARER_RIGHT = REPO / "assets/products/source-photos/black-rose/br-007-shorts-wearer-right.jpeg"
BACKGROUND = (
    REPO
    / "renders/scroll-world/BR-COMMERCE-2/generated-backgrounds/BR-COMMERCE-2-empty-enclosed-font-statue-gallery-plate-v2.png"
)
STATUE = (
    REPO
    / "renders/scroll-world/BR-COMMERCE-2/protected-composite-layers-v4/black-rose-font-statue-alpha-v1-scene-scale.png"
)
OUT_DIR = REPO / "renders/scroll-world/BR-COMMERCE-2/protected-composite-layers-v6"
REVIEW_DIR = REPO / "renders/scroll-world/BR-COMMERCE-2/review"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_bgr(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    image = cv2.imread(str(path), flags)
    if image is None:
        raise FileNotFoundError(path)
    return image


def polygon_mask(shape: tuple[int, int], points: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=np.uint8)
    cv2.fillPoly(mask, [points.astype(np.int32)], 255)
    return mask


def warp_panel(
    source: np.ndarray,
    source_quad: np.ndarray,
    destination_quad: np.ndarray,
    destination_polygon: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    height, width = output_shape
    transform = cv2.getPerspectiveTransform(
        source_quad.astype(np.float32), destination_quad.astype(np.float32)
    )
    warped = cv2.warpPerspective(
        source,
        transform,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )
    warped_mask = polygon_mask((height, width), destination_polygon)
    return warped, warped_mask


def match_panel_light(source: np.ndarray, target: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Keep source chroma/details while inheriting the target garment's light falloff."""
    source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
    target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)
    soft_target_l = cv2.GaussianBlur(target_lab[:, :, 0], (0, 0), 5.0)
    source_l = source_lab[:, :, 0]
    active = mask > 0
    if np.any(active):
        source_mean = float(np.mean(source_l[active]))
        target_mean = float(np.mean(soft_target_l[active]))
        light_delta = np.clip(soft_target_l - target_mean, -24.0, 24.0)
        source_lab[:, :, 0] = np.clip(
            source_l + light_delta + (target_mean - source_mean) * 0.22, 0, 255
        )
    return cv2.cvtColor(source_lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def composite_panel(base: np.ndarray, patch: np.ndarray, mask: np.ndarray) -> np.ndarray:
    alpha = cv2.GaussianBlur(mask, (0, 0), 0.65).astype(np.float32) / 255.0
    alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]
    merged = np.rint(patch.astype(np.float32) * alpha + base.astype(np.float32) * (1.0 - alpha))
    return np.clip(merged, 0, 255).astype(np.uint8)


def extract_logo(
    source: np.ndarray,
    crop: tuple[int, int, int, int],
    *,
    gray_limit: int,
    saturation_limit: int | None = None,
) -> np.ndarray:
    """Extract stitched logo pixels from the pale physical mesh around them."""
    x, y, width, height = crop
    patch = source[y : y + height, x : x + width].copy()
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    selected = gray < gray_limit
    if saturation_limit is not None:
        selected |= (hsv[:, :, 1] > saturation_limit) & (hsv[:, :, 2] < 225)
    seed = selected.astype(np.uint8) * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(seed)
    alpha = np.zeros_like(seed)
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] >= 5:
            alpha[labels == label] = 255
    alpha = cv2.dilate(alpha, np.ones((2, 2), np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.55)
    rgba = cv2.cvtColor(patch, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    return rgba


def warp_rgba(
    source: np.ndarray, destination_quad: np.ndarray, output_shape: tuple[int, int]
) -> np.ndarray:
    height, width = output_shape
    source_quad = np.array(
        [
            [0, 0],
            [source.shape[1] - 1, 0],
            [0, source.shape[0] - 1],
            [source.shape[1] - 1, source.shape[0] - 1],
        ],
        dtype=np.float32,
    )
    transform = cv2.getPerspectiveTransform(source_quad, destination_quad.astype(np.float32))
    return cv2.warpPerspective(
        source,
        transform,
        (width, height),
        flags=cv2.INTER_LANCZOS4,
        borderMode=cv2.BORDER_CONSTANT,
    )


def composite_rgba(base: np.ndarray, overlay: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    alpha = overlay[:, :, 3].astype(np.float32) / 255.0
    merged = np.rint(
        overlay[:, :, :3].astype(np.float32) * alpha[:, :, None]
        + base.astype(np.float32) * (1.0 - alpha[:, :, None])
    )
    return np.clip(merged, 0, 255).astype(np.uint8), (alpha > 0).astype(np.uint8) * 255


def derive_subject_alpha(rgb: np.ndarray) -> np.ndarray:
    """Remove the two known checker colors and retain connected human silhouettes."""
    values = rgb.astype(np.int16)
    backgrounds = (
        np.array([242, 242, 242], dtype=np.int16),
        np.array([254, 254, 254], dtype=np.int16),
    )
    distance = np.minimum(
        np.linalg.norm(values - backgrounds[0], axis=2),
        np.linalg.norm(values - backgrounds[1], axis=2),
    )
    seed = (distance > 11.0).astype(np.uint8) * 255
    seed = cv2.morphologyEx(seed, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)

    contours, _ = cv2.findContours(seed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hard = np.zeros(seed.shape, dtype=np.uint8)
    for contour in contours:
        if cv2.contourArea(contour) >= 35.0:
            cv2.drawContours(hard, [contour], -1, 255, thickness=cv2.FILLED)

    # A tight sub-pixel feather avoids carrying the baked checker into the scene.
    hard = cv2.morphologyEx(hard, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1)
    hard = cv2.erode(hard, np.ones((3, 3), np.uint8), iterations=2)
    alpha = cv2.GaussianBlur(hard, (0, 0), 0.55)
    alpha[alpha < 6] = 0
    alpha[alpha > 249] = 255
    return alpha


def alpha_over(base: np.ndarray, overlay: np.ndarray, x: int = 0, y: int = 0) -> np.ndarray:
    result = base.copy()
    oh, ow = overlay.shape[:2]
    bh, bw = base.shape[:2]
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + ow, bw), min(y + oh, bh)
    if x0 >= x1 or y0 >= y1:
        return result
    src = overlay[y0 - y : y1 - y, x0 - x : x1 - x]
    dst = result[y0:y1, x0:x1]
    alpha = src[:, :, 3:4].astype(np.float32) / 255.0
    result[y0:y1, x0:x1] = np.rint(src[:, :, :3] * alpha + dst * (1.0 - alpha)).astype(np.uint8)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-receipt", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    target = read_bgr(TARGET)
    physical_left = read_bgr(WEARER_LEFT)
    physical_right = read_bgr(WEARER_RIGHT)
    height, width = target.shape[:2]

    # Independent physical authorities; never mirror a panel to manufacture its opposite.
    # First remove the wrong wearer-right Black Rose mark from viewer-right, sampling only
    # the surrounding white mesh of that same worn panel.
    erase_mask = polygon_mask(
        (height, width), np.array([[669, 522], [687, 518], [691, 568], [673, 574]])
    )
    patched = cv2.inpaint(target, erase_mask, 3.0, cv2.INPAINT_TELEA)

    # Viewer-left is wearer-right: exact colored Black Rose/cloud embroidery.
    black_rose_logo = extract_logo(
        physical_right, (388, 500, 125, 180), gray_limit=92, saturation_limit=48
    )
    black_rose_overlay = warp_rgba(
        black_rose_logo,
        np.array([[497, 520], [511, 516], [499, 568], [513, 563]]),
        (height, width),
    )
    patched, black_rose_mask = composite_rgba(patched, black_rose_overlay)

    # Viewer-right is wearer-left: exact tonal Love Hurts embroidery.
    love_hurts_logo = extract_logo(physical_left, (330, 525, 205, 155), gray_limit=102)
    love_hurts_overlay = warp_rgba(
        love_hurts_logo,
        np.array([[669, 526], [687, 521], [673, 567], [691, 562]]),
        (height, width),
    )
    patched, love_hurts_mask = composite_rgba(patched, love_hurts_overlay)

    edit_mask = cv2.max(erase_mask, cv2.max(black_rose_mask, love_hurts_mask))
    edit_mask[edit_mask > 0] = 255

    # Enforce the product-fidelity invariant at byte level.
    outside = edit_mask == 0
    patched[outside] = target[outside]
    changed = np.any(patched != target, axis=2)
    outside_changed = int(np.count_nonzero(changed & outside))
    inside_changed = int(np.count_nonzero(changed & (edit_mask > 0)))
    if outside_changed != 0 or inside_changed == 0:
        raise RuntimeError(
            f"localized patch invariant failed: outside={outside_changed}, inside={inside_changed}"
        )

    rgb_path = OUT_DIR / "br-commerce-2-dual-cast-br007-corrected-rgb-v6.png"
    mask_path = OUT_DIR / "br007-side-panels-explicit-mask-v6.png"
    alpha_path = OUT_DIR / "br-commerce-2-dual-cast-br007-corrected-alpha-v6.png"
    scene_path = REVIEW_DIR / "BR-COMMERCE-2-exact-br007-font-statue-review-v6.png"
    proof_path = REVIEW_DIR / "BR-COMMERCE-2-v6-mask-alpha-proof.png"
    metrics_path = OUT_DIR / "br-commerce-2-v6-pixel-metrics.json"

    cv2.imwrite(str(rgb_path), patched)
    cv2.imwrite(str(mask_path), edit_mask)

    alpha = derive_subject_alpha(patched)
    rgba = cv2.cvtColor(patched, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha
    cv2.imwrite(str(alpha_path), rgba)

    background = read_bgr(BACKGROUND)
    statue = read_bgr(STATUE, cv2.IMREAD_UNCHANGED)
    if statue.shape[2] != 4:
        raise RuntimeError("font statue must have an alpha channel")
    scene = alpha_over(background, statue, x=1030, y=95)
    contact_shadow = np.zeros((height, width), dtype=np.uint8)
    cv2.ellipse(contact_shadow, (590, 864), (102, 17), 0, 0, 360, 185, -1)
    cv2.ellipse(contact_shadow, (958, 856), (88, 14), 0, 0, 360, 165, -1)
    contact_shadow = cv2.GaussianBlur(contact_shadow, (0, 0), 10.0)
    shadow_factor = 1.0 - (contact_shadow.astype(np.float32) / 255.0)[:, :, None] * 0.68
    scene = np.clip(scene.astype(np.float32) * shadow_factor, 0, 255).astype(np.uint8)
    scene = alpha_over(scene, rgba, x=0, y=0)
    cv2.imwrite(str(scene_path), scene)

    proof = scene.copy()
    magenta = np.zeros_like(proof)
    magenta[:, :, :] = (255, 0, 255)
    transparent = alpha == 0
    proof[transparent] = np.rint(proof[transparent] * 0.45 + magenta[transparent] * 0.55).astype(
        np.uint8
    )
    contours, _ = cv2.findContours(edit_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(proof, contours, -1, (0, 255, 255), 2)
    cv2.imwrite(str(proof_path), proof)

    metrics = {
        "schema_version": "skyyrose.localized-product-patch-metrics.v1",
        "target": {"path": str(TARGET.relative_to(REPO)), "sha256": sha256(TARGET)},
        "output_rgb": {"path": str(rgb_path.relative_to(REPO)), "sha256": sha256(rgb_path)},
        "output_alpha": {"path": str(alpha_path.relative_to(REPO)), "sha256": sha256(alpha_path)},
        "explicit_mask": {"path": str(mask_path.relative_to(REPO)), "sha256": sha256(mask_path)},
        "review_scene": {"path": str(scene_path.relative_to(REPO)), "sha256": sha256(scene_path)},
        "verification": {
            "outside_mask_changed_pixels": outside_changed,
            "inside_mask_changed_pixels": inside_changed,
            "mask_pixels": int(np.count_nonzero(edit_mask)),
            "alpha_zero_pixels": int(np.count_nonzero(alpha == 0)),
            "alpha_opaque_pixels": int(np.count_nonzero(alpha == 255)),
            "alpha_partial_pixels": int(np.count_nonzero((alpha > 0) & (alpha < 255))),
        },
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    if args.write_receipt:
        print(json.dumps(metrics, indent=2))
    else:
        print(
            f"built {scene_path.relative_to(REPO)}; outside-mask changed={outside_changed}; "
            f"inside-mask changed={inside_changed}"
        )


if __name__ == "__main__":
    main()
