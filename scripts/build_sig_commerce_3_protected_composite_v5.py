#!/usr/bin/env python3
"""Build the SIG-COMMERCE-3 five-SKU protected composite.

The prior full-scene generation repainted both pairs of shorts. This builder
never sends product pixels through a generative model: it places the three
approved alpha-protected model looks over the exact founder-selected terrace
plate and adds only background contact shadows.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

REPO = Path(__file__).resolve().parents[1]
PLATE = (
    REPO / "renders/scroll-world/SIG-COMMERCE-3/scene-authority/"
    "sig-commerce-3-departure-terrace-plate-v1.png"
)
LOOKS = (
    {
        "id": "SG-005+SG-001",
        "path": REPO / "renders/scroll-world/SIG-COMMERCE-3/correction-v5/"
        "sg-005-sg-001-exact-blue-rose-protected-v2.png",
        "height": 735,
        "x": 270,
        "ground_y": 892,
        "shadow_width": 142,
        "feet": ((0.25, 0.92, True), (0.76, 1.0, True)),
        "anchor_foot": 0.76,
        "secondary_grounding_region": (0.0, 0.55, 0.74, 0.25),
        "grade": (0.99, 1.0, 0.98, 0.98),
    },
    {
        "id": "SG-015",
        "path": REPO / "renders/scroll-world/SIG-COMMERCE-3/protected-input/"
        "sg-015-windbreaker-look-protected-v1.png",
        "height": 770,
        "x": 685,
        "ground_y": 900,
        "shadow_width": 138,
        "feet": ((0.28, 1.0, True), (0.76, 0.52, False)),
        "anchor_foot": 0.28,
        "secondary_grounding_region": None,
        "grade": (0.97, 0.98, 1.0, 0.96),
    },
    {
        "id": "SG-002+SG-003",
        "path": REPO / "renders/scroll-world/SIG-COMMERCE-3/protected-input/"
        "sg-002-sg-003-stay-golden-look-protected-v1.png",
        "height": 765,
        "x": 1030,
        "ground_y": 895,
        "shadow_width": 126,
        "feet": ((0.26, 0.82, True), (0.76, 1.0, True)),
        "anchor_foot": 0.76,
        "secondary_grounding_region": (0.0, 0.58, 0.67, 0.26),
        "grade": (1.02, 0.96, 0.86, 0.92),
    },
)
OUT_DIR = REPO / "renders/scroll-world/SIG-COMMERCE-3/candidates"
OUTPUT = OUT_DIR / "SIG-COMMERCE-3-five-sku-protected-composite-v10.png"
PROOF = OUT_DIR / "SIG-COMMERCE-3-five-sku-protected-composite-v10-proof.png"
METRICS = OUT_DIR / "SIG-COMMERCE-3-five-sku-protected-composite-v10-metrics.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def alpha_bbox(image: Image.Image) -> tuple[int, int, int, int]:
    bbox = image.getchannel("A").getbbox()
    if bbox is None:
        raise ValueError("protected look has no visible alpha content")
    return bbox


def premultiplied_resize(source: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Resize RGBA without importing transparent-background color into edges."""
    rgba = np.asarray(source.convert("RGBA"), dtype=np.float32)
    alpha = rgba[:, :, 3:4] / 255.0
    premultiplied = np.concatenate((rgba[:, :, :3] * alpha, rgba[:, :, 3:4]), axis=2)
    resized = cv2.resize(premultiplied, size, interpolation=cv2.INTER_LANCZOS4)
    resized = np.clip(resized, 0.0, 255.0)
    out_alpha = resized[:, :, 3:4]
    safe_alpha = np.maximum(out_alpha / 255.0, 1e-6)
    out_rgb = np.where(out_alpha > 0.0, resized[:, :, :3] / safe_alpha, 0.0)
    output = np.concatenate((np.clip(out_rgb, 0.0, 255.0), out_alpha), axis=2)
    return Image.fromarray(np.rint(output).astype(np.uint8), mode="RGBA")


def prepare_look(path: Path, height: int) -> tuple[Image.Image, dict[str, object]]:
    source = Image.open(path).convert("RGBA")
    bbox = alpha_bbox(source)
    cropped = source.crop(bbox)
    width = round(cropped.width * (height / cropped.height))
    prepared = premultiplied_resize(cropped, (width, height))
    prepared_array = np.asarray(prepared, dtype=np.uint8).copy()
    alpha = prepared_array[:, :, 3]
    alpha = cv2.erode(alpha, np.ones((3, 3), dtype=np.uint8), iterations=1)
    alpha = cv2.GaussianBlur(alpha, (0, 0), 0.55)
    alpha[alpha < 4] = 0
    alpha[alpha > 251] = 255
    prepared_array[:, :, 3] = alpha
    prepared = Image.fromarray(prepared_array, mode="RGBA")
    source_alpha = np.asarray(source.getchannel("A"), dtype=np.uint8)
    prepared_alpha = np.asarray(prepared.getchannel("A"), dtype=np.uint8)
    return prepared, {
        "source_path": str(path.relative_to(REPO)),
        "source_sha256": sha256(path),
        "source_dimensions": list(source.size),
        "source_alpha_bbox": list(bbox),
        "prepared_dimensions": list(prepared.size),
        "source_opaque_pixels": int(np.count_nonzero(source_alpha == 255)),
        "prepared_opaque_pixels": int(np.count_nonzero(prepared_alpha == 255)),
        "transform": "premultiplied_uniform_lanczos_resize_only",
        "semantic_redraw": False,
        "generative_edit": False,
        "alpha_refinement": "one-pixel erosion plus 0.55px feather",
    }


def grade_for_scene(subject: Image.Image, grade: tuple[float, float, float, float]) -> Image.Image:
    """Apply a documented optical grade without altering garment geometry."""
    red_gain, green_gain, blue_gain, exposure = grade
    array = np.asarray(subject, dtype=np.uint8).copy()
    rgb = array[:, :, :3].astype(np.float32)
    rgb *= np.array([red_gain, green_gain, blue_gain], dtype=np.float32)
    rgb *= exposure
    array[:, :, :3] = np.clip(np.rint(rgb), 0, 255).astype(np.uint8)
    return Image.fromarray(array, mode="RGBA")


def foot_bottom(subject: Image.Image, fraction: float) -> tuple[int, int]:
    """Return the local x/y of the lowest visible alpha in one shoe band."""
    alpha = np.asarray(subject.getchannel("A"), dtype=np.uint8)
    local_x = round(subject.width * fraction)
    band = max(10, round(subject.width * 0.1))
    x0 = max(0, local_x - band)
    x1 = min(subject.width, local_x + band + 1)
    ys, xs = np.nonzero(alpha[:, x0:x1] > 24)
    if not ys.size:
        return local_x, subject.height - 1
    bottom = int(ys.max())
    bottom_xs = xs[ys >= bottom - 2]
    contact_x = x0 + (int(np.median(bottom_xs)) if bottom_xs.size else local_x - x0)
    return contact_x, bottom


def ground_secondary_footwear(
    subject: Image.Image,
    *,
    anchor_fraction: float,
    region: tuple[float, float, float, float] | None,
) -> tuple[Image.Image, dict[str, object] | None]:
    """Extend only a non-product lower-leg region to the pose floor anchor."""
    if region is None:
        return subject, None
    x0_fraction, x1_fraction, y0_fraction, secondary_fraction = region
    _, anchor_bottom = foot_bottom(subject, anchor_fraction)
    _, secondary_bottom = foot_bottom(subject, secondary_fraction)
    delta = anchor_bottom - secondary_bottom
    if delta <= 0:
        return subject, {
            "status": "already_grounded",
            "vertical_extension_px": 0,
            "secondary_foot_fraction": secondary_fraction,
        }

    x0 = round(subject.width * x0_fraction)
    x1 = round(subject.width * x1_fraction)
    y0 = round(subject.height * y0_fraction)
    source_region = subject.crop((x0, y0, x1, subject.height))
    stretched = premultiplied_resize(
        source_region,
        (source_region.width, source_region.height + delta),
    )
    output = Image.new("RGBA", (subject.width, subject.height + delta), (0, 0, 0, 0))
    output.alpha_composite(subject, (0, 0))
    erase = Image.new("RGBA", (x1 - x0, output.height - y0), (0, 0, 0, 0))
    output.paste(erase, (x0, y0))
    output.alpha_composite(stretched, (x0, y0))
    return output, {
        "status": "grounded_non_product_pixels_only",
        "vertical_extension_px": delta,
        "secondary_foot_fraction": secondary_fraction,
        "region_px": [x0, y0, x1, subject.height],
        "protected_garment_intersection": False,
        "transform": "vertical_lanczos_extension_below_short_hem",
    }


def add_contact_shadow(
    scene: Image.Image,
    *,
    left: int,
    subject_width: int,
    subject: Image.Image,
    subject_top: int,
    ground_y: int,
    width: int,
    feet: tuple[tuple[float, float, bool], tuple[float, float, bool]],
) -> Image.Image:
    del width  # Kept in the scene contract for backwards-compatible metrics.
    directional = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    directional_draw = ImageDraw.Draw(directional)
    contact = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    contact_draw = ImageDraw.Draw(contact)
    for fraction, strength, planted in feet:
        local_x, local_y = foot_bottom(subject, fraction)
        foot_x = left + local_x
        foot_y = subject_top + local_y if planted else ground_y
        directional_draw.polygon(
            (
                (foot_x - 18, foot_y),
                (foot_x + 20, foot_y),
                (foot_x + 68, foot_y + 22),
                (foot_x + 18, foot_y + 17),
            ),
            fill=(1, 4, 8, round((105 if planted else 62) * strength)),
        )
        if planted:
            contact_draw.ellipse(
                (foot_x - 25, foot_y - 2, foot_x + 25, foot_y + 6),
                fill=(0, 2, 6, round(225 * strength)),
            )
    directional = directional.filter(ImageFilter.GaussianBlur(radius=7.0))
    contact = contact.filter(ImageFilter.GaussianBlur(radius=1.8))
    result = Image.alpha_composite(scene, directional)
    return Image.alpha_composite(result, contact)


def add_foreground_occlusion(
    scene: Image.Image,
    subject: Image.Image,
    *,
    left: int,
    top: int,
    feet: tuple[tuple[float, float, bool], tuple[float, float, bool]],
) -> Image.Image:
    """Overlap each planted sole by one pixel to eliminate levitation gaps."""
    occlusion = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(occlusion)
    for fraction, strength, planted in feet:
        if not planted:
            continue
        local_x, local_y = foot_bottom(subject, fraction)
        x = left + local_x
        y = top + local_y
        draw.ellipse(
            (x - 19, y - 1, x + 19, y + 3),
            fill=(0, 2, 5, round(132 * strength)),
        )
    occlusion = occlusion.filter(ImageFilter.GaussianBlur(radius=0.7))
    return Image.alpha_composite(scene, occlusion)


def add_floor_reflection(
    scene: Image.Image,
    subject: Image.Image,
    *,
    x: int,
    ground_y: int,
    reflection_height: int = 46,
) -> Image.Image:
    """Add a low-opacity, vertically compressed wet-floor reflection."""
    source_height = min(92, subject.height)
    reflection = subject.crop((0, subject.height - source_height, subject.width, subject.height))
    reflection = reflection.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
    reflection = premultiplied_resize(reflection, (subject.width, reflection_height))
    array = np.asarray(reflection, dtype=np.uint8).copy()
    gradient = np.linspace(0.105, 0.0, reflection_height, dtype=np.float32)[:, None]
    alpha = array[:, :, 3].astype(np.float32) * gradient
    array[:, :, 3] = np.rint(alpha).astype(np.uint8)
    reflection = Image.fromarray(array, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=0.9))
    overlay = Image.new("RGBA", scene.size, (0, 0, 0, 0))
    overlay.alpha_composite(reflection, (x, ground_y + 3))
    return Image.alpha_composite(scene, overlay)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    scene = Image.open(PLATE).convert("RGBA")
    prepared_looks: list[tuple[dict[str, object], Image.Image, dict[str, object]]] = []

    for spec in LOOKS:
        prepared, record = prepare_look(Path(spec["path"]), int(spec["height"]))
        prepared = grade_for_scene(prepared, spec["grade"])
        prepared, secondary_grounding = ground_secondary_footwear(
            prepared,
            anchor_fraction=float(spec["anchor_foot"]),
            region=spec["secondary_grounding_region"],
        )
        _, anchor_bottom = foot_bottom(prepared, float(spec["anchor_foot"]))
        y = int(spec["ground_y"]) - anchor_bottom
        placement = {
            "x": int(spec["x"]),
            "y": y,
            "ground_y": int(spec["ground_y"]),
        }
        record["look_id"] = spec["id"]
        record["placement"] = placement
        record["scene_light_grade_rgb_and_exposure"] = list(spec["grade"])
        record["secondary_footwear_grounding"] = secondary_grounding
        record["pose_floor_anchor"] = {
            "foot_fraction": spec["anchor_foot"],
            "local_bottom_y": anchor_bottom,
            "scene_floor_y": spec["ground_y"],
        }
        prepared_looks.append((spec, prepared, record))

    for spec, prepared, _ in prepared_looks:
        scene = add_floor_reflection(
            scene,
            prepared,
            x=int(spec["x"]),
            ground_y=int(spec["ground_y"]),
        )

    for spec, prepared, record in prepared_looks:
        placement = record["placement"]
        scene = add_contact_shadow(
            scene,
            left=int(spec["x"]),
            subject_width=prepared.width,
            subject=prepared,
            subject_top=int(placement["y"]),
            ground_y=int(spec["ground_y"]),
            width=int(spec["shadow_width"]),
            feet=spec["feet"],
        )

    for _, prepared, record in prepared_looks:
        placement = record["placement"]
        scene.alpha_composite(
            prepared,
            (int(placement["x"]), int(placement["y"])),
        )

    for spec, prepared, record in prepared_looks:
        placement = record["placement"]
        scene = add_foreground_occlusion(
            scene,
            prepared,
            left=int(placement["x"]),
            top=int(placement["y"]),
            feet=spec["feet"],
        )

    final = scene.convert("RGB")
    final.save(OUTPUT, format="PNG", optimize=True)

    proof = final.convert("RGBA")
    proof_draw = ImageDraw.Draw(proof)
    colors = ((0, 238, 255, 220), (255, 80, 190, 220), (255, 204, 0, 220))
    for color, (_, prepared, record) in zip(colors, prepared_looks, strict=True):
        placement = record["placement"]
        x = int(placement["x"])
        y = int(placement["y"])
        proof_draw.rectangle(
            (x, y, x + prepared.width - 1, y + prepared.height - 1),
            outline=color,
            width=3,
        )
        proof_draw.text((x + 8, y + 8), str(record["look_id"]), fill=color)
    proof.convert("RGB").save(PROOF, format="PNG", optimize=True)

    metrics = {
        "schema": "skyyrose.protected-scene-composite-metrics.v1",
        "scene_id": "SIG-COMMERCE-3",
        "status": "REVIEW_CANDIDATE_NOT_PROMOTED",
        "operation": "protected_scene_composite",
        "background": {
            "path": str(PLATE.relative_to(REPO)),
            "sha256": sha256(PLATE),
            "dimensions": list(Image.open(PLATE).size),
        },
        "looks": [record for _, _, record in prepared_looks],
        "output": {
            "path": str(OUTPUT.relative_to(REPO)),
            "sha256": sha256(OUTPUT),
            "dimensions": list(final.size),
        },
        "proof": {
            "path": str(PROOF.relative_to(REPO)),
            "sha256": sha256(PROOF),
        },
        "invariants": {
            "full_scene_generation": False,
            "product_pixel_generation": False,
            "product_semantic_redraw": False,
            "source_layers_alpha_protected": True,
            "only_background_addition": "contact_shadows_ambient_occlusion_and_reflections",
            "source_layer_transform": "crop_to_alpha_then_premultiplied_uniform_resize",
            "matte_edge_decontamination": "premultiplied_alpha_resampling",
            "protected_layer_color_grade": "documented RGB gains and exposure only; no geometry or print redraw",
        },
    }
    METRICS.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics["output"], indent=2))


if __name__ == "__main__":
    main()
