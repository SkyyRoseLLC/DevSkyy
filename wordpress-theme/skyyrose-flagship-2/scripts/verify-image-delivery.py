#!/usr/bin/env python3
"""Verify V2 image delivery contracts without inspecting live WooCommerce."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from PIL import Image


THEME = Path(__file__).resolve().parents[1]
OPTIMIZATION = THEME / "data/image-optimization.json"
PRODUCT_MEDIA = THEME / "data/opening-product-media.json"


def fail(message: str) -> None:
    print(f"FAIL image delivery: {message}", file=sys.stderr)
    raise SystemExit(1)


def require_file(relative: str) -> Path:
    path = THEME / relative
    if not path.is_file() or path.stat().st_size == 0:
        fail(f"missing image asset: {relative}")
    return path


def require_budget(relative: str, budget: int) -> None:
    path = require_file(relative)
    if path.stat().st_size > budget:
        fail(f"{relative} is {path.stat().st_size} bytes; budget is {budget}")


def check_alpha(relative: str) -> None:
    path = require_file(relative)
    try:
        with Image.open(path) as image:
            if "A" not in image.getbands():
                fail(f"transparent brand asset has no alpha channel: {relative}")
            alpha = image.getchannel("A")
            if alpha.getextrema()[1] == 0:
                fail(f"transparent brand asset is fully transparent: {relative}")
    except OSError as error:
        fail(f"cannot inspect {relative}: {error}")


def check_registered_template_images() -> None:
    image_pattern = re.compile(
        r"(?:skyyrose2_sot_asset_uri\(\s*['\"]([^'\"]+)['\"]|SKYYROSE2_URI\s*\.\s*['\"](/assets/sot/[^'\"]+)['\"])",
    )
    extensions = (".avif", ".gif", ".jpeg", ".jpg", ".png", ".webp")
    for php_file in THEME.rglob("*.php"):
        if "/vendor/" in php_file.as_posix():
            continue
        for match in image_pattern.finditer(php_file.read_text(encoding="utf-8")):
            relative = match.group(1) or match.group(2).removeprefix("/assets/sot/")
            if relative.lower().endswith(extensions):
                require_file(f"assets/sot/{relative}")


def main() -> int:
    optimization = json.loads(OPTIMIZATION.read_text(encoding="utf-8"))
    product_media = json.loads(PRODUCT_MEDIA.read_text(encoding="utf-8"))
    policy = optimization["policy"]

    for base in optimization["hero_bases"]:
        for width in (640, 1024, 1440):
            require_budget(
                f"assets/sot/images/hero/responsive/{base}-{width}w.webp",
                int(policy["max_served_hero_bytes"]),
            )

    for editorial in optimization.get("editorial_derivative_sets", []):
        for width in editorial["widths"]:
            require_budget(
                f"{editorial['derivative_root']}/{editorial['basename']}-{width}w.webp",
                int(policy["max_served_editorial_bytes"]),
            )

    for record in product_media["products"].values():
        roles = [view.get("role") for view in record.get("views", [])]
        if record.get("status") == "blocked":
            if roles and roles[0] == "on_model_front":
                fail("blocked product media cannot claim an on-model front")
        elif not roles or roles[0] != "on_model_front":
            fail("every product media record must lead with on_model_front")
        for view in record.get("views", []):
            require_budget(view["derivative"], int(product_media["delivery"]["max_bytes"]))

    for relative in optimization.get("transparent_brand_assets", []):
        check_alpha(relative)

    check_registered_template_images()
    print("PASS image delivery: responsive, registered, budgeted, and alpha checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
