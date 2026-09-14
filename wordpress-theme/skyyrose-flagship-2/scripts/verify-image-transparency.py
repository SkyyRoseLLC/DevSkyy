#!/usr/bin/env python3
"""Fail when a required compositing asset has no transparent pixels."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

try:
    from PIL import Image, UnidentifiedImageError
except ImportError as exc:
    print(
        f"verify-image-transparency requires Pillow (pip install Pillow): {exc}",
        file=sys.stderr,
    )
    raise SystemExit(1)


LOGGER = logging.getLogger(__name__)


def has_transparency(path: Path) -> bool:
    """Require transparency in every displayed (composited) animation frame."""
    with Image.open(path) as image:
        for frame in range(getattr(image, "n_frames", 1)):
            # Pillow seeks decoded display frames, including animation blending.
            image.seek(frame)
            rgba = image.convert("RGBA")
            alpha_minimum, _ = rgba.getchannel("A").getextrema()
            if alpha_minimum == 255:
                return False
        return True


def main(paths: list[str]) -> int:
    if not paths:
        print(
            "verify-image-transparency: no paths supplied — pass at least one asset path",
            file=sys.stderr,
        )
        return 1
    for raw_path in paths:
        path = Path(raw_path)
        try:
            if not has_transparency(path):
                print(
                    f"Brand asset has a frame without transparent pixels: {path}", file=sys.stderr
                )
                return 1
        except (OSError, UnidentifiedImageError):
            LOGGER.exception("Unable to inspect brand asset %s", path)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
