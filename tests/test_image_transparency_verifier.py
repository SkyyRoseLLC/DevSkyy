"""Exercise static and composited animated assets through the release verifier."""

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "wordpress-theme/skyyrose-flagship-2/scripts/verify-image-transparency.py"
)
spec = importlib.util.spec_from_file_location("image_transparency_verifier", SCRIPT)
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def frame(color, alpha):
    image = Image.new("RGBA", (16, 16), color)
    image.putpixel((0, 0), (*color[:3], alpha))
    return image


@pytest.mark.parametrize("alpha,expected", [(0, True), (127, True), (255, False)])
def test_static_alpha(tmp_path, alpha, expected):
    path = tmp_path / "image.webp"
    frame((255, 0, 0, 255), alpha).save(path, lossless=True)
    assert verifier.has_transparency(path) is expected
    assert verifier.main([str(path)]) == (0 if expected else 1)


@pytest.mark.parametrize(
    "alphas,expected", [([0, 0, 0], True), ([0, 255, 0], False), ([255, 0], False)]
)
def test_all_composited_webp_frames(tmp_path, alphas, expected):
    path = tmp_path / "animation.webp"
    frames = [frame((index * 60, 100, 200, 255), alpha) for index, alpha in enumerate(alphas)]
    frames[0].save(
        path, save_all=True, append_images=frames[1:], duration=100, loop=0, lossless=True
    )
    with Image.open(path) as decoded:
        assert decoded.n_frames == len(alphas)
        observed = []
        for index in range(decoded.n_frames):
            decoded.seek(index)
            observed.append(decoded.convert("RGBA").getchannel("A").getextrema()[0])
        assert observed == alphas  # Verify the fixture really reproduces the later opaque frame.
    assert verifier.has_transparency(path) is expected
    assert verifier.main([str(path)]) == (0 if expected else 1)


@pytest.mark.parametrize("kind", ["missing", "malformed"])
def test_invalid_assets_fail_cli_with_diagnostic(tmp_path, kind):
    path = tmp_path / "invalid.webp"
    if kind == "malformed":
        path.write_bytes(b"not an image")
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(path)], capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "Unable to inspect brand asset" in result.stderr
    assert "Traceback" in result.stderr


def test_empty_arguments_fail():
    result = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "no paths supplied" in result.stderr


def test_every_requested_asset_is_checked(tmp_path):
    transparent, opaque = tmp_path / "transparent.png", tmp_path / "opaque.png"
    frame((0, 0, 0, 255), 0).save(transparent)
    frame((0, 0, 0, 255), 255).save(opaque)
    assert verifier.main([str(transparent), str(opaque)]) == 1
