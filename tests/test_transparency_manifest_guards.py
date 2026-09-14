"""Exercise the actual shell guards without invoking unrelated certification gates."""

import json
import re
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "wordpress-theme/skyyrose-flagship-2/scripts"


@pytest.mark.parametrize(
    ("script", "key"),
    [
        ("verify-marketplace.sh", "transparent_brand_assets"),
        ("verify-v2-candidate.sh", "transparent_assets"),
    ],
)
@pytest.mark.parametrize(
    ("value", "succeeds"),
    [
        (["assets/logo.webp"], True),
        ([], False),
        (None, False),
        ("__missing__", False),
        ({}, False),
        ("assets/logo.webp", False),
        ([""], False),
        ([123], False),
    ],
)
def test_manifest_guard(tmp_path: Path, script: str, key: str, value, succeeds: bool):
    source = (SCRIPTS / script).read_text()
    guard = re.search(rf"^if ! jq -e '\.{key} .*?^fi$", source, re.MULTILINE | re.DOTALL)
    assert guard, f"{script} must guard its manifest before iterating assets"
    assert guard.start() < source.index(f"jq -r '.{key}[]'")
    data = tmp_path / "data"
    data.mkdir()
    manifest = data / "image-optimization.json"
    manifest.write_text(json.dumps({} if value == "__missing__" else {key: value}))
    result = subprocess.run(
        [
            "bash",
            "-c",
            'cd -- "$1" || exit\ntransparency_manifest="$2"\n' + guard.group(),
            "guard",
            str(tmp_path),
            str(manifest),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert (result.returncode == 0) is succeeds, result.stderr
    if not succeeds:
        assert "nonempty array of asset paths" in result.stderr
