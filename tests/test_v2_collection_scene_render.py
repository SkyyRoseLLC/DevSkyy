"""Exercise the real scene templates using the explicitly offline preview adapter."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def render_php(code):
    php = shutil.which("php")
    assert php, "PHP is required for the V2 scene rendering regression"
    result = subprocess.run(
        [php, "-d", "output_buffering=4096", "-r", code],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "Fatal error" not in result.stdout
    return result.stdout


@pytest.mark.parametrize(
    "collection,prefix", [("black-rose", "br"), ("love-hurts", "lh"), ("signature", "sig")]
)
def test_every_approved_chapter_remains_rendered(collection, prefix):
    output = render_php(
        f'$_GET["route"]={json.dumps(collection)}; require "tools/v2-theme-preview.php";'
    )
    for chapter in range(1, 4):
        assert output.count(f'data-scene-id="{prefix}-commerce-{chapter}"') == 1
    assert "data-recovery-track" in output


@pytest.mark.parametrize(
    "mutation",
    [
        '$scene["image"] = "generated-candidates/unapproved.webp";',
        '$scene["product_bindings"] = array("wrong-sku");',
        '$scene["hero_composition"]["variants"] = array(array("asset" => "candidate.webp"));',
    ],
)
def test_direct_template_rejects_unapproved_scene_bindings(mutation):
    output = render_php(
        '$_GET["route"]="black-rose"; ob_start(); require "tools/v2-theme-preview.php"; ob_end_clean();'
        '$scene=skyyrose2_collection_commerce_scenes("black-rose")[0];'
        + mutation
        + '$args=array("scene"=>$scene,"collection"=>"black-rose","index"=>0);'
        'require "wordpress-theme/skyyrose-flagship-2/template-parts/commerce/hero-composed-scene.php";'
    )
    assert output == ""
