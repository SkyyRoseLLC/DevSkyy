"""Regression coverage for render-pipeline credential precedence."""

from __future__ import annotations

import ast
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "scripts" / "oai_render" / "config.py"


def test_render_config_does_not_import_global_config_loader() -> None:
    """The global package loads legacy .env.hf with override=True on import."""
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "config.load_env" not in imports


def test_render_config_preserves_caller_openai_key() -> None:
    """A caller-provided project key must survive render-config import."""
    sentinel = "sk-proj-render-config-sentinel"
    env = os.environ.copy()
    env["OPENAI_API_KEY"] = sentinel
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; "
                f"sys.path.insert(0, {str(ROOT / 'scripts')!r}); "
                "from oai_render import config; "
                f"raise SystemExit(0 if config.get_api_key() == {sentinel!r} else 1)"
            ),
        ],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
