"""Regression coverage for render-pipeline credential precedence."""

from __future__ import annotations

import ast
from pathlib import Path

from oai_render import config

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "scripts" / "oai_render" / "config.py"


def test_render_model_uses_the_registry_supported_identifier() -> None:
    """A made-up dated snapshot must not bypass the fail-closed registry."""
    assert config.MODEL == "gpt-image-2"


def test_render_config_does_not_import_global_config_loader() -> None:
    """The global package can load unrelated legacy env files on import."""
    tree = ast.parse(CONFIG_PATH.read_text(encoding="utf-8"))
    imports = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }
    assert "config.load_env" not in imports


def test_render_env_local_key_overrides_base_env(monkeypatch, tmp_path: Path) -> None:
    """A project-local renderer key wins over the low-priority base env file."""
    monkeypatch.delenv(config.API_KEY_ENV, raising=False)
    (tmp_path / ".env").write_text(f"{config.API_KEY_ENV}=base-key\n", encoding="utf-8")
    (tmp_path / ".env.local").write_text(f"{config.API_KEY_ENV}=local-key\n", encoding="utf-8")

    config.load_render_env(tmp_path)

    assert config.get_api_key() == "local-key"


def test_render_env_preserves_caller_key(monkeypatch, tmp_path: Path) -> None:
    """A process-provided key remains higher priority than any local file."""
    monkeypatch.setenv(config.API_KEY_ENV, "caller-key")
    (tmp_path / ".env.local").write_text(f"{config.API_KEY_ENV}=local-key\n", encoding="utf-8")

    config.load_render_env(tmp_path)

    assert config.get_api_key() == "caller-key"
