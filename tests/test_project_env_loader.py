"""Regression tests for the shared project environment loader."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path


def _load_env_module(monkeypatch):
    """Load the module without importing config/__init__.py side effects."""
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        types.SimpleNamespace(load_dotenv=lambda *args, **kwargs: None),
    )
    module_path = Path(__file__).parents[1] / "config" / "load_env.py"
    spec = importlib.util.spec_from_file_location("project_env_loader_test", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_load_project_env_imports_secrets_after_root_env(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """The gitignored secrets file must override non-secret defaults."""
    env_loader = _load_env_module(monkeypatch)
    (tmp_path / ".env").write_text("SERVICE_TOKEN=placeholder\n", encoding="utf-8")
    (tmp_path / ".env.secrets").write_text("SERVICE_TOKEN=secret\n", encoding="utf-8")
    (tmp_path / "gemini").mkdir()

    calls: list[tuple[Path, bool]] = []

    def fake_load_dotenv(path: Path, *, override: bool = False) -> None:
        calls.append((path, override))

    monkeypatch.setattr(env_loader, "_find_project_root", lambda: tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        types.SimpleNamespace(load_dotenv=fake_load_dotenv),
    )

    assert env_loader.load_project_env() == tmp_path
    assert calls == [
        (tmp_path / ".env", False),
        (tmp_path / ".env.secrets", True),
    ]


def test_load_project_env_keeps_provider_file_as_final_override(
    monkeypatch,
    tmp_path: Path,
) -> None:
    """Provider-specific credentials retain the existing highest priority."""
    env_loader = _load_env_module(monkeypatch)
    (tmp_path / ".env.secrets").write_text("GEMINI_API_KEY=shared\n", encoding="utf-8")
    gemini_dir = tmp_path / "gemini"
    gemini_dir.mkdir()
    (gemini_dir / ".env").write_text("GEMINI_API_KEY=provider\n", encoding="utf-8")

    calls: list[tuple[Path, bool]] = []

    def fake_load_dotenv(path: Path, *, override: bool = False) -> None:
        calls.append((path, override))

    monkeypatch.setattr(env_loader, "_find_project_root", lambda: tmp_path)
    monkeypatch.setitem(
        sys.modules,
        "dotenv",
        types.SimpleNamespace(load_dotenv=fake_load_dotenv),
    )

    env_loader.load_project_env()
    assert calls == [
        (tmp_path / ".env.secrets", True),
        (gemini_dir / ".env", True),
    ]
