"""Regression tests for the frontend catalog deployment replica guard."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_validator():
    module_name = "validate_catalog_consistency_replica_guard"
    spec = importlib.util.spec_from_file_location(
        module_name, _REPO_ROOT / "scripts" / "validate_catalog_consistency.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


validator = _load_validator()


def test_catalog_replica_matches_real_catalog():
    result = validator.check_catalog_replica()
    assert result.passed, [result.message, *result.details]


def test_catalog_replica_rejects_drift(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical.csv"
    replica = tmp_path / "replica.csv"
    canonical.write_bytes(b"sku,branding_spec\nbr-012,80 percent torso width\n")
    replica.write_bytes(b"sku,branding_spec\nbr-012,arched across chest\n")
    monkeypatch.setattr(validator, "_CATALOG_CSV", canonical)
    monkeypatch.setattr(validator, "_FRONTEND_CATALOG_REPLICA", replica)

    result = validator.check_catalog_replica()

    assert not result.passed
    assert "differs" in result.message


def test_catalog_replica_rejects_missing_replica(tmp_path, monkeypatch):
    canonical = tmp_path / "canonical.csv"
    canonical.write_bytes(b"sku,branding_spec\nbr-012,80 percent torso width\n")
    monkeypatch.setattr(validator, "_CATALOG_CSV", canonical)
    monkeypatch.setattr(validator, "_FRONTEND_CATALOG_REPLICA", tmp_path / "missing.csv")

    result = validator.check_catalog_replica()

    assert not result.passed
    assert "not found" in result.message
