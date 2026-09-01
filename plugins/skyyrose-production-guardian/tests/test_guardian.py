from __future__ import annotations

import importlib.util
import json
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[1] / "scripts/guardian.py"
SPEC = importlib.util.spec_from_file_location("guardian", MODULE_PATH)
assert SPEC and SPEC.loader
guardian = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(guardian)


def test_redact_removes_nested_secrets() -> None:
    payload = {"token": "secret", "nested": {"password": "secret", "value": 1}}
    assert guardian.redact(payload) == {
        "token": "[REDACTED]",
        "nested": {"password": "[REDACTED]", "value": 1},
    }


def test_static_gate_rejects_legacy_source(tmp_path: Path, monkeypatch) -> None:
    consumer = tmp_path / "consumer.py"
    consumer.write_text('PATH = "data/collections/black-rose/sot.json"\n')
    settings = tmp_path / ".claude/settings.json"
    settings.parent.mkdir()
    settings.write_text("{}")
    monkeypatch.setitem(guardian.CONFIG, "production_consumers", ["consumer.py"])
    findings = guardian.static_findings(tmp_path)
    assert any(item["rule"] == "no-legacy-production-source" for item in findings)


def test_receipt_is_blocked_for_dirty_tree(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(guardian, "changed_paths", lambda root: ["changed.py"])
    monkeypatch.setattr(
        guardian.subprocess,
        "run",
        lambda *args, **kwargs: type("Result", (), {"stdout": "abc123\n"})(),
    )
    path = guardian.write_receipt(tmp_path, [], [])
    receipt = json.loads(path.read_text())
    assert receipt["status"] == "blocked"
    assert receipt["dirty"] is True
