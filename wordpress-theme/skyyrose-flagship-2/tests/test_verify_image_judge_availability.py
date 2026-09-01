"""Regression tests for the fail-closed GPT review-authority probe."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest

THEME_DIR = Path(__file__).resolve().parents[1]
SCRIPT = THEME_DIR / "scripts/verify-image-judge-availability.py"
SPEC = importlib.util.spec_from_file_location("judge_availability", SCRIPT)
assert SPEC and SPEC.loader
JUDGES = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(JUDGES)


def test_credential_candidates_are_ordered_and_deduplicated(monkeypatch) -> None:
    monkeypatch.setenv("FIRST_TEST_KEY", "same-key")
    monkeypatch.setenv("SECOND_TEST_KEY", "same-key")
    monkeypatch.setenv("THIRD_TEST_KEY", "different-key")

    assert JUDGES.credential_candidates(
        ("FIRST_TEST_KEY", "SECOND_TEST_KEY", "THIRD_TEST_KEY")
    ) == ["same-key", "different-key"]


def test_probe_falls_through_exhausted_key_to_live_key() -> None:
    attempted: list[str] = []

    def callback(credential: str) -> None:
        attempted.append(credential)
        if credential == "exhausted":
            raise RuntimeError("quota exceeded")

    result = JUDGES.probe_credentials("required-model", ["exhausted", "live"], callback)

    assert attempted == ["exhausted", "live"]
    assert result == {
        "model": "required-model",
        "available": True,
        "reason": "live_model_probe_passed",
        "configured_credentials_tried": 2,
    }


def test_probe_fails_closed_when_every_key_is_exhausted() -> None:
    def callback(_credential: str) -> None:
        raise RuntimeError("quota exceeded")

    result = JUDGES.probe_credentials("required-model", ["first", "second"], callback)

    assert result["available"] is False
    assert result["reason"] == "inference_quota_or_credit_exhausted"
    assert result["configured_credentials_tried"] == 2


def test_probe_fails_closed_without_credentials() -> None:
    result = JUDGES.probe_credentials("required-model", [], lambda _credential: None)

    assert result == {
        "model": "required-model",
        "available": False,
        "reason": "credential_not_configured",
        "configured_credentials_tried": 0,
    }


@pytest.mark.parametrize("interrupt", [KeyboardInterrupt(), GeneratorExit()])
def test_probe_propagates_process_control_interrupts(interrupt: BaseException) -> None:
    def callback(_credential: str) -> None:
        raise interrupt

    with pytest.raises(type(interrupt)):
        JUDGES.probe_credentials("required-model", ["first", "second"], callback)


def test_probe_can_fail_closed_on_sdk_system_exit_then_try_fallback() -> None:
    attempted: list[str] = []

    def callback(credential: str) -> None:
        attempted.append(credential)
        if credential == "misconfigured":
            raise SystemExit(2)

    result = JUDGES.probe_credentials("required-model", ["misconfigured", "live"], callback)

    assert attempted == ["misconfigured", "live"]
    assert result["available"] is True


def test_main_probes_only_gpt_review_authority(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    labels: list[str] = []

    def fake_probe(label: str, credentials: list[str], callback) -> dict:
        del credentials, callback
        labels.append(label)
        return {
            "model": label,
            "available": True,
            "reason": "live_model_probe_passed",
            "configured_credentials_tried": 1,
        }

    output = tmp_path / "judges.json"
    monkeypatch.setattr(JUDGES, "OUTPUT", output)
    monkeypatch.setattr(JUDGES, "load_local_environment", lambda: None)
    monkeypatch.setattr(JUDGES, "credential_candidates", lambda _names: ["redacted"])
    monkeypatch.setattr(JUDGES, "probe_credentials", fake_probe)

    assert JUDGES.main() == 0
    assert labels == ["gpt-5.5-pro"]
    assert [item["model"] for item in json.loads(output.read_text())["judges"]] == ["gpt-5.5-pro"]
