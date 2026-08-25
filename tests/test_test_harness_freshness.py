"""Regression coverage for the test runner's freshness guarantees."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_default_dev_environment_collects_default_suite_dependencies() -> None:
    """Dependencies imported by default-collected tests must be in the dev extra."""
    pyproject = (ROOT / "pyproject.toml").read_text()

    for dependency in ("pytest-timeout", "pygltflib", "pyotp", "rich", "typer"):
        assert f'"{dependency}' in pyproject

    assert "timeout = 10" in pyproject


def test_retry_gate_reexecutes_the_current_suite_without_cached_failures() -> None:
    """A retry must not revive stale ``--last-failed`` entries from old runs."""
    hook = (ROOT / ".codex/hooks/stop-test-gate.sh").read_text()

    assert "--last-failed" not in hook
    assert "--cache-clear" in hook
    assert "run_current_suite" in hook


def test_ci_does_not_restore_pytest_last_failed_state() -> None:
    """Pytest's last-failed cache is local retry state, not a CI build artifact."""
    workflow = (ROOT / ".github/workflows/ci.yml").read_text()

    assert "Cache pytest" not in workflow
    assert "path: .pytest_cache" not in workflow


def test_default_suite_excludes_the_real_agent_smoke_path() -> None:
    """The baseline suite must not bootstrap a real agent runtime."""
    smoke_test = (ROOT / "tests/aos/test_smoke_real_agent.py").read_text()

    assert "@pytest.mark.integration" in smoke_test
