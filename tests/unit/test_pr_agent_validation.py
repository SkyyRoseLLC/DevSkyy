"""Execute actual workflow shell steps with local Git and a fake GitHub API."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[2]
REPO = "SkyyRoseLLC/DevSkyy"
BRANCH = "fix/fixture"


def workflow(name: str) -> dict:
    return yaml.safe_load((ROOT / ".github/workflows" / name).read_text())


def shell_step(name: str, job: str, step: str) -> str:
    return next(s["run"] for s in workflow(name)["jobs"][job]["steps"] if s["name"] == step)


DISPATCH = shell_step("pr-agent.yml", "pr-agent", "Dispatch core validation for the published fix")
GUARD = shell_step("ci.yml", "validation-head", "Reject stale or mismatched validation requests")


@pytest.fixture
def local_api(tmp_path: Path) -> dict:
    def git(*args: str) -> str:
        # subprocess cwd disables macOS posix_spawn after native ML imports.
        # Let Git change directory inside the child instead.
        return subprocess.check_output(
            ["git", "-C", str(tmp_path), "-c", "core.fsmonitor=false", *args], text=True
        ).strip()

    git("init", "-q", "-b", BRANCH)
    git("config", "user.name", "Offline Fixture")
    git("config", "user.email", "fixture@example.invalid")
    git("config", "core.hooksPath", str(tmp_path / "absent-hooks"))
    (tmp_path / "file.txt").write_text("original")
    git("add", "file.txt")
    git("commit", "-qm", "original")
    original = git("rev-parse", "HEAD")
    (tmp_path / "file.txt").write_text("fixed")
    git("commit", "-qam", "fix")
    head = git("rev-parse", "HEAD")
    metadata = {
        "head": {"repo": {"full_name": REPO}, "ref": BRANCH, "sha": head},
        "state": "open",
        "draft": False,
        "user": {"type": "User"},
    }
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = bin_dir / "gh"
    fake.write_text(f"#!{sys.executable}\n" + """import json, os, sys
from pathlib import Path
args=sys.argv[1:]
if args == ['api', 'repos/SkyyRoseLLC/DevSkyy/pulls/914']:
    print(os.environ['METADATA'])
elif args == ['api', 'repos/SkyyRoseLLC/DevSkyy', '--jq', '.default_branch']:
    print('main')
elif args == ['api', '--method', 'POST', 'repos/SkyyRoseLLC/DevSkyy/actions/workflows/ci.yml/dispatches', '--input', '-']:
    Path(os.environ['DISPATCH_RECEIPT']).write_text(sys.stdin.read())
    sys.exit(int(os.environ.get('API_FAILURE', '0')))
else:
    raise AssertionError(args)
""")
    fake.chmod(0o755)
    env = {
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "GITHUB_REPOSITORY": REPO,
        "GH_TOKEN": "synthetic-offline-token",
        "PR_NUMBER": "914",
        "PR_HEAD_REF": BRANCH,
        "ORIGINAL_HEAD_SHA": original,
        "EXPECTED_HEAD_SHA": head,
        "VALIDATION_PR": "914",
        "EVENT_SHA": head,
        "EVENT_REF": BRANCH,
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary"),
        "DISPATCH_RECEIPT": str(tmp_path / "dispatch.json"),
    }
    return {"cwd": tmp_path, "env": env, "metadata": metadata, "head": head, "original": original}


def run_step(script: str, fixture: dict) -> subprocess.CompletedProcess:
    env = dict(fixture["env"], METADATA=json.dumps(fixture["metadata"]))
    return subprocess.run(
        ["bash", "-c", 'cd -- "$1" || exit\n' + script, "offline-fixture", str(fixture["cwd"])],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )


def test_published_fix_dispatches_exact_head_without_skips_or_deployment(local_api: dict) -> None:
    result = run_step(DISPATCH, local_api)
    assert result.returncode == 0, result.stderr
    payload = json.loads(Path(local_api["env"]["DISPATCH_RECEIPT"]).read_text())
    assert payload == {
        "ref": BRANCH,
        "inputs": {
            "expected_head_sha": local_api["head"],
            "validation_pr": "914",
            "skip_e2e": False,
            "force_deploy": False,
        },
    }
    assert "Validation is pending" in Path(local_api["env"]["GITHUB_STEP_SUMMARY"]).read_text()


def test_no_published_fix_does_not_dispatch(local_api: dict) -> None:
    local_api["metadata"]["head"]["sha"] = local_api["original"]
    assert run_step(DISPATCH, local_api).returncode == 0
    assert not Path(local_api["env"]["DISPATCH_RECEIPT"]).exists()


@pytest.mark.parametrize("script", [DISPATCH, GUARD])
@pytest.mark.parametrize("case", ["fork", "draft", "closed", "bot", "default", "branch", "head"])
def test_untrusted_or_mismatched_head_rejected(local_api: dict, script: str, case: str) -> None:
    metadata = local_api["metadata"]
    if case == "fork":
        metadata["head"]["repo"]["full_name"] = "other/repo"
    elif case == "draft":
        metadata["draft"] = True
    elif case == "closed":
        metadata["state"] = "closed"
    elif case == "bot":
        metadata["user"]["type"] = "Bot"
    elif case == "default":
        metadata["head"]["ref"] = "main"
    elif case == "branch":
        metadata["head"]["ref"] = "other-branch"
    else:
        metadata["head"]["sha"] = "a" * 40
    assert run_step(script, local_api).returncode != 0
    assert not Path(local_api["env"]["DISPATCH_RECEIPT"]).exists()


def test_branch_advancing_between_dispatch_and_run_cannot_validate_old_sha(local_api: dict) -> None:
    local_api["env"]["EVENT_SHA"] = "b" * 40
    local_api["metadata"]["head"]["sha"] = "b" * 40
    assert run_step(GUARD, local_api).returncode != 0


def test_exact_head_guard_accepts_current_pr(local_api: dict) -> None:
    assert run_step(GUARD, local_api).returncode == 0


@pytest.mark.parametrize("field", ["EXPECTED_HEAD_SHA", "VALIDATION_PR"])
@pytest.mark.parametrize("value", ["", "invalid"])
def test_partial_or_invalid_dispatch_identity_rejected(
    local_api: dict, field: str, value: str
) -> None:
    local_api["env"][field] = value
    assert run_step(GUARD, local_api).returncode != 0


def test_regular_ci_without_dispatch_identity_preserves_existing_behavior(local_api: dict) -> None:
    local_api["env"]["EXPECTED_HEAD_SHA"] = ""
    local_api["env"]["VALIDATION_PR"] = ""
    assert run_step(GUARD, local_api).returncode == 0


def test_unrelated_local_head_cannot_dispatch(local_api: dict) -> None:
    local_api["env"]["ORIGINAL_HEAD_SHA"] = "0" * 40
    assert run_step(DISPATCH, local_api).returncode != 0
    assert not Path(local_api["env"]["DISPATCH_RECEIPT"]).exists()


def test_dispatch_api_failure_is_not_reported_as_requested(local_api: dict) -> None:
    local_api["env"]["API_FAILURE"] = "23"
    assert run_step(DISPATCH, local_api).returncode == 23
    assert not Path(local_api["env"]["GITHUB_STEP_SUMMARY"]).exists()


def test_workflow_wiring_preserves_checks_and_immutable_checkout() -> None:
    ci = workflow("ci.yml")
    inputs = ci[True]["workflow_dispatch"]["inputs"]
    assert inputs["skip_e2e"]["default"] is False
    assert inputs["force_deploy"]["default"] is False
    assert "expected_head_sha" in ci["concurrency"]["group"]
    for name in [
        "lint",
        "python-tests",
        "security",
        "frontend-tests",
        "threejs-tests",
        "wordpress-theme",
        "deps-resolve-canary",
    ]:
        assert "validation-head" in ci["jobs"][name]["needs"]
    for job in ci["jobs"].values():
        for step in job.get("steps", []):
            if step.get("uses", "").startswith("actions/checkout@"):
                assert step["with"]["ref"] == "${{ inputs.expected_head_sha || github.sha }}"
    assert ci["jobs"]["deploy-staging"]["if"] is False
    assert ci["jobs"]["deploy-production"]["if"] is False
    agent = workflow("pr-agent.yml")["jobs"]["pr-agent"]
    assert agent["permissions"]["actions"] == "write"
    assert "!cancelled()" in agent["steps"][-1]["if"]
