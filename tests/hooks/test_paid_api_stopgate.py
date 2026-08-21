from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK = REPO_ROOT / ".claude" / "hooks" / "paid-api-stopgate.sh"


def _run(command: str) -> subprocess.CompletedProcess[str]:
    payload = json.dumps({"tool_name": "Bash", "tool_input": {"command": command}})
    return subprocess.run(
        ["bash", str(HOOK)],
        input=payload,
        text=True,
        capture_output=True,
        check=False,
    )


def test_scoped_tmp_recursive_delete_is_not_misclassified() -> None:
    result = _run("rm -rf /tmp/skyyrose-product-audit-123")
    assert result.returncode == 0, result.stderr


def test_scoped_worktree_recursive_delete_is_not_misclassified() -> None:
    result = _run("rm -rf /Users/theceo/DevSkyy-product-card-approved/.cache/audit")
    assert result.returncode == 0, result.stderr


def test_root_and_system_recursive_deletes_still_block() -> None:
    for command in ("rm -rf /", "rm -rf ~", "rm -rf $HOME", "rm -rf /System/Library"):
        result = _run(command)
        assert result.returncode == 2, command
        assert "IRREVERSIBLE data loss" in result.stderr


def test_paid_image_generation_still_blocks() -> None:
    result = _run("python -m renders.generate --sku br-004")
    assert result.returncode == 2
    assert "image generation" in result.stderr


def test_leading_acknowledgement_still_bypasses() -> None:
    result = _run("STOPSHOW_ACK=1 python -m renders.generate --sku br-004")
    assert result.returncode == 0, result.stderr
