from __future__ import annotations

import json
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
HOOK_ROOT = REPO_ROOT / ".claude" / "hooks"


def test_all_repository_shell_hooks_parse() -> None:
    failures: list[str] = []
    for hook in sorted(HOOK_ROOT.rglob("*.sh")):
        result = subprocess.run(
            ["bash", "-n", str(hook)],
            text=True,
            capture_output=True,
            check=False,
        )
        if result.returncode:
            failures.append(f"{hook.relative_to(REPO_ROOT)}: {result.stderr.strip()}")
    assert not failures, "\n".join(failures)


def test_active_hook_settings_are_worktree_relative() -> None:
    settings_path = REPO_ROOT / ".claude" / "settings.json"
    settings = json.loads(settings_path.read_text(encoding="utf-8"))
    commands = [
        hook["command"]
        for groups in settings["hooks"].values()
        for group in groups
        for hook in group.get("hooks", [])
    ]
    assert all("/Users/theceo/DevSkyy/" not in command for command in commands)


def test_no_active_hook_references_a_missing_script() -> None:
    settings = json.loads((REPO_ROOT / ".claude" / "settings.json").read_text())
    serialized = json.dumps(settings["hooks"])
    assert "subagent-stop-tidy.sh" not in serialized
