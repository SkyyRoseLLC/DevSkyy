#!/usr/bin/env python3
"""Regression tests for the source-aware formatting matrix."""

from __future__ import annotations

import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call, patch

SCRIPT_PATH = Path(__file__).with_name("check-formatting.py")
SPEC = importlib.util.spec_from_file_location("fashion_check_formatting", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib contract guard
    raise RuntimeError(f"cannot import formatting matrix from {SCRIPT_PATH}")
MATRIX = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MATRIX)


class ChangedFileScopeTests(unittest.TestCase):
    def test_git_diff_paths_are_relative_to_nested_package(self) -> None:
        with patch.object(
            MATRIX,
            "run_git",
            side_effect=[b"runtime/director.py\0", b"scripts/new_check.py\0"],
        ) as run_git:
            paths = MATRIX.changed_files()

        self.assertEqual(paths, {"runtime/director.py", "scripts/new_check.py"})
        self.assertEqual(
            run_git.call_args_list,
            [
                call(
                    "diff",
                    "--relative",
                    "--name-only",
                    "--diff-filter=ACMRTUXB",
                    "-z",
                    "HEAD",
                ),
                call("ls-files", "--others", "--exclude-standard", "-z"),
            ],
        )

    def test_changed_candidates_reject_paths_outside_package(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            package_root = temporary_root / "package"
            source = package_root / "runtime" / "director.py"
            source.parent.mkdir(parents=True)
            source.write_text("pass\n", encoding="utf-8")
            outside = temporary_root / "outside.py"
            outside.write_text("pass\n", encoding="utf-8")

            with (
                patch.object(MATRIX, "REPO_ROOT", package_root),
                patch.object(MATRIX, "has_git_worktree", return_value=True),
                patch.object(
                    MATRIX,
                    "changed_files",
                    return_value={"runtime/director.py", "../outside.py"},
                ),
            ):
                with self.assertRaisesRegex(MATRIX.MatrixError, "escapes package root"):
                    MATRIX.candidate_files(changed_only=True)

    def test_changed_files_in_real_nested_repository(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            repository = Path(temporary_directory)
            package_root = repository / "plugins" / "fashion-theme-team"
            package_root.mkdir(parents=True)
            for name in ("staged.py", "unstaged.py"):
                (package_root / name).write_text("VALUE = 1\n", encoding="utf-8")

            self.run_git(repository, "init", "--quiet")
            self.run_git(repository, "config", "user.name", "Format Matrix Test")
            self.run_git(repository, "config", "user.email", "format-matrix@example.invalid")
            self.run_git(repository, "add", "plugins/fashion-theme-team")
            self.run_git(repository, "commit", "--quiet", "-m", "baseline")

            (package_root / "staged.py").write_text("VALUE = 2\n", encoding="utf-8")
            self.run_git(repository, "add", "plugins/fashion-theme-team/staged.py")
            (package_root / "unstaged.py").write_text("VALUE = 3\n", encoding="utf-8")
            (package_root / "untracked.py").write_text("VALUE = 4\n", encoding="utf-8")

            with patch.object(MATRIX, "REPO_ROOT", package_root):
                paths = MATRIX.changed_files()

        self.assertEqual(paths, {"staged.py", "unstaged.py", "untracked.py"})

    def test_git_listed_symlink_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            temporary_root = Path(temporary_directory)
            package_root = temporary_root / "package"
            package_root.mkdir()
            outside = temporary_root / "outside.py"
            outside.write_text("pass\n", encoding="utf-8")
            (package_root / "linked.py").symlink_to(outside)

            with (
                patch.object(MATRIX, "REPO_ROOT", package_root),
                patch.object(MATRIX, "has_git_worktree", return_value=True),
                patch.object(MATRIX, "changed_files", return_value={"linked.py"}),
            ):
                with self.assertRaisesRegex(MATRIX.MatrixError, "must not be a symlink"):
                    MATRIX.candidate_files(changed_only=True)

    @staticmethod
    def run_git(repository: Path, *arguments: str) -> None:
        subprocess.run(["git", *arguments], cwd=repository, check=True, stdout=subprocess.DEVNULL)


if __name__ == "__main__":
    unittest.main()
