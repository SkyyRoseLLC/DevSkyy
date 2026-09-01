#!/usr/bin/env python3
"""Regression tests for Fashion Theme Team distribution boundaries."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

SCRIPT_PATH = Path(__file__).with_name("sync-team-distributions.py")
SPEC = importlib.util.spec_from_file_location("fashion_sync_team_distributions", SCRIPT_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - importlib contract guard
    raise RuntimeError(f"cannot import distribution synchronizer from {SCRIPT_PATH}")
SYNC = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = SYNC
SPEC.loader.exec_module(SYNC)


class PackageBoundaryTests(unittest.TestCase):
    def test_machine_local_caches_and_environments_are_excluded(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package_root = Path(temporary_directory)
            expected = {
                Path(".codex-plugin/plugin.json"),
                Path("runtime/director.py"),
            }
            for expected_relative in expected:
                path = package_root / expected_relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("source\n", encoding="utf-8")

            for local_relative in (
                ".mypy_cache/3.14/module.meta.json",
                ".pytest_cache/state",
                ".ruff_cache/state",
                ".venv/bin/python",
                ".venv-format/bin/black",
                "node_modules/prettier/index.js",
                "runtime/__pycache__/director.pyc",
            ):
                path = package_root / local_relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("local\n", encoding="utf-8")

            observed = SYNC.package_files(package_root)

        self.assertEqual(observed, expected)

    def test_symlinked_package_content_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            package_root = Path(temporary_directory)
            target = package_root / "target.py"
            target.write_text("source\n", encoding="utf-8")
            (package_root / "linked.py").symlink_to(target)

            with self.assertRaisesRegex(RuntimeError, "symlinked package content"):
                SYNC.package_files(package_root)


if __name__ == "__main__":
    unittest.main()
