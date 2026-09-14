#!/usr/bin/env python3
"""Vendor the approved branded-skill source set as an immutable plugin input."""

from __future__ import annotations

import argparse
import filecmp
import shutil
import tempfile
from pathlib import Path


def copy_sources(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for skill in sorted(source.glob("*/*/SKILL.md")):
        destination = target / skill.relative_to(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(skill, destination)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    sources = sorted(args.source_root.glob("*/*/SKILL.md"))
    if len(sources) != 234:
        raise SystemExit(f"expected 234 source skills, found {len(sources)}")
    if args.check:
        with tempfile.TemporaryDirectory(prefix="ftt-vendor-") as temp:
            expected = Path(temp) / "branded-skills"
            copy_sources(args.source_root, expected)
            comparison = filecmp.dircmp(expected, args.output_root)
            mismatches = (
                list(comparison.left_only)
                + list(comparison.right_only)
                + list(comparison.diff_files)
                + list(comparison.funny_files)
            )
            stack = list(comparison.subdirs.values())
            while stack:
                child = stack.pop()
                mismatches.extend(
                    child.left_only + child.right_only + child.diff_files + child.funny_files
                )
                stack.extend(child.subdirs.values())
            if mismatches:
                raise SystemExit(
                    "vendored source set is stale: " + ", ".join(sorted(set(mismatches)))
                )
        print("PASS: 234 vendored branded-skill sources match the approved source root")
        return 0
    if args.output_root.exists():
        shutil.rmtree(args.output_root)
    copy_sources(args.source_root, args.output_root)
    print(f"Vendored 234 branded-skill sources into {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
