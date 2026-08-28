#!/usr/bin/env python3
"""Check or mirror the canonical Fashion Theme Team package to one local target.

The utility operates only on an explicit existing Fashion Theme Team directory.
It intentionally excludes Git metadata and local runtime/cache artefacts so a
distribution can keep its own repository history while package content remains
identical to the canonical source.
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

PACKAGE_NAME = "fashion-theme-team"
LOCAL_PATH_PARTS = {
    ".git",
    ".ruff_cache",
    "__pycache__",
    ".pytest_cache",
    ".DS_Store",
}
LOCAL_FILE_NAMES = {".coverage"}
LOCAL_RUNTIME_PREFIXES = {
    Path("runtime/elite_web_builder/output"),
    Path("runtime/elite_web_builder/.coverage"),
}


@dataclass(frozen=True)
class Difference:
    missing: tuple[Path, ...]
    changed: tuple[Path, ...]
    stale: tuple[Path, ...]

    @property
    def is_clean(self) -> bool:
        return not (self.missing or self.changed or self.stale)


def is_local_path(path: Path) -> bool:
    return (
        any(part in LOCAL_PATH_PARTS for part in path.parts)
        or path.name in LOCAL_FILE_NAMES
        or any(path == prefix or prefix in path.parents for prefix in LOCAL_RUNTIME_PREFIXES)
    )


def package_files(root: Path) -> set[Path]:
    files: set[Path] = set()
    for candidate in root.rglob("*"):
        relative = candidate.relative_to(root)
        if is_local_path(relative):
            continue
        if candidate.is_file() and not candidate.is_symlink():
            files.add(relative)
        elif candidate.is_symlink():
            raise RuntimeError(f"symlinked package content is not supported: {candidate}")
    return files


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_package(root: Path, label: str) -> None:
    manifest = root / ".codex-plugin/plugin.json"
    if not root.is_dir():
        raise RuntimeError(f"{label} is not a directory: {root}")
    if not manifest.is_file():
        raise RuntimeError(f"{label} is not a Fashion Theme Team package: {manifest} is missing")
    if '"name": "fashion-theme-team"' not in manifest.read_text(encoding="utf-8"):
        raise RuntimeError(f"{label} manifest does not identify {PACKAGE_NAME}")


def compare(source: Path, target: Path) -> Difference:
    source_files = package_files(source)
    target_files = package_files(target)
    missing = tuple(sorted(source_files - target_files))
    stale = tuple(sorted(target_files - source_files))
    changed = tuple(
        sorted(
            relative
            for relative in source_files & target_files
            if sha256(source / relative) != sha256(target / relative)
        )
    )
    return Difference(missing=missing, changed=changed, stale=stale)


def print_difference(difference: Difference) -> None:
    for label, paths in (
        ("MISSING", difference.missing),
        ("CHANGED", difference.changed),
        ("STALE", difference.stale),
    ):
        for relative in paths:
            print(f"{label} {relative}")
    print(
        "summary "
        f"missing={len(difference.missing)} "
        f"changed={len(difference.changed)} "
        f"stale={len(difference.stale)}"
    )


def remove_empty_parent_directories(root: Path, relative: Path) -> None:
    parent = (root / relative).parent
    while parent != root:
        try:
            parent.rmdir()
        except OSError:
            break
        parent = parent.parent


def apply(source: Path, target: Path, difference: Difference, prune: bool) -> None:
    for relative in (*difference.missing, *difference.changed):
        destination = target / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source / relative, destination)
        print(f"WROTE {relative}")
    if prune:
        for relative in difference.stale:
            destination = target / relative
            destination.unlink()
            remove_empty_parent_directories(target, relative)
            print(f"REMOVED {relative}")
    elif difference.stale:
        print("STALE files retained; rerun with --prune for exact package parity.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target",
        type=Path,
        required=True,
        help="existing local Fashion Theme Team distribution to inspect or synchronize",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help="report drift and exit nonzero when present",
    )
    mode.add_argument(
        "--apply",
        action="store_true",
        help="copy missing and changed package files into target",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="with --apply, remove package files that no longer exist in the canonical source",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.prune and not args.apply:
        raise RuntimeError("--prune requires --apply")
    source = Path(__file__).resolve().parents[1]
    target = args.target.expanduser().resolve()
    if source == target:
        raise RuntimeError("target must be a distribution, not the canonical source")
    validate_package(source, "canonical source")
    validate_package(target, "target")
    difference = compare(source, target)
    if not args.apply:
        print_difference(difference)
        return 0 if difference.is_clean else 1
    apply(source, target, difference, args.prune)
    remaining = compare(source, target)
    print_difference(remaining)
    return 0 if remaining.is_clean else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except RuntimeError as error:
        print(f"FAIL: {error}", file=sys.stderr)
        raise SystemExit(2) from error
