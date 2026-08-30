#!/usr/bin/env python3
"""Run the Fashion Theme Team's deterministic, source-aware format matrix.

The matrix deliberately separates source formatting from generated/vendor parity.
It uses repo-local tools when available, validates syntax for every supported
language, and keeps a deterministic text policy as the portable baseline. Use
``--fix`` for safe newline/whitespace normalization plus formatter autofixes.
"""

from __future__ import annotations

import argparse
import ast
import fnmatch
import json
import os
import re
import shutil
import subprocess
import sys
import tomllib
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Sequence

SCRIPT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_ROOT.parent
CONFIG_PATH = REPO_ROOT / "formatting.toml"
PYTHON_VERSION_PATTERN = re.compile(r"(?:black, )(?P<version>[^ ]+)")


class MatrixError(RuntimeError):
    """A configuration or execution error in the formatting matrix."""


class _HTMLProbe(HTMLParser):
    """Feed HTML to the stdlib parser so malformed markup fails closed."""

    def error(self, message: str) -> None:  # pragma: no cover - Parser API on older Python
        raise MatrixError(message)


def load_config() -> dict[str, Any]:
    try:
        with CONFIG_PATH.open("rb") as handle:
            config = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise MatrixError(f"cannot load {CONFIG_PATH}: {exc}") from exc
    if config.get("schema") != "fashion-theme-team-formatting.v1":
        raise MatrixError("unsupported formatting.toml schema")
    return config


def run_git(*args: str) -> bytes:
    try:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        detail = getattr(exc, "stderr", b"").decode(errors="replace").strip()
        raise MatrixError(f"git {' '.join(args)} failed{': ' + detail if detail else ''}") from exc


def has_git_worktree() -> bool:
    """Return whether the package is running from a Git worktree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=REPO_ROOT,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return False
    return result.returncode == 0 and result.stdout.strip() == "true"


def decode_nul_list(payload: bytes) -> list[str]:
    return [item.decode("utf-8", errors="surrogateescape") for item in payload.split(b"\0") if item]


def changed_files() -> set[str]:
    paths = set(
        decode_nul_list(run_git("diff", "--name-only", "--diff-filter=ACMRTUXB", "-z", "HEAD"))
    )
    paths.update(decode_nul_list(run_git("ls-files", "--others", "--exclude-standard", "-z")))
    return paths


def candidate_files(changed_only: bool) -> list[str]:
    if not has_git_worktree():
        if changed_only:
            raise MatrixError("--changed requires a Git worktree; run --check in a portable copy")
        return sorted(
            path.relative_to(REPO_ROOT).as_posix()
            for path in REPO_ROOT.rglob("*")
            if path.is_file() and not path.is_symlink()
        )
    if changed_only:
        paths = changed_files()
    else:
        paths = set(
            decode_nul_list(run_git("ls-files", "--cached", "--others", "--exclude-standard", "-z"))
        )
    return sorted(path for path in paths if (REPO_ROOT / path).is_file())


def matches_exclude(path: str, pattern: str) -> bool:
    normalized = PurePosixPath(path).as_posix()
    return fnmatch.fnmatchcase(normalized, pattern)


def is_excluded(path: str, config: dict[str, Any]) -> bool:
    return any(matches_exclude(path, pattern) for pattern in config["scope"]["exclude"])


def extension(path: str) -> str:
    return Path(path).suffix.lower()


def language_for(path: str, config: dict[str, Any]) -> str | None:
    name = PurePosixPath(path).name
    for language, policy in config["language"].items():
        if name in policy.get("filenames", []):
            return language
        if extension(path) in policy.get("extensions", []):
            return language
    if extension(path) in config.get("binary", {}).get("extensions", []):
        return "binary"
    return None


def classify_files(
    paths: Iterable[str], config: dict[str, Any]
) -> tuple[dict[str, list[Path]], list[str], list[str]]:
    grouped: dict[str, list[Path]] = {}
    excluded: list[str] = []
    unknown: list[str] = []
    for relative in paths:
        if is_excluded(relative, config):
            excluded.append(relative)
            continue
        language = language_for(relative, config)
        if language == "binary":
            excluded.append(relative)
        elif language is None:
            unknown.append(relative)
        else:
            grouped.setdefault(language, []).append(REPO_ROOT / relative)
    return grouped, excluded, unknown


def read_text(path: Path) -> str:
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            return handle.read()
    except (OSError, UnicodeError) as exc:
        raise MatrixError(f"cannot read {path}: {exc}") from exc


def text_policy_errors(path: Path, text: str, language: str, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    relative = path.relative_to(REPO_ROOT).as_posix()
    if "\r" in text:
        errors.append(f"{relative}: line endings must be LF")
    if not text.endswith("\n"):
        errors.append(f"{relative}: missing final newline")
    if "\x00" in text:
        errors.append(f"{relative}: NUL byte found in text source")
    hard_break = config["policy"]["markdown_hard_break"]
    for line_number, line in enumerate(text.splitlines(), start=1):
        trailing = line[len(line.rstrip(" \t")) :]
        if not trailing:
            continue
        if language == "markdown" and hard_break == "exactly-two-spaces" and trailing == "  ":
            continue
        errors.append(f"{relative}:{line_number}: trailing whitespace")
    return errors


def normalize_text(path: Path, text: str, language: str, config: dict[str, Any]) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = normalized.split("\n")
    hard_break = config["policy"]["markdown_hard_break"]
    for index, line in enumerate(lines):
        trailing = line[len(line.rstrip(" \t")) :]
        if language == "markdown" and hard_break == "exactly-two-spaces" and trailing == "  ":
            continue
        lines[index] = line.rstrip(" \t")
    normalized = "\n".join(lines)
    if not normalized.endswith("\n"):
        normalized += "\n"
    return normalized


def resolve_tool(name: str) -> str | None:
    env_name = f"FASHION_FORMATTER_{name.upper().replace('-', '_')}"
    configured = os.environ.get(env_name)
    if configured:
        return configured
    for relative in (
        Path("node_modules") / ".bin" / name,
        Path(".venv-format") / "bin" / name,
        Path(".venv") / "bin" / name,
    ):
        candidate = REPO_ROOT / relative
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return shutil.which(name)


def tool_version(command: str, language: str, version_args: Sequence[str] = ("--version",)) -> str:
    try:
        result = subprocess.run(
            [command, *version_args],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MatrixError(f"unable to query {language} formatter {command}: {exc}") from exc
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else "unknown"


def enforce_pinned_version(
    command: str,
    language: str,
    expected: str,
    version_args: Sequence[str] = ("--version",),
) -> None:
    if not expected:
        return
    observed = tool_version(command, language, version_args)
    if language == "python":
        match = PYTHON_VERSION_PATTERN.search(observed)
        observed_version = match.group("version") if match else observed
    elif language == "isort":
        observed_version = observed
    elif language == "ruff":
        observed_version = observed.removeprefix("ruff ").split()[0]
    else:
        observed_version = observed
    if observed_version != expected and not observed_version.startswith(f"{expected} "):
        raise MatrixError(
            f"{language} formatter version drift: expected {expected}, observed {observed}; "
            "install requirements-formatting.txt or set a repo-local formatter"
        )


def run_command(command: Sequence[str], files: Sequence[Path]) -> None:
    if not files:
        return
    args = [*command, *(str(path) for path in files)]
    try:
        subprocess.run(args, cwd=REPO_ROOT, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise MatrixError(f"formatter command failed: {' '.join(args)}") from exc


def run_black(files: list[Path], fix: bool, config: dict[str, Any]) -> str:
    command = resolve_tool("black")
    if command is None:
        if os.environ.get("FASHION_REQUIRE_FORMATTER_TOOLS") == "1":
            raise MatrixError(
                "black is required but unavailable; install requirements-formatting.txt"
            )
        for path in files:
            try:
                ast.parse(read_text(path), filename=str(path))
            except (SyntaxError, ValueError) as exc:
                raise MatrixError(f"python syntax check failed for {path}: {exc}") from exc
        return "fallback: ast.parse (black unavailable)"
    enforce_pinned_version(
        command, "python", config["language"]["python"].get("formatter_version", "")
    )
    run_command([command] if fix else [command, "--check", "--diff"], files)
    return f"black {config['language']['python'].get('formatter_version', '')}".rstrip()


def run_python_quality_checks(
    files: list[Path],
    changed_paths: set[str],
    fix: bool,
    config: dict[str, Any],
) -> list[str]:
    """Run import/lint formatters on changed Python files only.

    Existing runtime files predate this matrix and have intentionally different
    import grouping. Checking only changed Python files prevents unrelated churn,
    while ensuring every new or modified Python file passes the full lane.
    """
    changed = [path for path in files if path.relative_to(REPO_ROOT).as_posix() in changed_paths]
    if not changed:
        return []
    python_config = config["language"]["python"]
    statuses: list[str] = []
    required = os.environ.get("FASHION_REQUIRE_FORMATTER_TOOLS") == "1"
    for tool_name, version_key, version_args, command_args in (
        (
            python_config.get("imports_formatter", "isort"),
            "imports_formatter_version",
            ("--version-number",),
            ["--check-only", "--diff"],
        ),
        (
            python_config.get("lint_formatter", "ruff"),
            "lint_formatter_version",
            ("--version",),
            ["check", "--select", "I", "--output-format", "concise"],
        ),
    ):
        command = resolve_tool(tool_name)
        if command is None:
            message = f"{tool_name} unavailable for changed Python files"
            if required:
                raise MatrixError(f"{message}; install requirements-formatting.txt")
            statuses.append(f"{tool_name} unavailable (baseline only)")
            continue
        enforce_pinned_version(command, tool_name, python_config.get(version_key, ""), version_args)
        if fix:
            if tool_name == "ruff":
                run_command([command, "check", "--select", "I", "--fix"], changed)
            else:
                run_command([command], changed)
        else:
            run_command([command, *command_args], changed)
        statuses.append(f"{tool_name} {python_config.get(version_key, '')}".rstrip())
    return statuses


def optional_formatter_command(
    language: str, config: dict[str, Any]
) -> tuple[str, list[str]] | None:
    policy = config["language"][language]
    formatter = policy.get("formatter", "")
    if formatter in {"", "deterministic text policy", "tomllib"}:
        return None
    expected_version = policy.get("formatter_version", "")
    if not expected_version:
        # A globally installed optional formatter must never make matrix results
        # differ across machines. Enable it only by pinning its version in config.
        return None
    command = resolve_tool(formatter)
    if command is None:
        return None
    enforce_pinned_version(command, language, expected_version)
    return command, list(config["language"][language].get("formatter_args", []))


def run_optional_formatter(
    language: str,
    command: str,
    formatter_args: list[str],
    files: list[Path],
    fix: bool,
) -> None:
    """Run an installed formatter with its tool-specific check/fix contract."""
    if language == "javascript" or language in {"json", "markdown", "css", "html", "yaml"}:
        mode = "--write" if fix else "--check"
        run_command([command, mode, *formatter_args], files)
        return
    if language == "shell":
        run_command([command, *formatter_args] if fix else [command, "-d", *formatter_args], files)
        return
    if language == "php":
        mode = ["fix", "--using-cache=no"] if fix else ["check", "--using-cache=no"]
        run_command([command, *mode], files)
        return
    run_command([command, *formatter_args], files)


def validate_syntax(language: str, files: list[Path]) -> None:
    for path in files:
        text = read_text(path)
        try:
            if language == "shell":
                subprocess.run(
                    ["bash", "-n", str(path)], cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL
                )
            elif language == "javascript":
                node = shutil.which("node")
                if node is None:
                    raise MatrixError("node is required for JavaScript syntax checks")
                subprocess.run(
                    [node, "--check", str(path)],
                    cwd=REPO_ROOT,
                    check=True,
                    stdout=subprocess.DEVNULL,
                )
            elif language == "json":
                json.loads(text)
            elif language == "html":
                parser = _HTMLProbe(convert_charrefs=False)
                parser.feed(text)
                parser.close()
            elif language == "yaml":
                try:
                    import yaml  # type: ignore[import-not-found]

                    yaml.safe_load(text)
                except ImportError as exc:
                    raise MatrixError(
                        "PyYAML is required for YAML syntax checks; install requirements-formatting.txt"
                    ) from exc
            elif language == "toml":
                tomllib.loads(text)
            elif language == "php":
                php = shutil.which("php")
                if php is None:
                    raise MatrixError("php is required for PHP syntax checks")
                subprocess.run(
                    [php, "-l", str(path)], cwd=REPO_ROOT, check=True, stdout=subprocess.DEVNULL
                )
        except (
            OSError,
            subprocess.CalledProcessError,
            SyntaxError,
            TypeError,
            ValueError,
            MatrixError,
        ) as exc:
            raise MatrixError(f"{language} syntax check failed for {path}: {exc}") from exc


def format_group(
    language: str,
    files: list[Path],
    changed_paths: set[str],
    fix: bool,
    config: dict[str, Any],
) -> tuple[str, list[str]]:
    if language == "python":
        formatter_status = run_black(files, fix, config)
        quality_status = run_python_quality_checks(files, changed_paths, fix, config)
        if quality_status:
            formatter_status = "; ".join([formatter_status, *quality_status])
    else:
        formatter = optional_formatter_command(language, config)
        if formatter:
            command, formatter_args = formatter
            run_optional_formatter(language, command, formatter_args, files, fix)
            formatter_status = Path(command).name
        else:
            formatter_status = config["language"][language].get(
                "fallback", "deterministic text policy"
            )

    errors: list[str] = []
    for path in files:
        text = read_text(path)
        if fix:
            normalized = normalize_text(path, text, language, config)
            if normalized != text:
                path.write_text(normalized, encoding="utf-8", newline="")
                text = normalized
        errors.extend(text_policy_errors(path, text, language, config))
    validate_syntax(language, files)
    if not fix and errors:
        return formatter_status, errors
    return formatter_status, errors


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="check source formatting (default)")
    mode.add_argument(
        "--fix", action="store_true", help="autofix supported formatters and text policy"
    )
    parser.add_argument(
        "--changed",
        action="store_true",
        help="limit the matrix to staged, unstaged, and untracked source files",
    )
    parser.add_argument("--verbose", action="store_true", help="print every file in the matrix")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    fix = args.fix
    config = load_config()
    paths = candidate_files(args.changed)
    changed_paths = changed_files() if has_git_worktree() else set()
    groups, excluded, unknown = classify_files(paths, config)
    if unknown:
        print("FAIL: unclassified text/source files:", file=sys.stderr)
        for relative in unknown:
            print(f"  {relative}", file=sys.stderr)
        return 1

    total = sum(len(files) for files in groups.values())
    print(
        f"FORMAT MATRIX: {'fix' if fix else 'check'} {total} source files; {len(excluded)} excluded"
    )
    failures: list[str] = []
    for language in sorted(groups):
        files = groups[language]
        if args.verbose:
            for path in files:
                print(f"  {language}: {path.relative_to(REPO_ROOT)}")
        try:
            status, errors = format_group(language, files, changed_paths, fix, config)
        except MatrixError as exc:
            failures.append(str(exc))
            continue
        if errors:
            failures.extend(errors)
            print(f"FAIL {language}: {len(errors)} policy violation(s)", file=sys.stderr)
        else:
            print(f"PASS {language}: {len(files)} files ({status})")

    if failures:
        print("FORMAT MATRIX FAILED:", file=sys.stderr)
        for failure in failures:
            print(f"  {failure}", file=sys.stderr)
        if not fix:
            print("Run scripts/check-formatting.sh --fix, then review the diff.", file=sys.stderr)
        return 1
    print("PASS: deterministic formatting matrix")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
