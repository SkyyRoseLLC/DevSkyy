#!/usr/bin/env python3
"""Deterministic learning, hardening, repair, and ship gates for DevSkyy."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((PLUGIN_ROOT / "config/guardian.json").read_text(encoding="utf-8"))
SECRET_KEYS = {"authorization", "password", "secret", "token", "api_key", "cookie"}


def repo_root(value: str | None) -> Path:
    if value:
        return Path(value).resolve()
    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        check=True,
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve()


def state_dir(root: Path) -> Path:
    path = root / ".wolf" / "production-guardian"
    path.mkdir(parents=True, exist_ok=True)
    return path


def redact(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            key: "[REDACTED]" if key.lower() in SECRET_KEYS else redact(item)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [redact(item) for item in value]
    return value


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")
    path.chmod(0o600)


def changed_paths(root: Path) -> list[str]:
    commands = (
        ["git", "diff", "--name-only"],
        ["git", "diff", "--cached", "--name-only"],
        ["git", "ls-files", "--others", "--exclude-standard"],
    )
    paths: set[str] = set()
    for command in commands:
        result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
        paths.update(line for line in result.stdout.splitlines() if line)
    return sorted(paths)


def static_findings(root: Path) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    for relative in CONFIG["production_consumers"]:
        path = root / relative
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for forbidden in CONFIG["forbidden_production_sources"]:
            if forbidden in text:
                findings.append(
                    {
                        "rule": "no-legacy-production-source",
                        "path": relative,
                        "detail": f"production consumer references retired source {forbidden}",
                    }
                )

    settings = root / ".claude/settings.json"
    if settings.is_file() and re.search(r"/Users/[^/]+/DevSkyy/", settings.read_text()):
        findings.append(
            {
                "rule": "worktree-portability",
                "path": ".claude/settings.json",
                "detail": "hook configuration hardcodes a primary checkout",
            }
        )

    changed = changed_paths(root)
    changed_set = set(changed)
    for relative in changed:
        if relative.endswith(".css") and not relative.endswith(".min.css"):
            minified = relative[:-4] + ".min.css"
            if (root / minified).is_file() and minified not in changed_set:
                findings.append(
                    {"rule": "minified-pair", "path": relative, "detail": f"missing {minified}"}
                )
        if relative.endswith(".js") and not relative.endswith(".min.js"):
            minified = relative[:-3] + ".min.js"
            if (root / minified).is_file() and minified not in changed_set:
                findings.append(
                    {"rule": "minified-pair", "path": relative, "detail": f"missing {minified}"}
                )
    return findings


def run_checks(root: Path, mode: str) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    findings = static_findings(root)
    runs: list[dict[str, Any]] = []
    if mode in {"ship", "ci"}:
        for command in CONFIG["ship_checks"]:
            result = subprocess.run(command, cwd=root, capture_output=True, text=True, check=False)
            runs.append(
                {
                    "command": command,
                    "exit_code": result.returncode,
                    "stdout": result.stdout[-4000:],
                    "stderr": result.stderr[-4000:],
                }
            )
            if result.returncode:
                findings.append(
                    {
                        "rule": "ship-check",
                        "path": command[1] if len(command) > 1 else command[0],
                        "detail": f"exit {result.returncode}: {' '.join(command)}",
                    }
                )
    return findings, runs


def write_receipt(root: Path, findings: list[dict[str, str]], runs: list[dict[str, Any]]) -> Path:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    dirty = bool(changed_paths(root))
    receipt = {
        "schema": "skyyrose.production-guardian.receipt.v1",
        "created_at": int(time.time()),
        "head": head,
        "dirty": dirty,
        "status": "pass" if not findings and not dirty else "blocked",
        "findings": findings,
        "runs": runs,
    }
    path = state_dir(root) / "ship-receipt.json"
    path.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    return path


def command_check(args: argparse.Namespace) -> int:
    root = repo_root(args.root)
    findings, runs = run_checks(root, args.mode)
    receipt = write_receipt(root, findings, runs) if args.mode in {"ship", "ci"} else None
    print(json.dumps({"status": "pass" if not findings else "blocked", "findings": findings}, indent=2))
    if receipt:
        print(f"receipt={receipt}")
    return 1 if findings else 0


def command_observe(args: argparse.Namespace) -> int:
    root = repo_root(args.root)
    raw = sys.stdin.read().strip()
    payload = json.loads(raw) if raw else {}
    record = {
        "timestamp": int(time.time()),
        "event": args.event,
        "head": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=False
        ).stdout.strip(),
        "payload": redact(payload),
    }
    append_jsonl(state_dir(root) / "observations.jsonl", record)
    return 0


def command_learn(args: argparse.Namespace) -> int:
    root = repo_root(args.root)
    buglog = root / ".wolf/buglog.json"
    bugs = json.loads(buglog.read_text(encoding="utf-8")) if buglog.is_file() else []
    counts: Counter[str] = Counter()
    evidence: dict[str, list[str]] = {}
    for bug in bugs:
        signature = "|".join(sorted(bug.get("tags", []))) or bug.get("root_cause", "unknown")
        occurrences = max(1, int(bug.get("occurrences", 1)))
        counts[signature] += occurrences
        evidence.setdefault(signature, []).append(str(bug.get("id", "unknown")))
    threshold = int(CONFIG["learning"]["recommendation_occurrences"])
    recommendations = [
        {"signature": signature, "occurrences": count, "bugs": evidence[signature][:20]}
        for signature, count in counts.most_common()
        if count >= threshold
    ]
    output = state_dir(root) / "recommendations.json"
    output.write_text(json.dumps(recommendations, indent=2) + "\n", encoding="utf-8")
    print(f"learned_recommendations={len(recommendations)} path={output}")
    return 0


def command_repair(args: argparse.Namespace) -> int:
    root = repo_root(args.root)
    for command in CONFIG["safe_repairs"]:
        print(("DRY RUN " if args.dry_run else "RUN ") + " ".join(command))
        if not args.dry_run:
            result = subprocess.run(command, cwd=root, check=False)
            if result.returncode:
                return result.returncode
    return 0


def command_preflight(args: argparse.Namespace) -> int:
    root = repo_root(args.root)
    command = args.command or sys.stdin.read()
    if not any(pattern in command.lower() for pattern in CONFIG["protected_commands"]):
        return 0
    receipt_path = state_dir(root) / "ship-receipt.json"
    if not receipt_path.is_file():
        print("BLOCKED: protected command requires a production-guardian ship receipt", file=sys.stderr)
        return 2
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.strip()
    fresh = int(time.time()) - int(receipt.get("created_at", 0)) <= 3600
    if receipt.get("status") != "pass" or receipt.get("head") != head or not fresh:
        print("BLOCKED: guardian receipt is failed, stale, or for another commit", file=sys.stderr)
        return 2
    return 0


def parser() -> argparse.ArgumentParser:
    cli = argparse.ArgumentParser(description=__doc__)
    commands = cli.add_subparsers(dest="action", required=True)
    for name in ("check", "observe", "learn", "repair", "preflight"):
        item = commands.add_parser(name)
        item.add_argument("--root")
    commands.choices["check"].add_argument("--mode", choices=("fast", "ship", "ci"), default="fast")
    commands.choices["observe"].add_argument("--event", required=True)
    commands.choices["repair"].add_argument("--dry-run", action="store_true")
    commands.choices["preflight"].add_argument("--command")
    return cli


def main() -> int:
    args = parser().parse_args()
    return {
        "check": command_check,
        "observe": command_observe,
        "learn": command_learn,
        "repair": command_repair,
        "preflight": command_preflight,
    }[args.action](args)


if __name__ == "__main__":
    raise SystemExit(main())
