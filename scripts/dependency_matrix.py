#!/usr/bin/env python3
"""Generate a domain-oriented dependency inventory and gap report.

Scans canonical dependency manifests and lockfiles in the repository and emits a
machine- and human-readable report for:

- `package.json` dependencies (runtime/dev/optional/peer)
- `pyproject.toml` dependencies (project + optional groups)
- `requirements*.txt` dependency lines
- lockfile coverage and pairing gaps

The scan deliberately skips generated/vendor paths such as `node_modules`,
`.next`, and build artifacts so the matrix stays operationally useful.
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from datetime import datetime, timezone


try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[import-not-found]

import json as _json


ROOT = Path(".")
SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    ".next",
    ".nuxt",
    "dist",
    "build",
    "out",
    ".turbo",
    "coverage",
    ".mypy_cache",
    ".pytest_cache",
    ".cache",
    "vendor",
    "__pycache__",
    ".yarn",
    ".pnpm-store",
    "archive/provider-reference-kits/.legacy",
}


def normalize_package_name(spec: str) -> str:
    """Return the raw requirement token as a package key for reporting."""

    return spec.split("[", 1)[0].split("==", 1)[0].split(">=", 1)[0].split("<=", 1)[0].split("~=", 1)[0].split(">", 1)[0].split("<", 1)[0].split("!=", 1)[0].split(";", 1)[0].strip()


def classify_domain(path: Path) -> str:
    parts = path.parts

    if len(parts) == 1:
        return "root"

    top = parts[0]
    if top == "frontend":
        return "frontend"
    if top in {"api", "agents", "orchestration", "scripts"}:
        return "api-orchestration-agents"
    if top == "wordpress-theme":
        if len(parts) > 1 and parts[1] == "skyyrose-flagship-2":
            return "wordpress-theme-flagship-2"
        if len(parts) > 1 and parts[1] == "skyyrose-flagship":
            return "wordpress-theme-flagship"
        return "wordpress-theme"
    if top == "design-system":
        return "design-system"
    if top == "sdk":
        return "sdk"
    if top == "dev":
        return "dev-tools"
    if top == "openai-responses-starter-app":
        return "openai-starter"
    if top == "hf-spaces":
        return "hf-spaces"
    if top == "skyyrose-suite":
        return "skyyrose-suite"
    if top == "plugins":
        return "plugins"
    if top == "tests":
        return "tests"
    if top == "wordpress":
        return "wordpress"
    if top == "devskyy-sdk-app":
        return "devskyy-sdk-app"
    if top == "claude-context":
        return "claude-context"
    if top == "archive":
        return "archive"
    return top


def parse_package_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    return {
        "dependencies": payload.get("dependencies", {}) or {},
        "devDependencies": payload.get("devDependencies", {}) or {},
        "optionalDependencies": payload.get("optionalDependencies", {}) or {},
        "peerDependencies": payload.get("peerDependencies", {}) or {},
        "scripts": sorted((payload.get("scripts") or {}).keys()),
        "packageManager": payload.get("packageManager"),
    }


def parse_lockfile(path: Path) -> dict[str, Any]:
    payload = _json.loads(path.read_text())
    return {
        "name": payload.get("name"),
        "version": payload.get("version"),
        "lockfileVersion": payload.get("lockfileVersion"),
        "package_count": len(payload.get("packages", {}) or {}),
        "dependency_count": len(payload.get("dependencies", {}) or {}),
    }


def parse_pyproject(path: Path) -> dict[str, Any]:
    payload = tomllib.loads(path.read_text())
    project = payload.get("project", {})
    dependencies = project.get("dependencies", []) or []
    optional_groups = project.get("optional-dependencies", {})

    if not isinstance(optional_groups, dict):
        optional_groups = {}

    return {
        "projectDependencies": [
            item for item in (dependencies or []) if isinstance(item, str)
        ],
        "optionalDependencies": {
            name: [item for item in items if isinstance(item, str)]
            for name, items in optional_groups.items()
        },
    }


def parse_requirements(path: Path) -> list[str]:
    lines: list[str] = []
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-r") or line.startswith("--"):
            continue
        normalized = line.split(" #", 1)[0].strip()
        if normalized:
            lines.append(normalized)
    return lines


def is_unpinned(spec: str) -> bool:
    base = spec.split(";")[0].strip()
    return not bool(re.search(r"(==|>=|<=|~=|>|<|!=)", base))


def is_manifest(path: Path) -> bool:
    if path.name == "package.json":
        return True
    if path.name == "package-lock.json":
        return True
    if path.name == "pyproject.toml":
        return True
    if path.name == "requirements.txt":
        return True
    if path.name == "requirements-dev.txt":
        return True
    if fnmatch.fnmatch(path.name, "requirements*.txt"):
        return True
    return False


def collect_manifests() -> list[Path]:
    found: list[Path] = []
    for root, dirs, files in os.walk(ROOT):
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".") or d in {".github", ".vscode", ".github"}]
        current = Path(root)
        if any(part in SKIP_DIRS for part in current.parts):
            continue
        for filename in files:
            candidate = current / filename
            if not is_manifest(candidate):
                continue
            if any(part in SKIP_DIRS for part in candidate.parts):
                continue
            found.append(candidate.relative_to(ROOT))
    return sorted(set(found))


def build_matrix(paths: list[Path]) -> dict[str, Any]:
    domains: dict[str, list[dict[str, Any]]] = defaultdict(list)
    lockfiles: set[Path] = set()
    manifests: set[Path] = set(paths)

    rows: list[dict[str, Any]] = []
    for path in paths:
        row: dict[str, Any] = {
            "path": str(path).replace("\\", "/"),
            "domain": classify_domain(path),
            "type": path.name,
            "sizeBytes": path.stat().st_size,
            "dependencies": {},
            "counts": {},
        }

        if path.name == "package.json":
            info = parse_package_json(path)
            row["dependencies"] = {
                "dependencies": info["dependencies"],
                "devDependencies": info["devDependencies"],
                "optionalDependencies": info["optionalDependencies"],
                "peerDependencies": info["peerDependencies"],
            }
            row["counts"] = {
                "dependencies": len(info["dependencies"]),
                "devDependencies": len(info["devDependencies"]),
                "optionalDependencies": len(info["optionalDependencies"]),
                "peerDependencies": len(info["peerDependencies"]),
                "scripts": len(info["scripts"]),
            }
        elif path.name == "package-lock.json":
            info = parse_lockfile(path)
            lockfiles.add(path)
            row["dependencies"] = info
            row["counts"] = {
                "lockfileVersion": info["lockfileVersion"],
                "package_count": info["package_count"],
                "dependency_count": info["dependency_count"],
            }
        elif path.name == "pyproject.toml":
            info = parse_pyproject(path)
            row["dependencies"] = info
            row["counts"] = {
                "projectDependencies": len(info["projectDependencies"]),
                "optionalGroups": len(info["optionalDependencies"]),
                "optionalDependencies": sum(len(v) for v in info["optionalDependencies"].values()),
            }
        else:
            req = parse_requirements(path)
            row["requirements"] = req
            row["counts"] = {
                "requirementsEntries": len(req),
                "unpinnedEntries": sum(1 for item in req if is_unpinned(item)),
                "exactPins": sum(1 for item in req if "==" in item),
            }

        domains[row["domain"]].append(row)
        rows.append(row)

    # Missing lockfile for npm manifests
    missing_locks: list[str] = []
    for row in rows:
        p = Path(row["path"])
        if p.name == "package.json":
            if not (p.with_name("package-lock.json") in manifests):
                missing_locks.append(row["path"])

    # Orphan lockfiles
    orphan_locks: list[str] = []
    for p in sorted(lockfiles):
        if not (p.with_name("package.json") in manifests):
            orphan_locks.append(str(p).replace("\\", "/"))

    # Unpinned requirements
    unpinned_reqs: list[str] = []
    for row in rows:
        if row["type"].startswith("requirements") and row["counts"]["unpinnedEntries"]:
            unpinned_reqs.append(row["path"])

    totals = {
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "manifests": len(rows),
        "packageJson": sum(1 for r in rows if r["type"] == "package.json"),
        "lockFiles": sum(1 for r in rows if r["type"] == "package-lock.json"),
        "pyproject": sum(1 for r in rows if r["type"] == "pyproject.toml"),
        "requirements": sum(1 for r in rows if r["type"].startswith("requirements")),
        "totalDependencies": {
            "npmDependencies": sum(r.get("counts", {}).get("dependencies", 0) for r in rows if r["type"] == "package.json"),
            "npmDevDependencies": sum(r.get("counts", {}).get("devDependencies", 0) for r in rows if r["type"] == "package.json"),
            "npmOptionalDependencies": sum(r.get("counts", {}).get("optionalDependencies", 0) for r in rows if r["type"] == "package.json"),
            "npmPeerDependencies": sum(r.get("counts", {}).get("peerDependencies", 0) for r in rows if r["type"] == "package.json"),
        },
        "gaps": {
            "missingLockFiles": len(missing_locks),
            "orphanLockFiles": len(orphan_locks),
            "requirementsUnpinned": len(unpinned_reqs),
        },
    }

    return {
        "totals": totals,
        "domains": {domain: sorted(manifests, key=lambda item: item["path"]) for domain, manifests in domains.items()},
        "gaps": {
            "missingLockFiles": sorted(missing_locks),
            "orphanLockFiles": sorted(orphan_locks),
            "requirementsUnpinned": sorted(unpinned_reqs),
        },
    }


def render_markdown(matrix: dict[str, Any]) -> str:
    lines: list[str] = []
    totals = matrix["totals"]
    lines.append("# Dependency Matrix")
    lines.append(f"Generated: {totals['generatedAt']}")
    lines.append("")
    lines.append("## Totals")
    lines.append(f"- Manifests: {totals['manifests']}")
    lines.append(f"- package.json: {totals['packageJson']}")
    lines.append(f"- package-lock.json: {totals['lockFiles']}")
    lines.append(f"- pyproject.toml: {totals['pyproject']}")
    lines.append(f"- requirements files: {totals['requirements']}")
    lines.append("")
    lines.append("### Dependency counts")
    for key, value in totals["totalDependencies"].items():
        lines.append(f"- {key}: {value}")
    lines.append("")

    lines.append("## Domain Matrix")
    for domain in sorted(matrix["domains"].keys()):
        lines.append("")
        lines.append(f"### {domain}")
        for manifest in matrix["domains"][domain]:
            lines.append(f"- `{manifest['path']}`")
            counts = manifest.get("counts", {})
            if manifest["type"] == "package.json":
                lines.append(
                    f"  - deps: {counts.get('dependencies', 0)} | dev: {counts.get('devDependencies', 0)} | optional: {counts.get('optionalDependencies', 0)} | peer: {counts.get('peerDependencies', 0)}"
                )
            elif manifest["type"].startswith("requirements"):
                lines.append(
                    f"  - requirements lines: {counts.get('requirementsEntries', 0)} (unpinned: {counts.get('unpinnedEntries', 0)})"
                )
            elif manifest["type"] == "pyproject.toml":
                lines.append(
                    f"  - project deps: {counts.get('projectDependencies', 0)} | optional groups: {counts.get('optionalGroups', 0)}"
                )
            else:
                lines.append(
                    f"  - lockfileVersion {counts.get('lockfileVersion')} with {counts.get('package_count', 0)} packages"
                )

    lines.append("")
    lines.append("## Gaps")
    lines.append(f"- Missing lockfiles: {totals['gaps']['missingLockFiles']}")
    if matrix["gaps"]["missingLockFiles"]:
        for item in matrix["gaps"]["missingLockFiles"]:
            lines.append(f"  - {item}")
    lines.append(f"- Orphan lockfiles: {totals['gaps']['orphanLockFiles']}")
    if matrix["gaps"]["orphanLockFiles"]:
        for item in matrix["gaps"]["orphanLockFiles"]:
            lines.append(f"  - {item}")
    lines.append(f"- Unpinned requirements files: {totals['gaps']['requirementsUnpinned']}")
    if matrix["gaps"]["requirementsUnpinned"]:
        for item in matrix["gaps"]["requirementsUnpinned"]:
            lines.append(f"  - {item}")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate dependency matrix by domain")
    parser.add_argument("--out-json", help="Optional JSON output path")
    parser.add_argument("--out-md", help="Optional Markdown output path")
    parser.add_argument("--strict", action="store_true", help="Exit non-zero when gaps are found")
    args = parser.parse_args()

    manifests = collect_manifests()
    matrix = build_matrix(manifests)

    print(f"TOTAL_MANIFESTS={matrix['totals']['manifests']}")
    print(f"TOTAL_PACKAGE_JSON={matrix['totals']['packageJson']}")
    print(f"TOTAL_PYPROJECT={matrix['totals']['pyproject']}")
    print(f"TOTAL_REQUIREMENTS={matrix['totals']['requirements']}")
    print(f"TOTAL_LOCKFILES={matrix['totals']['lockFiles']}")
    print(f"GAPS_MISSING_LOCK={matrix['totals']['gaps']['missingLockFiles']}")
    print(f"GAPS_ORPHAN_LOCK={matrix['totals']['gaps']['orphanLockFiles']}")
    print(f"GAPS_UNPINNED_REQ={matrix['totals']['gaps']['requirementsUnpinned']}")

    if args.out_json:
        Path(args.out_json).write_text(json.dumps(matrix, indent=2) + "\n", encoding="utf-8")
        print(f"JSON output: {args.out_json}")

    if args.out_md:
        Path(args.out_md).write_text(render_markdown(matrix), encoding="utf-8")
        print(f"Markdown output: {args.out_md}")

    if args.strict and matrix["totals"]["gaps"]["missingLockFiles"]:
        return 1
    if args.strict and matrix["totals"]["gaps"]["orphanLockFiles"]:
        return 1
    if args.strict and matrix["totals"]["gaps"]["requirementsUnpinned"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
