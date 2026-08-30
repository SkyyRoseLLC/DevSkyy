#!/usr/bin/env python3
"""Fail-closed validation for the branded-skill source registry."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--plugin-root", type=Path)
    parser.add_argument("--expected-count", type=int, default=234)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    registry = json.loads(args.registry.read_text(encoding="utf-8"))
    plugin_root = (
        args.plugin_root.resolve() if args.plugin_root else Path(__file__).resolve().parents[5]
    )
    errors: list[str] = []
    records = registry.get("sources", [])
    if len(records) != args.expected_count or registry.get("source_count") != args.expected_count:
        errors.append(f"expected {args.expected_count} records")
    ids = [record.get("id") for record in records]
    paths = [record.get("source_path") for record in records]
    if len(set(ids)) != len(ids):
        errors.append("duplicate source id")
    if len(set(paths)) != len(paths):
        errors.append("duplicate source path")
    if paths != sorted(paths):
        errors.append("records are not in lexical source-path order")
    authority_ids = {item["id"] for item in registry["authority_requirements"]["domains"]}
    for record in records:
        path = plugin_root / registry["source_root"] / record["source_path"]
        if not path.is_file():
            errors.append(f"missing source: {path}")
            continue
        data = path.read_bytes()
        if digest(data) != record["sha256"]:
            errors.append(f"source hash mismatch: {path}")
        text = data.decode("utf-8")
        raw = record["frontmatter"]["raw_exact"]
        if raw is None or not text.startswith(raw):
            errors.append(f"frontmatter mismatch: {path}")
        elif digest(raw.encode("utf-8")) != record["frontmatter"]["raw_sha256"]:
            errors.append(f"frontmatter hash mismatch: {path}")
        if record["authority_requirement_id"] not in authority_ids:
            errors.append(f"missing authority requirement: {path}")
        if record["classification"]["fashion_applicability"] not in {
            "high",
            "medium",
            "conditional",
            "low",
        }:
            errors.append(f"invalid fashion applicability: {path}")
    external_urls = []
    for requirement in registry["authority_requirements"]["domains"]:
        for authority in requirement["primary_authorities"]:
            url = authority["url"]
            if url.startswith("https://"):
                external_urls.append(url)
            elif url != "repo://SOT":
                errors.append(f"non-authenticated external authority URL: {url}")
    report = {
        "registry": args.registry.resolve().relative_to(plugin_root).as_posix(),
        "expected_count": args.expected_count,
        "observed_count": len(records),
        "unique_ids": len(set(ids)),
        "unique_paths": len(set(paths)),
        "source_hashes_verified": (
            len(records) if not any("source hash" in error for error in errors) else "FAILED"
        ),
        "frontmatter_verified": (
            len(records) if not any("frontmatter" in error for error in errors) else "FAILED"
        ),
        "authority_domain_count": len(authority_ids),
        "authenticated_https_authority_url_count": len(set(external_urls)),
        "authority_artifact_capture": "Validated separately by authority-capture.json; claim-specific anchors must still be fresh at use time",
        "unknown_count": len(registry.get("unknowns", [])),
        "errors": errors,
        "verdict": "PASS" if not errors else "FAIL",
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.report:
        args.report.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if not errors else 1


if __name__ == "__main__":
    raise SystemExit(main())
