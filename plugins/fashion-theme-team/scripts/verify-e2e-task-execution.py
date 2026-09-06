#!/usr/bin/env python3
"""Verify that every governed Fashion Theme Team capability has E2E control."""

from __future__ import annotations

import json
import sys
from pathlib import Path

SKILLS = (
    "fashion-theme-team",
    "elite-builder-runtime",
    "fashion-strategy-brain",
    "fashion-brand-experience",
    "fashion-design-system",
    "fashion-commerce-engineering",
    "fashion-frontend-motion",
    "fashion-a11y-performance",
    "fashion-visual-commerce-qa",
    "fashion-handoff-contracts",
    "fashion-release-evidence",
    "fashion-premium-feature-system",
    "fashion-branded-skills",
    "product-fidelity-image-edits",
)
MARKER = "fashion-e2e-task-execution"


def main(argv: list[str] | None = None) -> int:
    del argv
    root = Path(__file__).resolve().parents[1]
    e2e = root / "skills" / MARKER / "SKILL.md"
    contract = (
        root / "skills" / "fashion-theme-team" / "references" / "e2e-task-execution-contract.md"
    )
    manifest = root / ".codex-plugin" / "plugin.json"
    errors: list[str] = []
    if not e2e.is_file() or not e2e.read_text(encoding="utf-8").strip():
        errors.append("missing E2E execution skill")
    if not contract.is_file() or not contract.read_text(encoding="utf-8").strip():
        errors.append("missing E2E task contract")
    if "project manager" not in e2e.read_text(encoding="utf-8").lower():
        errors.append("E2E execution skill does not issue a project manager")
    try:
        plugin = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid plugin manifest: {error}")
        plugin = {}
    interface = plugin.get("interface", {})
    if "Strict End-to-End Task Control" not in interface.get("capabilities", []):
        errors.append("plugin manifest does not advertise strict E2E task control")
    if not any(
        "strict end-to-end task" in prompt.lower()
        for prompt in interface.get("defaultPrompt", [])
        if isinstance(prompt, str)
    ):
        errors.append("plugin manifest does not expose a strict E2E default prompt")
    for required_phrase in (
        "strict e2e",
        "acceptance criteria",
        "independent reviewer",
        "remediation loop",
        "founder approval",
        "mcp",
        "task roster",
        "job title",
    ):
        if required_phrase not in e2e.read_text(encoding="utf-8").lower():
            errors.append(f"E2E execution skill is missing {required_phrase} control")
    for name in SKILLS:
        skill = root / "skills" / name / "SKILL.md"
        if MARKER not in skill.read_text(encoding="utf-8"):
            errors.append(f"{name} does not route through {MARKER}")
    if errors:
        print("\n".join(f"FAIL: {error}" for error in errors), file=sys.stderr)
        return 1
    print(f"PASS: {len(SKILLS)} Fashion Theme Team capabilities route through E2E task control")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
