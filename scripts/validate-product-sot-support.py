#!/usr/bin/env python3
"""Audit whether every catalog SKU has support for each declared downstream use."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyyrose.core.product_sot_support import build_support_report, strict_blockers  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="Emit the full SKU support matrix.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail unless every published SKU has evidence for every support area.",
    )
    args = parser.parse_args()
    report = build_support_report()
    blockers = strict_blockers(report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(f"Product SOT support: {report['product_sot']['sku_count']} SKUs")
        for area, states in report["summary"]["by_area_state"].items():
            fragments = ", ".join(f"{state}={count}" for state, count in sorted(states.items()))
            print(f"{area}: {fragments}")
        if blockers:
            print(f"strict blockers: {len(blockers)}")
    if args.strict and blockers:
        for blocker in blockers:
            print(f"BLOCKED {blocker}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
