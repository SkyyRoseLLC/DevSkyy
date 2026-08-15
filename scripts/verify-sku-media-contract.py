#!/usr/bin/env python3
"""Report or enforce the all-SKU multi-view media production contract.

This never generates or promotes images.  It derives a deterministic job matrix
from the catalog-backed collection SOT so production runs can be reviewed before
any paid model call and verified before storefront use.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyyrose.core.sku_media_contract import (  # noqa: E402
    build_media_matrix,
    build_production_jobs,
    missing_views,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="emit the full deterministic production matrix"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero unless every required SKU view is present",
    )
    parser.add_argument(
        "--jobs",
        action="store_true",
        help="emit approval-gated multi-model production jobs instead of coverage",
    )
    parser.add_argument(
        "--include-ready",
        action="store_true",
        help="with --jobs, include views that are already complete",
    )
    args = parser.parse_args()

    matrix = build_media_matrix()
    missing = missing_views(matrix)
    if args.jobs:
        print(
            json.dumps(build_production_jobs(matrix, only_missing=not args.include_ready), indent=2)
        )
        return 0
    if args.json:
        print(json.dumps(matrix, indent=2))
    else:
        summary = matrix["summary"]
        print(
            "SKU media contract: "
            f"{summary['complete_sku_count']}/{summary['sku_count']} complete; "
            f"{len(missing)} required views missing"
        )
        for item in missing:
            print(f"MISSING {item}")

    if args.check and missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
