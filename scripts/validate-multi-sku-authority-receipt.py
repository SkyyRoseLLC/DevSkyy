#!/usr/bin/env python3
"""Fail closed on a combined-look source-authority provider receipt."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyyrose.core.on_model_media_intake import (  # noqa: E402
    MediaIntakeError,
    validate_multi_sku_generation_receipt,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-sot", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--require-output", action="store_true")
    args = parser.parse_args()
    try:
        report = validate_multi_sku_generation_receipt(
            args.product_sot,
            args.receipt,
            repo_root=args.repo_root,
            require_output=args.require_output,
        )
    except MediaIntakeError as exc:
        print(f"BLOCKED {exc}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
