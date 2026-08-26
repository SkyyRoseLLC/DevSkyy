#!/usr/bin/env python3
"""Fail closed unless V2 on-model card-media records match product SOT bytes."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from skyyrose.core.on_model_media_intake import MediaIntakeError, validate_media_intake  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--product-sot", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=ROOT)
    parser.add_argument("--json", action="store_true", help="Emit the full per-SKU report.")
    parser.add_argument(
        "--require-explicit-approval",
        action="store_true",
        help="Require an approval_reference for every accepted on-model front.",
    )
    args = parser.parse_args()
    try:
        report = validate_media_intake(
            args.product_sot,
            args.registry,
            repo_root=args.repo_root,
            require_explicit_approval=args.require_explicit_approval,
        )
    except MediaIntakeError as exc:
        print(f"BLOCKED {exc}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(
            "On-model media intake: "
            f"SUPPORTED={report['summary']['supported']} "
            f"BLOCKED={report['summary']['blocked']}"
        )
    return 0 if report["summary"]["blocked"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
