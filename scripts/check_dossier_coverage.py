"""Confirm every active SKU has a per-product design dossier.

Reads the canonical CSV at
`wordpress-theme/skyyrose-flagship/data/skyyrose-catalog.csv` and verifies
each active SKU has a matching dossier at
`wordpress-theme/skyyrose-flagship/data/dossiers/{slug}.md`.

The `dossier_slug` column is the sole mapping authority. Product names and
filenames are never used as a fallback.

Run from repo root:
    python scripts/check_dossier_coverage.py

Exits 0 if every SKU has a dossier, 1 otherwise. Suitable for CI.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skyyrose.core.dossier_loader import (
    DossierMissingError,
    DossierReferenceError,
    iter_dossier_bindings,
)


def main() -> int:
    missing: list[tuple[str, str, Path]] = []
    found: list[str] = []
    try:
        bindings = iter_dossier_bindings()
    except (DossierMissingError, DossierReferenceError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    for binding in bindings:
        if not binding.path.exists():
            missing.append((binding.sku, binding.name, binding.path))
        else:
            found.append(binding.sku)

    print(f"=== Dossier coverage ({len(found)}/{len(bindings)}) ===\n")

    if missing:
        print("MISSING dossiers:")
        for sku, name, path in missing:
            rel = path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path
            print(f"  ✗ {sku:<20} {name}")
            print(f"      expected: {rel}")
        print()
        print(f"summary: {len(missing)} dossier(s) missing of {len(bindings)} active SKUs")
        return 1

    print(f"summary: every active SKU has a dossier ({len(bindings)}/{len(bindings)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
