#!/usr/bin/env python3
"""Validate XML/SVG and apply only semantics-preserving text normalization.

General-purpose XML serializers can reorder namespaces, attributes, or mixed
content.  That is unsafe for hand-authored SVG and configuration files.  This
formatter therefore keeps the document text intact apart from normalizing line
endings to LF and ensuring exactly one final newline, while Expat verifies that
the result remains well-formed and rejects external entity expansion.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from xml.parsers import expat


def _validate_xml(content: bytes, path: Path) -> None:
    parser = expat.ParserCreate()

    def reject_external_entity(*_args: object) -> int:
        raise ValueError(f"external entities are not allowed in {path}")

    parser.ExternalEntityRefHandler = reject_external_entity
    try:
        parser.Parse(content, True)
    except (expat.ExpatError, ValueError) as exc:
        raise ValueError(f"invalid XML in {path}: {exc}") from exc


def format_markup(path: Path) -> bool:
    original = path.read_bytes()
    normalized = original.replace(b"\r\n", b"\n").replace(b"\r", b"\n").rstrip(b"\n") + b"\n"
    _validate_xml(normalized, path)
    if normalized == original:
        return False
    path.write_bytes(normalized)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()

    try:
        for path in args.paths:
            format_markup(path)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
