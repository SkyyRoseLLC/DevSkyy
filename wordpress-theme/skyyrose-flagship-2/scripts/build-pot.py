#!/usr/bin/env python3
"""Extract literal PHP translations into the theme POT catalog."""

from __future__ import annotations

import ast
import re
import sys
from collections import defaultdict
from pathlib import Path

THEME = Path(__file__).resolve().parents[1]
OUTPUT = THEME / "languages/skyyrose-flagship-2.pot"
CALL = re.compile(
    r"(?P<fn>__|_e|esc_html__|esc_html_e|esc_attr__|esc_attr_e)\(\s*"
    r"(?P<quote>'|\")(?P<message>(?:\\.|(?!\2).)*)\2\s*,\s*"
    r"(?P<domain>'|\")skyyrose-flagship-2\4",
    re.DOTALL,
)


def theme_version() -> str:
    """Read the public theme version from the canonical style header."""
    style = (THEME / "style.css").read_text(encoding="utf-8")
    match = re.search(r"^Version:\s*(\S+)\s*$", style, re.MULTILINE)
    if not match:
        raise ValueError("style.css has no Version header")
    return match.group(1)


def decode(message: str, quote: str) -> str:
    try:
        return ast.literal_eval(f"{quote}{message}{quote}")
    except (SyntaxError, ValueError):
        return message.replace(r"\'", "'").replace(r'\"', '"')


def po_quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")
    return f'"{escaped}"'


def collect() -> dict[str, list[str]]:
    messages: dict[str, list[str]] = defaultdict(list)
    for source in sorted(THEME.rglob("*.php")):
        if any(part in {"node_modules", "vendor"} for part in source.parts):
            continue
        text = source.read_text(encoding="utf-8")
        for match in CALL.finditer(text):
            message = decode(match.group("message"), match.group("quote"))
            line = text.count("\n", 0, match.start()) + 1
            messages[message].append(f"{source.relative_to(THEME)}:{line}")
    return dict(sorted(messages.items()))


def render(messages: dict[str, list[str]]) -> str:
    version = theme_version()
    header = [
        'msgid ""',
        'msgstr ""',
        f'"Project-Id-Version: SkyyRose Flagship 2 {version}\\n"',
        '"Report-Msgid-Bugs-To: https://skyyrose.co/contact/\\n"',
        '"POT-Creation-Date: 2026-08-14 00:00+0000\\n"',
        '"MIME-Version: 1.0\\n"',
        '"Content-Type: text/plain; charset=UTF-8\\n"',
        '"Content-Transfer-Encoding: 8bit\\n"',
        '"X-Domain: skyyrose-flagship-2\\n"',
        "",
    ]
    entries = list(header)
    for message, references in messages.items():
        entries.append(f"#: {' '.join(references)}")
        entries.append(f"msgid {po_quote(message)}")
        entries.append('msgstr ""')
        entries.append("")
    return "\n".join(entries)


def main() -> int:
    messages = collect()
    rendered = render(messages)
    if "--check" in sys.argv:
        if not OUTPUT.is_file() or OUTPUT.read_text(encoding="utf-8") != rendered:
            print("Translation catalog is stale. Run this script without --check.", file=sys.stderr)
            return 1
        print(f"Translation catalog is current ({len(messages)} messages).")
        return 0
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(rendered, encoding="utf-8")
    print(f"Wrote {OUTPUT} ({len(messages)} messages).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
