#!/usr/bin/env python3
"""Generate the [data-collection] token region of design-tokens.css from identity.json.

Also generates the GENERATED:global-fonts region in :root from data/brand/typography.json.

Writes ONLY between the GENERATED:* START/END markers. The rest of the CSS is
hand-authored and untouched. Run after editing any identity.json or typography.json.
The verifier asserts the live region matches a fresh generation.

Emits the tokens the theme consumes: accent, accent-rgb (computed), accent-dark,
secondary, plus the 2-role font model (font-script = the collection's matched
display font; font-caps = Cinzel; font-display aliases font-caps for the 15 legacy
consumers so interior headings stay legible and uniform). font-gothic is dropped
(no consumers). bg/text inherit from :root (unchanged behaviour).
"""

import json
import os
import re
import sys
from pathlib import Path

DATA = Path(__file__).resolve().parent
sys.path.insert(0, str(DATA))
import sot_common  # noqa: E402

# CSS path is overridable via SKYYROSE_TOKENS_CSS so tests can run the generator
# against a private tmp copy (hermetic isolation). Unset -> the real theme file.
_CSS_OVERRIDE = os.environ.get("SKYYROSE_TOKENS_CSS")
CSS = Path(_CSS_OVERRIDE) if _CSS_OVERRIDE else DATA.parent / "assets/css/design-tokens.css"
START = "/* GENERATED:collection-tokens START"
END = "/* GENERATED:collection-tokens END */"

# Global-fonts region markers (in :root)
GF_START = "/* GENERATED:global-fonts START"
GF_END = "/* GENERATED:global-fonts END */"

# Typography SOT path
TYPOGRAPHY_JSON = DATA / "brand" / "typography.json"


def hex_to_rgb(hex_: str) -> str:
    h = hex_.lstrip("#")
    return f"{int(h[0:2], 16)}, {int(h[2:4], 16)}, {int(h[4:6], 16)}"


def css_hex(value: str) -> str:
    """Spell a hex color as Prettier does without changing its color value."""
    value = value.lower()
    if re.fullmatch(r"#[0-9a-f]{6}", value) and all(
        value[index] == value[index + 1] for index in (1, 3, 5)
    ):
        return "#" + value[1::2]
    return value


def css_font_stack(value: str) -> str:
    """Use double-quoted CSS strings while preserving existing escape sequences."""
    return re.sub(
        r"'((?:\\.|[^'\\])*)'",
        lambda match: '"' + match.group(1).replace('"', r"\"") + '"',
        value,
    )


def block(ident: dict) -> str:
    p, f = ident["palette"], ident["fonts"]
    return (
        f'[data-collection="{ident["slug"]}"] {{\n'
        f"  --skyyrose-accent: {css_hex(p['accent'])};\n"
        f"  --skyyrose-accent-rgb: {hex_to_rgb(p['accent'])};\n"
        f"  --skyyrose-accent-dark: {css_hex(p['accent_dark'])};\n"
        f"  --skyyrose-secondary: {css_hex(p['secondary'])};\n"
        f"  --skyyrose-font-script: {json.dumps(f['script']['family'], ensure_ascii=False)}, cursive;\n"
        f"  --skyyrose-font-caps: {json.dumps(f['caps']['family'], ensure_ascii=False)}, serif;\n"
        f"  --skyyrose-font-display: var(--skyyrose-font-caps);\n"
        f"}}\n"
    )


def emit_global_fonts() -> str:
    """Emit formatter-stable declarations from the unchanged typography SOT."""
    typo = json.loads(TYPOGRAPHY_JSON.read_text())
    universal = typo["universal"]
    return (
        "".join(
            f"  --skyyrose-font-{role}: {css_font_stack(universal[role]['stack'])};\n"
            for role in ("display", "body", "mono", "ui")
        )
        + "  "
    )  # The closing marker remains indented inside :root.


def regen_region(css: str, start_marker: str, end_marker: str, new_body: str) -> tuple[str, bool]:
    """Replace content between start_marker line-end and end_marker with new_body.

    Returns (new_css, ok). ok=False means markers missing or out of order.
    """
    if start_marker not in css or end_marker not in css:
        return css, False
    s = css.index(start_marker)
    e = css.index(end_marker)
    if e < s:
        return css, False
    line_end = css.index("\n", s) + 1
    return css[:line_end] + new_body + css[e:], True


def main() -> int:
    css = CSS.read_text()

    # --- 1. Regenerate global-fonts region ---
    if GF_START not in css or GF_END not in css:
        print(
            f"ERROR: {CSS} is missing the GENERATED:global-fonts markers "
            f"('{GF_START.strip()}' ... '{GF_END.strip()}') — cannot regenerate the region.",
            file=sys.stderr,
        )
        return 1
    gf_s = css.index(GF_START)
    gf_e = css.index(GF_END)
    if gf_e < gf_s:
        print(
            f"ERROR: {CSS} has the GENERATED:global-fonts END marker before START "
            f"— cannot regenerate the region.",
            file=sys.stderr,
        )
        return 1
    css, ok = regen_region(css, GF_START, GF_END, emit_global_fonts())
    if not ok:
        print(
            f"ERROR: {CSS} global-fonts region regen failed unexpectedly.",
            file=sys.stderr,
        )
        return 1
    print("design-tokens.css: regenerated global-fonts :root block")

    # --- 2. Regenerate collection-tokens region ---
    if START not in css or END not in css:
        print(
            f"ERROR: {CSS} is missing the GENERATED:collection-tokens markers "
            f"('{START.strip()}' ... '{END.strip()}') — cannot regenerate the region.",
            file=sys.stderr,
        )
        return 1
    s = css.index(START)
    e = css.index(END)
    if e < s:
        print(
            f"ERROR: {CSS} has the GENERATED:collection-tokens END marker before START "
            f"— cannot regenerate the region.",
            file=sys.stderr,
        )
        return 1
    idents = sot_common.load_identity()
    body = "\n".join(block(idents[slug]) for slug in sorted(idents))
    line_end = css.index("\n", s) + 1
    css = css[:line_end] + body + css[e:]
    CSS.write_text(css)
    print(f"design-tokens.css: regenerated {len(idents)} collection blocks")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
