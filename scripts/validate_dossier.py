"""Validate per-product design dossiers.

Reads dossier markdown files under
`data/dossiers/`, checks YAML frontmatter,
controlled-vocabulary fields, and contradiction rules, then exits 0 (all pass)
or 1 (any fail) with a structured report.

Run from repo root:
    python scripts/validate_dossier.py                       # all dossiers
    python scripts/validate_dossier.py path/to/file.md ...   # specific files

Exits non-zero on any failure — suitable for CI and pre-commit hooks.
"""

from __future__ import annotations

import csv
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from skyyrose.core.dossier_loader import parse_dossier_markdown  # noqa: E402
from skyyrose.core.dossier_schema import (  # noqa: E402
    DossierSchema,
    DossierSchemaError,
    parse_branding_regions,
)

CSV_PATH = REPO_ROOT / "data/skyyrose-catalog.csv"
DOSSIERS_DIR = REPO_ROOT / "data/dossiers"

ALLOWED_COLLECTIONS = {"black-rose", "love-hurts", "signature", "kids-capsule"}

ALLOWED_TECHNIQUES = {
    "embossed",
    "debossed",
    "embroidered",
    "embroidered-patch",
    "printed",
    "screen-print",
    "sublimated",
    "stitched",
    "patch",
    "woven-label",
    "puff-print",
    "heat-transfer",
    "laser-engraved",
    "silicone",
    "silicone-appliqué",
    "tackle-twill",
}

GARMENT_TYPE_KEYWORDS = {
    "crewneck",
    "hoodie",
    "jersey",
    "jacket",
    "bomber",
    "sherpa",
    "shirt",
    "tee",
    "t-shirt",
    "tank",
    "dress",
    "joggers",
    "sweatpants",
    "pants",
    "shorts",
    "hat",
    "beanie",
    "cap",
    "windbreaker",
    "bag",
    "fanny-pack",
    "pack",
    "accessory",
    "set",
    "sweatshirt",
    "socks",
}


@dataclass
class DossierResult:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors


def load_active_skus() -> set[str]:
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"canonical CSV not found at {CSV_PATH}")
    with CSV_PATH.open() as f:
        return {row["sku"] for row in csv.DictReader(f)}


def parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
    """Parse a tiny YAML-like frontmatter block. We deliberately do NOT depend
    on PyYAML so this script can run in any minimal Python environment."""
    if not text.startswith("---"):
        return {}, text
    end = text.find("\n---", 3)
    if end == -1:
        return {}, text
    block = text[3:end].strip()
    rest = text[end + 4 :].lstrip("\n")
    fm: dict[str, str] = {}
    for line in block.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        fm[key.strip()] = value.strip().strip('"').strip("'")
    return fm, rest


def extract_section(body: str, heading_pattern: str) -> str | None:
    """Return the body of a `## {heading_pattern}` section up to the next `##`."""
    pattern = rf"^##\s+{heading_pattern}.*?$(.*?)(?=^##\s|\Z)"
    match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
    return match.group(1) if match else None


def extract_subsection(section: str, heading: str) -> str | None:
    pattern = rf"^###\s+{re.escape(heading)}.*?$(.*?)(?=^###\s|^##\s|\Z)"
    match = re.search(pattern, section, re.MULTILINE | re.DOTALL)
    return match.group(1) if match else None


@dataclass
class BrandingEntry:
    region: str
    dimensions: str
    description: str
    technique: str
    color: str


def parse_branding_entries(section: str) -> list[BrandingEntry]:
    """Use the same parser as SOT compilation and paid render preflight."""
    return [
        BrandingEntry(
            region=region.region,
            dimensions=region.dimensions or "unspecified",
            description=region.description,
            technique=region.technique,
            color=region.color_named or "source-reference exact color",
        )
        for region in parse_branding_regions(section)
    ]


def recognized_techniques(value: str) -> tuple[str, ...]:
    """Return every controlled-vocabulary technique named by a detailed value.

    Dossiers may append construction details such as ``(sewn-on hardware)`` or
    combine secondary processes with ``+``/``/``.  The full value must remain
    available to render prompts, while this validator confirms that it contains
    a machine-recognized primary technique.  Longest-first matching prevents
    ``embroidered-patch`` from collapsing to the broader ``patch`` token.
    """

    normalized = value.casefold().strip()
    recognized: list[str] = []
    for technique in sorted(ALLOWED_TECHNIQUES, key=len, reverse=True):
        pattern = rf"(?<![\w-]){re.escape(technique.casefold())}(?![\w-])"
        if re.search(pattern, normalized):
            recognized.append(technique)
    return tuple(recognized)


def canonical_technique(value: str) -> str:
    """Return the longest recognized technique, or an empty string."""

    techniques = recognized_techniques(value)
    return techniques[0] if techniques else ""


def validate_dossier(path: Path, active_skus: set[str]) -> DossierResult:
    result = DossierResult(path=path)

    if path.name == "_template.md":
        return result

    text = path.read_text(encoding="utf-8")
    fm, body = parse_frontmatter(text)

    for required in ("sku", "name", "collection"):
        if required not in fm or not fm[required]:
            result.errors.append(f"frontmatter missing required field: {required}")

    if "sku" in fm and fm["sku"] not in active_skus:
        result.errors.append(
            f"sku '{fm['sku']}' is not in canonical CSV ({CSV_PATH.relative_to(REPO_ROOT)})"
        )

    if "collection" in fm and fm["collection"] not in ALLOWED_COLLECTIONS:
        result.errors.append(
            f"collection '{fm['collection']}' is not one of {sorted(ALLOWED_COLLECTIONS)}"
        )

    expected_slug = path.stem
    if "sku" in fm:
        # Slug should be derived from the name, not the SKU. We don't recompute
        # slug → we just confirm filename matches a sensible pattern.
        if "/" in expected_slug or expected_slug.startswith("_"):
            result.errors.append(f"unexpected dossier filename '{path.name}'")

    lock_match = re.search(r"\*\*Garment type lock:\*\*\s*(.+?)(?:\n\n|\Z)", body, re.DOTALL)
    if not lock_match:
        result.errors.append("missing **Garment type lock:** line")
    else:
        lock_text = lock_match.group(1).lower()
        has_garment = any(g in lock_text for g in GARMENT_TYPE_KEYWORDS)
        has_not = "not" in lock_text or "no " in lock_text
        if not has_garment:
            result.errors.append(
                "garment type lock does not mention any recognized garment type "
                f"(expected one of {sorted(GARMENT_TYPE_KEYWORDS)})"
            )
        if not has_not:
            result.errors.append(
                "garment type lock must include at least one explicit 'NOT a {...}' disambiguator"
            )

    branding_section = extract_section(body, r"Branding")
    negative_section = extract_section(body, r"Negative")

    if not branding_section:
        result.errors.append("missing '## Branding' section")
    if not negative_section:
        result.errors.append("missing '## Negative' section")

    material_lock_version = fm.get("material_lock_version", "").strip()
    if material_lock_version:
        if material_lock_version != "v1":
            result.errors.append(
                f"unsupported material_lock_version {material_lock_version!r}; expected 'v1'"
            )
        else:
            raw_dossier = parse_dossier_markdown(text)
            if not raw_dossier.slug:
                raw_dossier.slug = path.stem
            try:
                schema = DossierSchema.from_raw(raw_dossier)
            except DossierSchemaError as exc:
                result.errors.append(f"material-lock schema invalid: {exc}")
            else:
                unlocked = [
                    region.region for region in schema.branding if region.material_lock is None
                ]
                if unlocked:
                    result.errors.append(
                        "material_lock_version v1 requires material/construction/surface/"
                        "attachment/verify/reject on every parsed branding region; missing: "
                        + ", ".join(unlocked)
                    )

    if branding_section:
        try:
            positive_entries = parse_branding_entries(branding_section)
        except DossierSchemaError as exc:
            result.errors.append(f"branding schema invalid: {exc}")
            positive_entries = []
        if not positive_entries:
            result.warnings.append(
                "branding section parsed zero entries — confirm format matches "
                "`- **region** (dimensions): description. **Technique:** X. **Color:** Y.`"
            )
        for entry in positive_entries:
            technique = canonical_technique(entry.technique)
            if not technique:
                result.errors.append(
                    f"technique '{entry.technique}' (region {entry.region}) "
                    "does not contain a recognized primary technique from "
                    f"controlled vocabulary {sorted(ALLOWED_TECHNIQUES)}"
                )
            for field_name, value in (
                ("dimensions", entry.dimensions),
                ("description", entry.description),
                ("color", entry.color),
            ):
                if not value or value.lower() in {"tbd", "todo", "n/a"}:
                    result.errors.append(
                        f"branding entry (region {entry.region}) has empty/placeholder {field_name}"
                    )

        if negative_section:
            # Skip exclusion phrases — "other than", "only at", "only on",
            # "except at", "except on" — these are restrictive negatives, not
            # contradictions ("no X other than at Y" means X is fine at Y, not
            # forbidden at Y).
            EXCLUSION_PHRASES = ("other than", "only at", "only on", "except at", "except on")
            for entry in positive_entries:
                techniques = recognized_techniques(entry.technique)
                if not techniques:
                    continue
                region = entry.region.lower()
                lines = negative_section.lower().splitlines()
                for line in lines:
                    if not line.strip().startswith("-"):
                        continue
                    if any(phrase in line for phrase in EXCLUSION_PHRASES):
                        continue
                    matching_techniques = [tech for tech in techniques if tech in line]
                    if matching_techniques and region in line:
                        result.errors.append(
                            "contradiction: positive lists technique "
                            f"'{entry.technique}' ({', '.join(matching_techniques)}) "
                            f"on region '{entry.region}', but negative also names that "
                            f"combination: '{line.strip().lstrip('- ').strip()}'"
                        )

    if negative_section and len(negative_section.strip()) < 30:
        result.warnings.append(
            "negative section is very short — list every conflation we want to prevent"
        )

    return result


def main(argv: list[str]) -> int:
    if not DOSSIERS_DIR.exists():
        print(f"error: dossiers directory not found at {DOSSIERS_DIR}", file=sys.stderr)
        return 2

    active_skus = load_active_skus()
    targets: list[Path]
    if len(argv) > 1:
        targets = [Path(p).resolve() for p in argv[1:]]
    else:
        targets = sorted(DOSSIERS_DIR.glob("*.md"))

    if not targets:
        print(f"error: no dossier files found under {DOSSIERS_DIR}", file=sys.stderr)
        return 2

    results = [validate_dossier(p, active_skus) for p in targets]

    failed = [r for r in results if not r.ok]
    warned = [r for r in results if r.warnings]

    for r in results:
        rel = r.path.relative_to(REPO_ROOT) if r.path.is_relative_to(REPO_ROOT) else r.path
        if r.errors:
            print(f"FAIL {rel}")
            for e in r.errors:
                print(f"  ✗ {e}")
        elif r.warnings:
            print(f"WARN {rel}")
        else:
            print(f"PASS {rel}")
        for w in r.warnings:
            print(f"  ⚠ {w}")

    print()
    print(f"summary: {len(results) - len(failed)}/{len(results)} passed", end="")
    if warned:
        print(f", {len(warned)} with warnings", end="")
    if failed:
        print(f", {len(failed)} FAILED", end="")
    print()

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
