"""Per-product design dossier loader — shared by all four catalog readers.

Reads markdown dossiers from
`wordpress-theme/skyyrose-flagship/data/dossiers/{slug}.md` and parses them
into a structured dict consumed by:
  - skyyrose.core.catalog_loader.get_product_with_dossier()
  - nano_banana.catalog
  - skyyrose.elite_studio.catalog
  - skyyrose.elite_studio.agents.three_d_agent (RAS prompt construction)

Hard-fails on missing dossier (H1 from plan): the canonical CSV's thin
`branding_spec` column is NOT a fallback — adding one is a backdoor that lets
us forget to author.

The parser is deliberately tolerant of the markdown indentation inside list
items but strict on the section headings and the technique vocabulary (caught
upstream by scripts/validate_dossier.py).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path

from skyyrose.core.catalog_loader import CATALOG_CSV, read_catalog_rows

DOSSIERS_DIR = CATALOG_CSV.parent / "dossiers"


class DossierMissingError(FileNotFoundError):
    """Raised when a product's dossier markdown file is not found.

    The pipeline fails loudly rather than fall back to the thin CSV
    `branding_spec` column. Author the dossier before rendering.
    """


class DossierReferenceError(ValueError):
    """Raised when a catalog dossier reference is absent or unsafe."""


_DOSSIER_SLUG_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


@dataclass(frozen=True)
class DossierBinding:
    """The catalog-declared association between one SKU and one dossier."""

    sku: str
    name: str
    collection: str
    slug: str
    path: Path


@dataclass(frozen=True)
class FashionThemeDossierContext:
    """Read-only SOT projection for Fashion Theme Team consumers.

    The markdown dossier and catalog row remain authoritative. This object is
    deliberately a projection: consumers may use its facts for storytelling,
    merchandising, PDP, fit, care, and visual handoffs, but must not write it
    back or infer an alternate dossier from a product name or filename.
    """

    sku: str
    name: str
    collection: str
    dossier_slug: str
    dossier_path: str
    catalog: dict[str, str]
    dossier: dict
    validated_dossier: dict

    def to_dict(self) -> dict:
        """Return a detached JSON-ready copy for an external consumer."""
        return {
            "sku": self.sku,
            "name": self.name,
            "collection": self.collection,
            "dossier_slug": self.dossier_slug,
            "dossier_path": self.dossier_path,
            "catalog": dict(self.catalog),
            "dossier": dict(self.dossier),
            "validated_dossier": dict(self.validated_dossier),
        }


@dataclass
class Dossier:
    sku: str
    name: str
    collection: str
    slug: str
    garment_type_lock: str
    branding_block: str
    negative_block: str
    scene_pose: str = ""
    scene_setting: str = ""
    logo_reference: str = ""
    extra_logos: list[str] = field(default_factory=list)
    reference_image: str = ""
    extra_references: list[str] = field(default_factory=list)
    raw: str = field(default="", repr=False)

    def to_dict(self) -> dict:
        return {
            "sku": self.sku,
            "name": self.name,
            "collection": self.collection,
            "slug": self.slug,
            "garment_type_lock": self.garment_type_lock,
            "branding_block": self.branding_block,
            "negative_block": self.negative_block,
            "scene_pose": self.scene_pose,
            "scene_setting": self.scene_setting,
            "logo_reference": self.logo_reference,
            "extra_logos": list(self.extra_logos),
            "reference_image": self.reference_image,
            "extra_references": list(self.extra_references),
        }


def _parse_frontmatter(text: str) -> tuple[dict[str, str], str]:
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
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fm[key.strip()] = value.strip().strip('"').strip("'")
    return fm, rest


# Compiled once at module load. The list-key pattern matches a YAML key
# followed by one-or-more indented bullets; the item pattern then extracts
# each bullet's value from the captured block.
_FRONTMATTER_LIST_KEY_RE = re.compile(
    r"^(\w[\w\-]*):\s*\n((?:[ \t]+-\s+.+\n?)+)",
    re.MULTILINE,
)
_FRONTMATTER_LIST_ITEM_RE = re.compile(
    r"^[ \t]+-\s+(.+?)\s*$",
    re.MULTILINE,
)


def _parse_frontmatter_lists(text: str) -> dict[str, list[str]]:
    """Extract YAML-list-style frontmatter values that `_parse_frontmatter` skips.

    Matches the canonical pattern:
        key:
          - item-one
          - item-two

    Returns a dict mapping key → list of stripped item strings. Keys whose
    values are scalar (already captured by `_parse_frontmatter`) are not
    returned here.
    """
    if not text.startswith("---"):
        return {}
    end = text.find("\n---", 3)
    if end == -1:
        return {}
    block = text[3:end]
    out: dict[str, list[str]] = {}
    for match in _FRONTMATTER_LIST_KEY_RE.finditer(block):
        key = match.group(1)
        bullets_block = match.group(2)
        items = [
            m.group(1).strip().strip('"').strip("'")
            for m in _FRONTMATTER_LIST_ITEM_RE.finditer(bullets_block)
        ]
        if items:
            out[key] = items
    return out


def _extract_section(body: str, heading_pattern: str) -> str:
    pattern = rf"^##\s+{heading_pattern}.*?$(.*?)(?=^##\s|\Z)"
    match = re.search(pattern, body, re.MULTILINE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_garment_lock(body: str) -> str:
    match = re.search(r"\*\*Garment type lock:\*\*\s*(.+?)(?:\n\n|\n##|\Z)", body, re.DOTALL)
    return match.group(1).strip() if match else ""


def _extract_scene_field(scene_section: str, label: str) -> str:
    pattern = rf"\*\*{re.escape(label)}:\*\*\s*(.+?)(?:\n-|\n\n|\Z)"
    match = re.search(pattern, scene_section, re.DOTALL)
    return match.group(1).strip() if match else ""


def parse_dossier_markdown(text: str) -> Dossier:
    """Parse a dossier markdown string into a structured Dossier object."""
    fm, body = _parse_frontmatter(text)
    fm_lists = _parse_frontmatter_lists(text)

    branding_section = _extract_section(body, r"Branding")
    negative_section = _extract_section(body, r"Negative")
    scene_section = _extract_section(body, r"Scene direction")

    garment_lock = _extract_garment_lock(body)

    return Dossier(
        sku=fm.get("sku", ""),
        name=fm.get("name", ""),
        collection=fm.get("collection", ""),
        slug=fm.get("slug", ""),
        garment_type_lock=garment_lock,
        branding_block=branding_section,
        negative_block=negative_section,
        scene_pose=_extract_scene_field(scene_section, "Pose"),
        scene_setting=_extract_scene_field(scene_section, "Setting"),
        logo_reference=fm.get("logo_reference", ""),
        extra_logos=fm_lists.get("extra_logos", []),
        reference_image=fm.get("reference_image", ""),
        extra_references=fm_lists.get("extra_references", []),
        raw=text,
    )


def dossier_path(slug: str, dossiers_dir: Path | None = None) -> Path:
    """Resolve a declared dossier slug to a safe markdown path.

    Slugs are identifiers, not paths. Rejecting separators and dot segments
    keeps every consumer inside the canonical dossier directory.
    """
    normalized_slug = slug.strip()
    if not _DOSSIER_SLUG_RE.fullmatch(normalized_slug):
        raise DossierReferenceError(
            f"Invalid dossier_slug {slug!r}; use lowercase letters, digits, and hyphens only."
        )
    base = (dossiers_dir or DOSSIERS_DIR).resolve()
    path = (base / f"{normalized_slug}.md").resolve()
    try:
        path.relative_to(base)
    except ValueError as exc:  # defensive: slug validation above should prevent this
        raise DossierReferenceError(f"Dossier path escapes canonical directory: {slug!r}") from exc
    return path


def dossier_binding_for_row(
    row: Mapping[str, str], dossiers_dir: Path | None = None
) -> DossierBinding:
    """Build one catalog-declared dossier binding without fallback inference."""
    sku = (row.get("sku") or "").strip()
    if not sku:
        raise DossierReferenceError("Catalog row has no SKU for dossier resolution.")
    slug = (row.get("dossier_slug") or "").strip()
    if not slug:
        raise DossierMissingError(
            f"SKU {sku!r} has no dossier_slug in {CATALOG_CSV}. "
            "Add the declared dossier_slug before loading."
        )
    return DossierBinding(
        sku=sku,
        name=(row.get("name") or "").strip(),
        collection=(row.get("collection") or "").strip(),
        slug=slug,
        path=dossier_path(slug, dossiers_dir),
    )


def iter_dossier_bindings(
    rows: Iterable[Mapping[str, str]] | None = None, dossiers_dir: Path | None = None
) -> tuple[DossierBinding, ...]:
    """Return every SKU binding, preserving intentional shared dossier slugs."""
    source_rows = read_catalog_rows() if rows is None else rows
    return tuple(dossier_binding_for_row(row, dossiers_dir) for row in source_rows)


def dossier_binding_for_sku(sku: str, dossiers_dir: Path | None = None) -> DossierBinding:
    """Resolve a SKU through the canonical CSV, never from a guessed filename."""
    normalized_sku = sku.strip()
    for row in read_catalog_rows():
        if row.get("sku") == normalized_sku:
            return dossier_binding_for_row(row, dossiers_dir)
    raise KeyError(f"SKU {sku!r} not found in {CATALOG_CSV}")


@cache
def load_dossier(slug: str, dossiers_dir: Path | None = None) -> Dossier:
    """Load and parse a dossier by slug. Raises DossierMissingError if absent.

    Memoized: callers should treat the returned Dossier as read-only —
    mutating fields mutates the shared cache.
    """
    path = dossier_path(slug, dossiers_dir)
    if not path.exists():
        raise DossierMissingError(
            f"No dossier at {path}. Author the dossier before rendering. "
            f"CSV branding_spec is not a fallback."
        )
    dossier = parse_dossier_markdown(path.read_text(encoding="utf-8"))
    if not dossier.slug:
        dossier.slug = slug
    return dossier


def get_product_with_dossier(sku: str) -> dict:
    """Return the canonical CSV row for `sku` merged with its parsed dossier.

    Raises:
        KeyError if SKU is not in the canonical CSV.
        DossierMissingError if the SKU's dossier file does not exist.
    """
    binding = dossier_binding_for_sku(sku)
    row = next(row for row in read_catalog_rows() if row.get("sku") == binding.sku)
    dossier = load_dossier(binding.slug)
    return {**row, "dossier": dossier.to_dict(), "_dossier": dossier}


def load_fashion_theme_dossier_context(sku: str) -> FashionThemeDossierContext:
    """Load validated, provenance-labelled facts for Fashion Theme Team.

    Importing the schema lazily avoids a module cycle while ensuring external
    design consumers receive validated facts rather than raw markdown text.
    """
    binding = dossier_binding_for_sku(sku)
    row = next(row for row in read_catalog_rows() if row.get("sku") == binding.sku)
    dossier = load_dossier(binding.slug)
    from skyyrose.core.dossier_schema import DossierSchema

    validated = DossierSchema.from_raw(dossier).model_dump(mode="json")
    try:
        source_path = str(binding.path.relative_to(CATALOG_CSV.parents[3]))
    except ValueError:
        source_path = str(binding.path)
    return FashionThemeDossierContext(
        sku=binding.sku,
        name=binding.name,
        collection=binding.collection,
        dossier_slug=binding.slug,
        dossier_path=source_path,
        catalog=dict(row),
        dossier=dossier.to_dict(),
        validated_dossier=validated,
    )


__all__ = [
    "DOSSIERS_DIR",
    "Dossier",
    "DossierBinding",
    "DossierMissingError",
    "DossierReferenceError",
    "FashionThemeDossierContext",
    "dossier_path",
    "dossier_binding_for_row",
    "dossier_binding_for_sku",
    "iter_dossier_bindings",
    "parse_dossier_markdown",
    "load_dossier",
    "load_fashion_theme_dossier_context",
    "get_product_with_dossier",
]
