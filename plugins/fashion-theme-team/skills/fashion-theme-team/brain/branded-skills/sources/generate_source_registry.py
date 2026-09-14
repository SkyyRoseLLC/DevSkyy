#!/usr/bin/env python3
"""Build the deterministic registry for the imported branded-skill sources."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

DOMAIN_LABELS = {
    "branding-design": "brand_identity_and_design",
    "content-copywriting": "content_and_copywriting",
    "e-commerce-products": "ecommerce_products_and_operations",
    "email-marketing-automation": "email_and_lifecycle_marketing",
    "sales-funnels": "sales_strategy_and_conversion",
    "seo-search": "search_and_discovery",
    "social-media": "social_media_and_community",
}

HIGH_TERMS = {
    "brand",
    "branding",
    "fashion",
    "merch",
    "product",
    "collection",
    "marketplace",
    "ecommerce",
    "e-commerce",
    "photography",
    "visual",
    "packaging",
    "unboxing",
    "influencer",
    "instagram",
    "pinterest",
    "loyalty",
    "cart",
    "checkout",
    "launch",
}
LOW_TERMS = {
    "real-estate",
    "property",
    "rental",
    "lease",
    "food",
    "restaurant",
    "recipe",
    "author",
    "book",
    "podcast",
    "coaching",
    "donation",
    "nonprofit",
    "food-truck",
}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extract_frontmatter(text: str) -> tuple[str | None, dict[str, str], str]:
    if not text.startswith("---\n"):
        return None, {}, "absent"
    end = text.find("\n---\n", 4)
    if end < 0:
        return None, {}, "malformed"
    raw = text[: end + 5]
    parsed: dict[str, str] = {}
    for key in ("name", "description", "allowed-tools"):
        match = re.search(rf"(?m)^{re.escape(key)}:\s*(.+)$", raw)
        if match:
            value = match.group(1).strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            parsed[key.replace("-", "_")] = value
    return raw, parsed, "present"


def applicability(domain: str, slug: str, description: str) -> tuple[str, list[str]]:
    haystack = f"{slug} {description}".lower()
    high_hits = sorted(term for term in HIGH_TERMS if term in haystack)
    low_hits = sorted(term for term in LOW_TERMS if term in haystack)
    if low_hits and not high_hits:
        return "low", [
            "adjacent vertical; requires fashion-specific adaptation",
            *low_hits,
        ]
    if domain == "branding-design" or len(high_hits) >= 2:
        return "high", ["direct fashion-commerce or brand-system use", *high_hits]
    if domain in {"e-commerce-products", "seo-search", "social-media"} or high_hits:
        return "medium", [
            "transferable to fashion commerce with canon and SOT controls",
            *high_hits,
        ]
    return "conditional", [
        "general commercial method; use only when routed by an approved fashion task"
    ]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--authorities", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--as-of", required=True)
    args = parser.parse_args()

    source_root = args.source_root.resolve()
    files = sorted(source_root.glob("*/*/SKILL.md"), key=lambda path: path.as_posix())
    authorities = json.loads(args.authorities.read_text(encoding="utf-8"))
    records = []
    unknowns = [
        {
            "authority_requirement_id": entry["id"],
            "field": "claim_specific_current_authority_capture",
            "status": "UNVERIFIED",
            "reason": "Authority requirements are registered, but current content must be retrieved and hashed at claim time.",
        }
        for entry in authorities["domains"]
    ]
    for path in files:
        data = path.read_bytes()
        text = data.decode("utf-8")
        relative = path.relative_to(source_root).as_posix()
        domain, slug, _ = relative.split("/")
        raw, parsed, frontmatter_status = extract_frontmatter(text)
        rating, reasons = applicability(domain, slug, parsed.get("description", ""))
        if frontmatter_status != "present":
            unknowns.append(
                {
                    "source_path": str(path),
                    "field": "frontmatter",
                    "status": frontmatter_status,
                }
            )
        if not parsed.get("name"):
            unknowns.append(
                {
                    "source_path": str(path),
                    "field": "frontmatter.name",
                    "status": "UNKNOWN",
                }
            )
        records.append(
            {
                "id": f"bundle:{domain}/{slug}",
                "source_path": relative,
                "source_path_relative_to_bundle": relative,
                "sha256": sha256(data),
                "byte_length": len(data),
                "frontmatter": {
                    "status": frontmatter_status,
                    "raw_exact": raw,
                    "raw_sha256": (sha256(raw.encode("utf-8")) if raw is not None else None),
                    "parsed_non_authoritative": parsed,
                },
                "classification": {
                    "bundle_domain": domain,
                    "normalized_domain": DOMAIN_LABELS[domain],
                    "fashion_applicability": rating,
                    "rationale": reasons,
                    "use_constraint": "Repository canon, catalog/product SOT, imagery provenance, current primary authority, and human approval boundaries override this source.",
                },
                "authority_requirement_id": f"authority:{domain}",
                "claim_status": "LOCAL_SOURCE_INVENTORIED_NOT_FACT_VERIFIED",
            }
        )

    domain_counts = dict.fromkeys(DOMAIN_LABELS, 0)
    applicability_counts: dict[str, int] = {}
    for record in records:
        domain_counts[record["classification"]["bundle_domain"]] += 1
        key = record["classification"]["fashion_applicability"]
        applicability_counts[key] = applicability_counts.get(key, 0) + 1
    manifest_material = "\n".join(
        f'{record["source_path"]}\t{record["sha256"]}' for record in records
    ).encode()
    registry = {
        "$schema": "./source-registry.schema.json",
        "registry_version": "1.0.0",
        "as_of": args.as_of,
        "source_root": "vendor/branded-skills",
        "inventory_rule": "Exactly two directory levels below source_root and named SKILL.md; lexical POSIX path order.",
        "source_count": len(records),
        "source_manifest_sha256": sha256(manifest_material),
        "counts": {
            "by_domain": domain_counts,
            "by_fashion_applicability": dict(sorted(applicability_counts.items())),
        },
        "policy": {
            "source_role": "Imported methods only; never an authority for brand, product, legal, platform, standards, performance, or conversion claims.",
            "external_claim_rule": "Use current authenticated HTTPS primary sources named by the matching domain requirement. If unavailable, stale, contradictory, or out of scope, mark the claim UNVERIFIED.",
            "identity_rule": "Identity is established by exact path, bytes, SHA-256, and exact raw frontmatter; filenames and parsed labels alone are non-authoritative.",
            "no_uplift_claims": "Marketing or design outcomes remain EXPERIMENT unless candidate-bound measurement proves them.",
        },
        "authority_requirements": authorities,
        "sources": records,
        "unknowns": unknowns,
    }
    args.output.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
