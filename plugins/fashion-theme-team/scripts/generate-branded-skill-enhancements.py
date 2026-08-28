#!/usr/bin/env python3
"""Generate deterministic SkyyRose enhancement manifests for branded skills."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import sys
import tempfile
from pathlib import Path

DOMAIN_CONTRACTS = {
    "branding-design": {
        "route": "brand_identity_governance",
        "owners": ["brand-experience-architect", "fashion-brand-systems-researcher"],
        "sources": ["ftc-advertising-truth", "wipo-brand-ip", "w3c-wcag-22"],
        "outputs": [
            "brand_contract",
            "touchpoint_matrix",
            "approval_register",
            "evidence_map",
        ],
    },
    "content-copywriting": {
        "route": "fashion_content_copy",
        "owners": [
            "fashion-accessibility-content-engineer",
            "fashion-knowledge-curator",
        ],
        "sources": ["ftc-advertising-truth", "google-search-essentials", "w3c-wcag-22"],
        "outputs": [
            "content_contract",
            "channel_copy",
            "claim_register",
            "evidence_map",
        ],
    },
    "email-marketing-automation": {
        "route": "fashion_lifecycle_email",
        "owners": [
            "fashion-merchandising-conversion-architect",
            "ecommerce-growth-analytics-engineer",
        ],
        "sources": [
            "ftc-can-spam",
            "ftc-endorsements",
            "ico-direct-marketing",
            "w3c-wcag-22",
        ],
        "outputs": [
            "lifecycle_contract",
            "message_sequence",
            "consent_suppression_matrix",
            "measurement_plan",
        ],
    },
    "e-commerce-products": {
        "route": "fashion_product_operations",
        "owners": ["catalog-sot-integrator", "fashion-product-fit-returns-specialist"],
        "sources": [
            "woocommerce-docs",
            "ftc-advertising-truth",
            "ftc-mail-order-rule",
            "w3c-wcag-22",
        ],
        "outputs": [
            "product_contract",
            "commerce_state_matrix",
            "service_policy_map",
            "evidence_map",
        ],
    },
    "sales-funnels": {
        "route": "ethical_fashion_funnel",
        "owners": [
            "fashion-commerce-strategist",
            "fashion-merchandising-conversion-architect",
        ],
        "sources": [
            "ftc-dark-patterns",
            "ftc-advertising-truth",
            "woocommerce-docs",
            "w3c-wcag-22",
        ],
        "outputs": [
            "funnel_contract",
            "journey_state_matrix",
            "experiment_plan",
            "rollback_rule",
        ],
    },
    "seo-search": {
        "route": "fashion_search_discovery",
        "owners": [
            "fashion-knowledge-curator",
            "fashion-accessibility-content-engineer",
        ],
        "sources": [
            "google-search-essentials",
            "google-structured-data",
            "schema-org",
            "w3c-wcag-22",
        ],
        "outputs": [
            "search_contract",
            "query_intent_map",
            "metadata_schema",
            "verification_report",
        ],
    },
    "social-media": {
        "route": "fashion_social_campaigns",
        "owners": ["brand-experience-architect", "ecommerce-growth-analytics-engineer"],
        "sources": [
            "ftc-endorsements",
            "meta-business-docs",
            "tiktok-business-docs",
            "w3c-wcag-22",
        ],
        "outputs": [
            "campaign_contract",
            "channel_adaptations",
            "rights_disclosure_register",
            "measurement_plan",
        ],
    },
}

COMMON_HARD_FAILS = [
    "invented_product_price_inventory_material_fit_policy_review_or_urgency",
    "unapproved_logo_or_brand_ownership_change",
    "unverified_or_unlicensed_media",
    "unsupported_conversion_or_performance_claim",
    "missing_accessibility_fallback_or_reduced_motion_state",
    "hidden_sponsorship_affiliate_or_incentive_disclosure",
    "candidate_or_source_hash_mismatch",
    "production_write_without_fresh_explicit_approval",
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def parse_frontmatter(text: str) -> dict[str, str]:
    match = re.match(r"^---\n(.*?)\n---\n", text, flags=re.DOTALL)
    if not match:
        raise ValueError("missing YAML frontmatter")
    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line or line.startswith((" ", "\t")):
            continue
        key, value = line.split(":", 1)
        result[key.strip()] = value.strip().strip('"').strip("'")
    return result


def capability_profile(text: str, frontmatter: dict[str, str]) -> dict[str, object]:
    headings = [
        match.group(1).strip()
        for match in re.finditer(r"^#{1,4}\s+(.+?)\s*$", text, flags=re.MULTILINE)
    ]
    headings = list(dict.fromkeys(headings))
    inputs = [
        heading
        for heading in headings
        if re.search(r"input|requirement|context|before|discovery", heading, re.I)
    ]
    outputs = [
        heading
        for heading in headings
        if re.search(r"output|deliverable|template|format|example", heading, re.I)
    ]
    verification = [
        heading
        for heading in headings
        if re.search(r"verif|test|check|quality|accept|success|measure", heading, re.I)
    ]
    risk_patterns = {
        "commerce": r"product|price|cart|checkout|order|inventory|return|refund",
        "claims": r"claim|testimonial|guarantee|conversion|revenue|result",
        "personal-data": r"email|phone|customer|personal|audience|segment",
        "publication": r"publish|post|send|launch|outreach|campaign",
        "rights": r"copyright|license|rights|photo|image|influencer",
        "regulated": r"legal|compliance|consent|privacy|can-spam|disclosure",
    }
    risk_terms = [name for name, pattern in risk_patterns.items() if re.search(pattern, text, re.I)]
    tools = (
        re.split(r"[\s,]+", frontmatter.get("allowed-tools", "").strip())
        if frontmatter.get("allowed-tools")
        else []
    )
    material = json.dumps(
        {
            "headings": headings,
            "tools": tools,
            "inputs": inputs,
            "outputs": outputs,
            "verification": verification,
            "risks": risk_terms,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return {
        "headings": headings or ["source-body-without-markdown-headings"],
        "declared_tools": [tool.lower() for tool in tools if tool],
        "input_sections": inputs,
        "output_sections": outputs,
        "verification_sections": verification,
        "risk_terms": risk_terms,
        "profile_sha256": sha256_bytes(material.encode("utf-8")),
    }


def gap_records(
    profile: dict[str, object], domain: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    gaps: list[dict[str, str]] = []
    closures: dict[str, dict[str, str]] = {}

    def add(
        gap_id: str,
        evidence: str,
        impact: str,
        closure_id: str,
        control: str,
        acceptance: str,
    ) -> None:
        gaps.append(
            {
                "id": gap_id,
                "evidence": evidence,
                "impact": impact,
                "closure_id": closure_id,
            }
        )
        closures[closure_id] = {
            "id": closure_id,
            "control": control,
            "acceptance": acceptance,
        }

    headings = profile["headings"]
    risks = profile["risk_terms"]
    add(
        "missing-skyyrose-authority-binding",
        f"Source heading inventory ({len(headings)} headings) contains no approved SkyyRose canon or catalog-SOT binding.",
        "Generic guidance could override brand or product truth.",
        "bind-skyyrose-canon-and-sot",
        "Load repository canon, approved catalog/media SOT, and founder decisions before source execution.",
        "Evidence records exact authority paths and no output contradicts them.",
    )
    add(
        "missing-candidate-evidence-binding",
        f"Source verification sections observed: {profile['verification_sections'] or ['none']}.",
        "Outputs could be accepted without exact candidate proof or an independent verdict.",
        "bind-candidate-and-independent-review",
        "Require candidate, source, contract, and artifact hashes plus an independent reviewer.",
        "All evidence hashes recompute and reviewer identity differs from every builder and owner.",
    )
    add(
        "missing-authenticated-claim-sources",
        f"Detected risk terms: {risks or ['none']}; source has no registered current-authority capture contract.",
        "External, legal, platform, or outcome claims could be stale or unsupported.",
        "retrieve-pinned-primary-authority",
        "Resolve each material claim to a current host-pinned primary source and claim anchor.",
        "Every material external claim has publisher, final URL, retrieval time, content hash, anchor, and freshness result.",
    )
    if not profile["input_sections"]:
        add(
            "missing-explicit-input-contract",
            "No source heading matched input, requirements, context, discovery, or before-work semantics.",
            "Execution can silently invent missing task facts.",
            "require-typed-task-inputs",
            "Require task, candidate ID, brand canon, and applicable SOT; unknowns fail closed.",
            "Missing required inputs produce UNKNOWN or BLOCKED rather than inferred values.",
        )
    if not profile["output_sections"]:
        add(
            "missing-explicit-output-contract",
            "No source heading matched output, deliverable, template, format, or example semantics.",
            "Completion shape and downstream compatibility are ambiguous.",
            "require-versioned-output-contract",
            "Declare artifact type, required files, stable IDs, statuses, and evidence schema.",
            "Output validates and contains every declared stable ID and required file.",
        )
    if not profile["verification_sections"]:
        add(
            "missing-explicit-verification-section",
            "No source heading matched verification, test, check, quality, acceptance, success, or measurement semantics.",
            "The source can finish without executable acceptance evidence.",
            "require-executable-acceptance",
            "Run source hash, schema, candidate artifact, domain, and independent-review checks.",
            "All applicable checks pass; skipped, stale, missing, or unknown evidence blocks PASS.",
        )
    if "personal-data" in risks or domain in {
        "email-marketing-automation",
        "social-media",
    }:
        add(
            "privacy-consent-suppression-gap",
            "Source capability touches audience, email, phone, customer, community, or social data.",
            "Execution could retain data or contact people without valid consent and suppression controls.",
            "enforce-privacy-consent-suppression",
            "Minimize and redact data; verify purpose, consent, suppression, retention, and deletion controls.",
            "Evidence proves authorized data use and suppression before any external action.",
        )
    if "commerce" in risks or domain in {"e-commerce-products", "sales-funnels"}:
        add(
            "commerce-truth-and-dark-pattern-gap",
            "Source capability touches product, offer, price, checkout, order, inventory, return, or conversion behavior.",
            "Execution could invent commerce facts, urgency, proof, or unmeasured uplift.",
            "enforce-commerce-truth-and-experiment-gates",
            "Bind live commerce facts to WooCommerce/SOT and classify proposed uplift as an instrumented experiment.",
            "Price, stock, policy, CTA, and order states match the authoritative response; no false urgency or unsupported uplift claim exists.",
        )
    if "rights" in risks or domain in {
        "branding-design",
        "content-copywriting",
        "social-media",
    }:
        add(
            "rights-and-media-provenance-gap",
            "Source capability touches copyright, licensing, imagery, influencers, brand, or reusable content.",
            "Unverified ownership, consent, or media identity could enter a human-facing artifact.",
            "enforce-rights-and-media-provenance",
            "Require registry identity, rights record, consent/disclosure where applicable, and eyes-on media review.",
            "Every used asset and endorsement resolves to approved provenance; filenames alone never establish identity.",
        )
    return gaps, list(closures.values())


def canonical_contract_sha(record: dict[str, object]) -> str:
    material = copy.deepcopy(record)
    provenance = material["provenance"]
    assert isinstance(provenance, dict)
    provenance.pop("contract_sha256", None)
    return sha256_bytes(
        json.dumps(material, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode(
            "utf-8"
        )
    )


def record_for(
    skill_file: Path,
    source_root: Path,
    lifecycle_state: str,
    reviewed_at: str | None,
    reviewed_by: str | None,
) -> dict[str, object]:
    relative = skill_file.relative_to(source_root)
    domain = relative.parts[0]
    contract = DOMAIN_CONTRACTS.get(domain)
    if contract is None:
        raise ValueError(f"unsupported branded-skill domain: {domain}")
    raw = skill_file.read_bytes()
    text = raw.decode("utf-8")
    frontmatter = parse_frontmatter(text)
    skill_id = frontmatter.get("name") or relative.parent.name
    contract_id = f"skyyrose.branded-skill.{skill_id}"
    source_sha = sha256_bytes(raw)
    profile = capability_profile(text, frontmatter)
    gaps, closures = gap_records(profile, domain)
    record: dict[str, object] = {
        "schema_version": "1.0.0",
        "contract_id": contract_id,
        "source": {
            "skill_id": skill_id,
            "relative_path": relative.as_posix(),
            "sha256": source_sha,
            "inventory_id": f"bundle:{domain}/{skill_id}",
            "body_policy": "reference-only-never-copy",
        },
        "source_capability_profile": profile,
        "routing": {
            "trigger_summary": frontmatter.get(
                "description",
                f"Route {skill_id} tasks through the authenticated SkyyRose contract.",
            ),
            "brain_route_ids": [contract["route"]],
            "load_policy": "metadata-first-source-on-match",
            "tool_profile": {
                "default": ["Read", "Grep", "Glob", "Bash"],
                "expansions": [],
            },
        },
        "inputs": {
            "required": [
                {
                    "name": "task",
                    "type": "string",
                    "authority": "user",
                    "sensitive": False,
                },
                {
                    "name": "candidate_id",
                    "type": "string",
                    "authority": "repository",
                    "sensitive": False,
                },
                {
                    "name": "brand_canon",
                    "type": "artifact-ref",
                    "authority": "approved-SOT",
                    "sensitive": False,
                },
            ],
            "optional": [
                {
                    "name": "catalog_and_rights",
                    "type": "artifact-ref",
                    "authority": "approved-SOT",
                    "sensitive": False,
                },
                {
                    "name": "current_sources",
                    "type": "array",
                    "authority": "official-documentation",
                    "sensitive": False,
                },
            ],
            "unknown_policy": "discover-read-only-then-mark-UNKNOWN",
            "sensitive_data_policy": "minimize-redact-never-persist-secrets-or-customer-data",
        },
        "authority": {
            "precedence": [
                "repository-instructions-and-approved-SOT",
                "current-official-documentation",
                "dated-registered-research",
                "durable-brain-guidance",
                "inference-or-experiment",
            ],
            "claim_classes": [
                "OBSERVED",
                "APPROVED",
                "RECOMMENDED",
                "UNKNOWN",
                "DURABLE",
                "CURRENT",
                "BRAND_SPECIFIC",
                "INFERENCE",
                "EXPERIMENT",
            ],
            "approval_boundaries": [
                "new-dependency",
                "credential",
                "paid-api",
                "external-write",
                "upload",
                "push",
                "protected-branch-merge",
                "destructive-action",
                "staging-mutation",
                "deployment",
                "production-mutation",
            ],
            "prohibited": [
                "secret-disclosure",
                "fabricated-evidence",
                "gate-waiver",
                "unauthorized-production-mutation",
                "invented-product-fact",
                "invented-urgency",
                "self-certification",
            ],
        },
        "skyyrose_overlay": {
            "brand_contract": "../../brand/skyyrose-artifact-system.json",
            "application": "required-for-fashion-commerce-output",
            "non_negotiables": [
                "garment-first",
                "truthful-product-data",
                "one-rose-gold-accent",
                "oakland-editorial-thesis",
                "SOT-or-founder-approved-media",
                "no-fabricated-commerce-proof",
                "logo-off-recognition",
            ],
        },
        "gap_analysis": {
            "method": "deterministic-source-review-plus-domain-contract",
            "source_body_reviewed": True,
            "gaps": gaps,
            "closures": closures,
            "residual_unknowns": ["task-specific-facts-until-runtime-authority-retrieval"],
        },
        "output": {
            "artifact_type": f"{skill_id}-enhanced-deliverable",
            "schema_ref": "schemas/enhanced-skill-evidence.schema.json",
            "required_files": [f"{skill_id}-deliverable", f"{skill_id}-evidence"],
            "stable_ids": [skill_id, contract["route"]],
            "status_vocabulary": [
                "PASS",
                "FAIL",
                "BLOCKED",
                "NOT_APPLICABLE",
                "UNKNOWN",
            ],
        },
        "requirements": {
            "ethics": {
                "applicability": "REQUIRED",
                "rules": [
                    "No dark patterns, fabricated social proof, false scarcity, or unsupported outcome claims."
                ],
            },
            "commerce": {
                "applicability": "CONDITIONAL",
                "rules": [
                    "WooCommerce and approved SOT own product, price, stock, variation, cart, checkout, and order truth."
                ],
            },
            "accessibility": {
                "applicability": "REQUIRED",
                "rules": [
                    "Use WCAG 2.2 AA semantics, keyboard access, focus, reflow, contrast, reduced motion, and equivalent fallbacks where applicable."
                ],
            },
            "performance": {
                "applicability": "CONDITIONAL",
                "rules": [
                    "Declare candidate-specific budgets, measure them, and preserve a usable degraded mode."
                ],
            },
            "security": {
                "applicability": "REQUIRED",
                "rules": [
                    "Validate and escape boundaries, protect secrets, minimize permissions, and verify dependency provenance."
                ],
            },
            "privacy": {
                "applicability": "REQUIRED",
                "rules": [
                    "Minimize and redact data; document consent, purpose, retention, and rollback for measurement."
                ],
            },
        },
        "evidence": {
            "candidate_binding": "all-evidence-must-share-candidate-id",
            "required_classes": [
                "source",
                "deterministic",
                "accessibility",
                "security",
                "release",
            ],
            "freshness": "mutation-invalidates-dependent-evidence",
            "independent_review": "builder-cannot-approve-own-output",
        },
        "fallback": {
            "degraded_mode": "Return a schema-valid BLOCKED artifact with actionable missing-authority notes and no invented facts.",
            "missing_input": "continue-independent-work-and-mark-dependent-claims-UNKNOWN-or-BLOCKED",
            "unavailable_tool": "use-documented-safe-fallback-or-BLOCKED-never-fabricate",
            "transient_retry": {"maximum": 2, "backoff": "bounded"},
            "terminal_status": "BLOCKED",
        },
        "verification": {
            "checks": [
                {
                    "id": "source-hash",
                    "command_or_method": "Recompute SHA-256 of the referenced SKILL.md.",
                    "pass_condition": "Hash equals source.sha256.",
                    "evidence_class": "source",
                },
                {
                    "id": "contract-schema",
                    "command_or_method": "Validate this record against enhanced-skill.schema.json.",
                    "pass_condition": "Draft 2020-12 validation passes with no errors.",
                    "evidence_class": "deterministic",
                },
                {
                    "id": "source-workflow",
                    "command_or_method": f"Execute and inspect the source capability sections bound by profile {profile['profile_sha256']}.",
                    "pass_condition": f"The {skill_id} deliverable covers its declared inputs, workflow, output, and acceptance sections without bypassing any gap closure.",
                    "evidence_class": "deterministic",
                },
                {
                    "id": "candidate-review",
                    "command_or_method": "A role independent of the builder reviews candidate-bound outputs and applicable gates.",
                    "pass_condition": "No FAIL, BLOCKED, or UNKNOWN check remains.",
                    "evidence_class": "release",
                },
            ],
            "verdict_owner": "independent-reviewer",
            "self_certification": False,
            "evidence_schema": "enhanced-skill-evidence.schema.json",
        },
        "lifecycle": {
            "state": lifecycle_state,
            "introduced": "1.0.0",
            "compatibility": "Additive governance overlay; source skill remains reference-only and unchanged.",
            "deprecation": {"status": "not-deprecated"},
            "migration": {"required": False, "guide": None},
            "rollback": {
                "last_known_good": "source-skill-without-overlay",
                "procedure": "Disable the branded router entry and execute only the unchanged source skill under repository instructions.",
            },
        },
        "provenance": {
            "contract_sha256": "0" * 64,
            "generated_by": "scripts/generate-branded-skill-enhancements.py",
            "generated_at": "2026-08-16T00:00:00-07:00",
            "reviewed_at": None,
            "reviewed_by": reviewed_by,
            "owners": contract["owners"],
            "change_record": "branded-skills-v1",
        },
    }
    record["provenance"]["reviewed_at"] = reviewed_at  # type: ignore[index]
    record["provenance"]["contract_sha256"] = canonical_contract_sha(record)  # type: ignore[index]
    return record


def generate(
    source_root: Path,
    output_root: Path,
    lifecycle_state: str,
    reviewed_at: str | None,
    reviewed_by: str | None,
) -> None:
    records_dir = output_root / "enhanced"
    records_dir.mkdir(parents=True, exist_ok=True)
    skills = sorted(source_root.glob("*/*/SKILL.md"))
    if lifecycle_state == "ACTIVE" and (reviewed_at is None or reviewed_by is None):
        raise ValueError("ACTIVE generation requires --reviewed-at and --reviewed-by")
    records = [
        record_for(path, source_root, lifecycle_state, reviewed_at, reviewed_by) for path in skills
    ]
    if len(records) != 234:
        raise ValueError(f"expected 234 source skills, found {len(records)}")
    ids = [str(record["source"]["skill_id"]) for record in records]  # type: ignore[index]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise ValueError(f"duplicate skill ids: {duplicates}")
    for record in records:
        target = records_dir / f"{record['source']['skill_id']}.json"  # type: ignore[index]
        target.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    aggregate = sha256_bytes("".join(record["source"]["sha256"] for record in records).encode())  # type: ignore[index]
    contract_aggregate = sha256_bytes("".join(record["provenance"]["contract_sha256"] for record in records).encode())  # type: ignore[index]
    domains = {
        domain: sum(1 for record in records if str(record["source"]["relative_path"]).split("/")[0] == domain)  # type: ignore[index]
        for domain in sorted(DOMAIN_CONTRACTS)
    }
    index = {
        "$schema": "schemas/enhancement-index.schema.json",
        "registry_id": "skyyrose-branded-skills-enhancement-v1",
        "version": "1.0.0",
        "brand": "skyyrose",
        "status": "active" if lifecycle_state == "ACTIVE" else "draft",
        "lifecycle_state": lifecycle_state,
        "reviewed_at": reviewed_at,
        "reviewed_by": reviewed_by,
        "skill_count": len(records),
        "domains": domains,
        "source_aggregate_sha256": aggregate,
        "contract_aggregate_sha256": contract_aggregate,
        "records": [
            {
                "id": record["source"]["skill_id"],
                "contract_id": record["contract_id"],
                "domain": record["source"]["relative_path"].split("/")[0],
                "route": record["routing"]["brain_route_ids"][0],
                "path": f"enhanced/{record['source']['skill_id']}.json",
            }
            for record in records
        ],
    }
    (output_root / "enhancement-index.json").write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def compare_trees(left: Path, right: Path) -> list[str]:
    left_files = {p.relative_to(left) for p in left.rglob("*") if p.is_file()}
    right_files = {p.relative_to(right) for p in right.rglob("*") if p.is_file()}
    errors = (
        [f"file-set mismatch: {sorted(left_files ^ right_files)}"]
        if left_files != right_files
        else []
    )
    for relative in sorted(left_files & right_files):
        if left.joinpath(relative).read_bytes() != right.joinpath(relative).read_bytes():
            errors.append(f"stale generated artifact: {relative}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--lifecycle-state", choices=("DRAFT", "ACTIVE"), default="DRAFT")
    parser.add_argument("--reviewed-at")
    parser.add_argument("--reviewed-by")
    args = parser.parse_args()
    if args.check:
        current_index = json.loads(
            (args.output_root / "enhancement-index.json").read_text(encoding="utf-8")
        )
        with tempfile.TemporaryDirectory(prefix="ftt-branded-skills-") as temp:
            generated = Path(temp) / "branded-skills"
            generate(
                args.source_root,
                generated,
                current_index["lifecycle_state"],
                current_index.get("reviewed_at"),
                current_index.get("reviewed_by"),
            )
            errors = compare_trees(generated / "enhanced", args.output_root / "enhanced")
            expected_index = generated / "enhancement-index.json"
            actual_index = args.output_root / "enhancement-index.json"
            if (
                not actual_index.is_file()
                or expected_index.read_bytes() != actual_index.read_bytes()
            ):
                errors.append("stale generated artifact: enhancement-index.json")
            if errors:
                print("\n".join(errors), file=sys.stderr)
                return 1
        print("PASS: 234 branded skill enhancements are deterministic and fresh")
        return 0
    records_dir = args.output_root / "enhanced"
    if records_dir.exists():
        shutil.rmtree(records_dir)
    index_file = args.output_root / "enhancement-index.json"
    if index_file.exists():
        index_file.unlink()
    generate(
        args.source_root,
        args.output_root,
        args.lifecycle_state,
        args.reviewed_at,
        args.reviewed_by,
    )
    print(f"Generated 234 enhanced skill manifests in {args.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
