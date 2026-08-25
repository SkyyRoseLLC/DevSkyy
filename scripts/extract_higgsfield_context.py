#!/usr/bin/env python3
"""Manually extract grounded Higgsfield context for internal pipeline research.

This command makes one live Context.dev request only when a developer runs it.
It does not persist the result or expose it through an HTTP endpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))


HIGGSFIELD_PIPELINE_SCHEMA = {
    "type": "object",
    "properties": {
        "platform_summary": {
            "type": ["string", "null"],
            "description": "A concise summary using only claims stated on Higgsfield's website.",
        },
        "primary_user_types": {
            "type": "array",
            "description": "User roles or customer groups explicitly named by Higgsfield.",
            "items": {"type": "string"},
        },
        "products": {
            "type": "array",
            "description": "Publicly described Higgsfield products, tools, or modules.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "category": {"type": ["string", "null"]},
                    "description": {"type": ["string", "null"]},
                    "product_url": {"type": ["string", "null"]},
                },
                "required": ["name", "category", "description", "product_url"],
                "additionalProperties": False,
            },
        },
        "user_workflows": {
            "type": "array",
            "description": "Publicly described user journeys from starting material to result.",
            "items": {
                "type": "object",
                "properties": {
                    "workflow_name": {"type": "string"},
                    "user_goal": {"type": ["string", "null"]},
                    "starting_inputs": {"type": "array", "items": {"type": "string"}},
                    "steps": {"type": "array", "items": {"type": "string"}},
                    "available_controls": {"type": "array", "items": {"type": "string"}},
                    "outputs": {"type": "array", "items": {"type": "string"}},
                    "source_url": {"type": ["string", "null"]},
                },
                "required": [
                    "workflow_name",
                    "user_goal",
                    "starting_inputs",
                    "steps",
                    "available_controls",
                    "outputs",
                    "source_url",
                ],
                "additionalProperties": False,
            },
        },
        "integrations_and_compatibility": {
            "type": "array",
            "description": "Named integrations, compatible tools, platforms, APIs, or formats.",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "relationship": {"type": ["string", "null"]},
                    "source_url": {"type": ["string", "null"]},
                },
                "required": ["name", "relationship", "source_url"],
                "additionalProperties": False,
            },
        },
        "pricing_and_usage_limits": {
            "type": "array",
            "description": "Public pricing plans, credits, usage limits, or commercial restrictions.",
            "items": {
                "type": "object",
                "properties": {
                    "plan_name": {"type": "string"},
                    "price": {"type": ["string", "null"]},
                    "billing_period": {"type": ["string", "null"]},
                    "included_usage": {"type": "array", "items": {"type": "string"}},
                    "limits_or_conditions": {"type": "array", "items": {"type": "string"}},
                    "source_url": {"type": ["string", "null"]},
                },
                "required": [
                    "plan_name",
                    "price",
                    "billing_period",
                    "included_usage",
                    "limits_or_conditions",
                    "source_url",
                ],
                "additionalProperties": False,
            },
        },
        "evidence_gaps": {
            "type": "array",
            "description": "Important operating details that the crawled pages do not support.",
            "items": {"type": "string"},
        },
    },
    "required": [
        "platform_summary",
        "primary_user_types",
        "products",
        "user_workflows",
        "integrations_and_compatibility",
        "pricing_and_usage_limits",
        "evidence_gaps",
    ],
    "additionalProperties": False,
}


def parse_args() -> argparse.Namespace:
    """Build the explicit live-extraction command interface."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="https://higgsfield.com", help="Starting HTTP(S) URL")
    parser.add_argument("--max-pages", type=int, default=25, help="Pages to analyze (1-50)")
    parser.add_argument("--max-depth", type=int, default=2, help="Maximum crawl link depth")
    parser.add_argument("--parse-pdfs", action="store_true", help="Allow parsing discovered PDFs")
    return parser.parse_args()


def main() -> None:
    """Run the bounded, fact-checked Higgsfield extraction on explicit request."""

    load_dotenv(REPOSITORY_ROOT / ".env", override=False)
    from integrations.context_dev import extract_structured_data

    args = parse_args()
    data = extract_structured_data(
        args.url,
        HIGGSFIELD_PIPELINE_SCHEMA,
        fact_check=True,
        follow_subdomains=True,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        parse_pdfs=args.parse_pdfs,
    )
    print(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
