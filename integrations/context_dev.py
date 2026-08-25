"""Server-side Context.dev structured web extraction.

This module deliberately exposes no FastAPI route. Pipeline code imports the
wrapper directly, keeping the Context.dev credential out of browser bundles and
preventing arbitrary public callers from spending extraction credits.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlparse

from context.dev import ContextDev


class ContextDevConfigurationError(RuntimeError):
    """Raised when the server-side Context.dev integration is not configured."""


def extract_structured_data(
    url: str,
    schema: Mapping[str, object],
    *,
    fact_check: bool = True,
    follow_subdomains: bool = False,
    max_pages: int = 5,
    max_depth: int | None = None,
    parse_pdfs: bool = False,
) -> dict[str, object]:
    """Extract page-grounded structured data matching ``schema`` from ``url``.

    The official Context.dev Python SDK is synchronous. Call this function from
    a worker or pipeline rather than a latency-sensitive async request handler.
    A missing credential fails closed before an outbound request is attempted.

    Args:
        url: HTTP(S) URL that begins the extraction crawl.
        schema: JSON Schema object defining the expected returned fields.
        fact_check: Require every value to be grounded in crawled page content.
        follow_subdomains: Allow the crawl to follow subdomain links.
        max_pages: Maximum number of pages to analyze (1 through 50).
        max_depth: Optional maximum link depth from the starting URL.
        parse_pdfs: Whether discovered PDF links may be parsed.

    Returns:
        The SDK's schema-matched ``response.data`` object.

    Raises:
        ContextDevConfigurationError: If ``CONTEXT_DEV_API_KEY`` is unset.
        ValueError: If URL or crawl inputs are invalid.
    """
    api_key = os.getenv("CONTEXT_DEV_API_KEY", "").strip()
    if not api_key:
        raise ContextDevConfigurationError(
            "CONTEXT_DEV_API_KEY must be set in the server environment before " "using Context.dev."
        )

    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("url must be an absolute HTTP(S) URL")
    if not isinstance(schema, Mapping) or schema.get("type") != "object":
        raise ValueError("schema must be a JSON Schema object with type 'object'")
    if not 1 <= max_pages <= 50:
        raise ValueError("max_pages must be between 1 and 50")
    if max_depth is not None and max_depth < 0:
        raise ValueError("max_depth must be zero or greater")

    request: dict[str, Any] = {
        "url": url,
        "schema": dict(schema),
        "fact_check": fact_check,
        "follow_subdomains": follow_subdomains,
        "max_pages": max_pages,
        "pdf": {"should_parse": parse_pdfs},
    }
    if max_depth is not None:
        request["max_depth"] = max_depth

    response = ContextDev(api_key=api_key).web.extract(**request)
    return response.data
