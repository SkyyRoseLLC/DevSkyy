"""Authenticated Context.dev structured web extraction API.

The route is intentionally small: it validates the caller-controlled crawl
target, bounds the provider request, and delegates all Context.dev work to the
server-only integration wrapper. Provider credentials never cross this API.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Literal

from context.dev import APIConnectionError, APIError, APIStatusError, APITimeoutError
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field, field_validator

from integrations.context_dev import ContextDevConfigurationError, extract_structured_data
from security.jwt_oauth2_auth import RoleChecker, TokenPayload, UserRole
from security.ssrf_protection import ssrf_protection

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/context-dev", tags=["Context.dev"])

# Context.dev permits a higher maximum, but the dashboard deliberately caps a
# single interactive request to keep the provider spend and wait time bounded.
MAX_DASHBOARD_PAGES = 25
MAX_DASHBOARD_DEPTH = 3
MAX_SCHEMA_BYTES = 100_000

require_context_dev_operator = RoleChecker(
    [UserRole.SUPER_ADMIN, UserRole.ADMIN, UserRole.DEVELOPER]
)


class ContextDevExtractionRequest(BaseModel):
    """A bounded, page-grounded extraction request from the operator dashboard."""

    url: str = Field(..., min_length=1, max_length=2_048)
    model_config = {"populate_by_name": True}

    json_schema: dict[str, Any] = Field(
        ...,
        alias="schema",
        serialization_alias="schema",
        description="JSON Schema object defining the requested extracted data.",
    )
    fact_check: Literal[True] = Field(
        default=True,
        description="Always require page-grounded values for dashboard extraction.",
    )
    follow_subdomains: bool = False
    max_pages: int = Field(default=10, ge=1, le=MAX_DASHBOARD_PAGES)
    max_depth: int | None = Field(default=2, ge=0, le=MAX_DASHBOARD_DEPTH)
    parse_pdfs: bool = False

    @field_validator("url")
    @classmethod
    def validate_target_url(cls, value: str) -> str:
        """Reject internal, metadata, non-HTTP, and non-resolvable crawl targets."""
        ssrf_protection.validate_url(value)
        return value

    @field_validator("json_schema")
    @classmethod
    def validate_schema(cls, value: dict[str, Any]) -> dict[str, Any]:
        if value.get("type") != "object":
            raise ValueError("schema must be a JSON Schema object with type 'object'")

        try:
            encoded = json.dumps(value)
        except (TypeError, ValueError) as exc:
            raise ValueError("schema must be JSON serializable") from exc
        if len(encoded.encode("utf-8")) > MAX_SCHEMA_BYTES:
            raise ValueError(f"schema must be no larger than {MAX_SCHEMA_BYTES} bytes")
        return value


class ContextDevExtractionResponse(BaseModel):
    """Schema-matched structured data returned by Context.dev."""

    data: dict[str, Any]


@router.post(
    "/extractions",
    response_model=ContextDevExtractionResponse,
    status_code=status.HTTP_200_OK,
    operation_id="extract_context_dev_structured_data",
)
async def create_extraction(
    request: ContextDevExtractionRequest,
    operator: TokenPayload = Depends(require_context_dev_operator),
) -> ContextDevExtractionResponse:
    """Crawl a website and return data matching the supplied JSON Schema.

    Only operator roles may call this endpoint. The global rate limiter applies
    the dedicated per-IP budget registered for this cost-bearing route.
    """
    try:
        data = await asyncio.to_thread(
            extract_structured_data,
            url=request.url,
            schema=request.json_schema,
            fact_check=request.fact_check,
            follow_subdomains=request.follow_subdomains,
            max_pages=request.max_pages,
            max_depth=request.max_depth,
            parse_pdfs=request.parse_pdfs,
        )
    except ContextDevConfigurationError as exc:
        logger.error("Context.dev extraction is unavailable: credential missing")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Context.dev extraction is not configured on this server.",
        ) from exc
    except APITimeoutError as exc:
        logger.warning("Context.dev extraction timed out for operator=%s", operator.sub)
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="Context.dev timed out while extracting this website.",
        ) from exc
    except APIConnectionError as exc:
        logger.warning("Context.dev connection failed for operator=%s", operator.sub)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not reach Context.dev for this extraction.",
        ) from exc
    except APIStatusError as exc:
        provider_status = exc.status_code
        logger.warning(
            "Context.dev rejected extraction for operator=%s with provider status=%s",
            operator.sub,
            provider_status,
        )
        if provider_status == status.HTTP_429_TOO_MANY_REQUESTS:
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Context.dev rate-limited this extraction. Try again shortly.",
            ) from exc
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Context.dev could not complete this extraction.",
        ) from exc
    except APIError as exc:
        logger.exception("Context.dev extraction failed for operator=%s", operator.sub)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Context.dev could not complete this extraction.",
        ) from exc

    return ContextDevExtractionResponse(data=data)
