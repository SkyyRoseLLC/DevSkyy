"""Wire tests for the authenticated Context.dev extraction endpoint.

All provider work is mocked. These tests prove DevSkyy's request validation,
authorization dependency, and wrapper hand-off without sending live crawls.
"""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _request_payload() -> dict[str, object]:
    return {
        "url": "https://higgsfield.com",
        "schema": {
            "type": "object",
            "properties": {"how_it_works": {"type": "string"}},
            "required": ["how_it_works"],
        },
        "fact_check": True,
        "follow_subdomains": True,
        "max_pages": 25,
        "max_depth": 2,
        "parse_pdfs": False,
    }


def _client(*, operator: bool = False) -> TestClient:
    from api.v1.context_dev import require_context_dev_operator, router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    if operator:
        app.dependency_overrides[require_context_dev_operator] = lambda: {"sub": "operator"}
    return TestClient(app, raise_server_exceptions=False)


def test_extract_structured_data_calls_server_side_wrapper(monkeypatch) -> None:
    """The public API contract passes bounded, validated options to the wrapper."""
    from api.v1 import context_dev

    calls: dict[str, object] = {}
    monkeypatch.setattr(context_dev.ssrf_protection, "validate_url", lambda url: True)

    def fake_extract(**kwargs):
        calls.update(kwargs)
        return {"how_it_works": "A grounded workflow description."}

    monkeypatch.setattr(context_dev, "extract_structured_data", fake_extract)

    response = _client(operator=True).post("/api/v1/context-dev/extractions", json=_request_payload())

    assert response.status_code == 200
    assert response.json() == {"data": {"how_it_works": "A grounded workflow description."}}
    assert calls == {
        "url": "https://higgsfield.com",
        "schema": _request_payload()["schema"],
        "fact_check": True,
        "follow_subdomains": True,
        "max_pages": 25,
        "max_depth": 2,
        "parse_pdfs": False,
    }


def test_extract_structured_data_rejects_non_object_schema(monkeypatch) -> None:
    """Schemas must declare a JSON Schema object before a provider call occurs."""
    from api.v1 import context_dev

    monkeypatch.setattr(context_dev.ssrf_protection, "validate_url", lambda url: True)

    payload = _request_payload()
    payload["schema"] = {"type": "array"}

    response = _client(operator=True).post("/api/v1/context-dev/extractions", json=payload)

    assert response.status_code == 422
    assert "object" in response.text


def test_extract_structured_data_requires_authenticated_operator() -> None:
    """A paid crawl is not available to anonymous callers."""
    response = _client().post("/api/v1/context-dev/extractions", json=_request_payload())

    assert response.status_code == 401


def test_extraction_route_has_a_dedicated_cost_rate_limit() -> None:
    """The global middleware has a narrow spend limit for the provider route."""
    from security.rate_limiting import rate_limiter

    rule = rate_limiter.endpoint_rules["/api/v1/context-dev/extractions"]
    assert rule.requests_per_minute == 3
    assert rule.burst_limit == 3
