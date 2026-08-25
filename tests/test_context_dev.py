"""Unit tests for the server-side Context.dev integration."""

from unittest.mock import Mock, patch

import pytest

from integrations.context_dev import ContextDevConfigurationError, extract_structured_data


SCHEMA = {
    "type": "object",
    "properties": {
        "mission_statement": {"type": "string"},
        "pricing_plans": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["mission_statement", "pricing_plans"],
}


def test_extract_structured_data_returns_schema_matched_data(monkeypatch: pytest.MonkeyPatch) -> None:
    """The wrapper returns only the SDK's schema-matched data payload."""
    monkeypatch.setenv("CONTEXT_DEV_API_KEY", "test-context-dev-key")
    response = Mock(data={"mission_statement": "Increase the GDP of the internet.", "pricing_plans": ["Pro"]})

    with patch("integrations.context_dev.ContextDev") as context_dev:
        context_dev.return_value.web.extract.return_value = response

        data = extract_structured_data(
            "https://higgsfield.com",
            SCHEMA,
            follow_subdomains=True,
            max_pages=10,
            max_depth=2,
        )

    assert data == response.data
    context_dev.assert_called_once_with(api_key="test-context-dev-key")
    context_dev.return_value.web.extract.assert_called_once_with(
        url="https://higgsfield.com",
        schema=SCHEMA,
        fact_check=True,
        follow_subdomains=True,
        max_pages=10,
        max_depth=2,
        pdf={"should_parse": False},
    )


def test_extract_structured_data_requires_an_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The integration fails closed when its server-only credential is absent."""
    monkeypatch.delenv("CONTEXT_DEV_API_KEY", raising=False)

    with pytest.raises(ContextDevConfigurationError, match="CONTEXT_DEV_API_KEY"):
        extract_structured_data("https://higgsfield.com", SCHEMA)
