"""RAG parser availability must fail closed without leaking internals."""

from __future__ import annotations

import asyncio
from unittest.mock import patch

import pytest
from fastapi import HTTPException

from api.v1.rag_anything import _service


def test_missing_rag_parser_returns_sanitized_503() -> None:
    with patch("api.v1.rag_anything.importlib.util.find_spec", return_value=None):
        with pytest.raises(HTTPException) as exc_info:
            asyncio.run(_service())

    assert exc_info.value.status_code == 503
    assert exc_info.value.detail == {
        "code": "rag_parser_unavailable",
        "message": "Document parsing is temporarily unavailable",
    }
