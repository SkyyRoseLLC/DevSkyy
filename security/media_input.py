"""Bounded, signature-verified image ingestion for API endpoints."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urljoin

import httpx
from fastapi import UploadFile

from security.ssrf_protection import ssrf_protection

DEFAULT_MAX_IMAGE_BYTES = 10 * 1024 * 1024
READ_CHUNK_BYTES = 64 * 1024
MAX_REMOTE_REDIRECTS = 3


class MediaInputError(ValueError):
    """Raised when an uploaded or remote image violates the input policy."""

    def __init__(self, message: str, *, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


class URLValidator(Protocol):
    def validate_url(self, url: str) -> bool: ...


@dataclass(frozen=True, slots=True)
class ValidatedImage:
    content: bytes
    content_type: str
    suffix: str


def max_image_bytes() -> int:
    """Return the configured image ceiling, falling back safely on bad input."""
    raw = os.getenv("MEDIA_MAX_IMAGE_BYTES", str(DEFAULT_MAX_IMAGE_BYTES))
    try:
        value = int(raw)
    except ValueError:
        return DEFAULT_MAX_IMAGE_BYTES
    return value if value > 0 else DEFAULT_MAX_IMAGE_BYTES


def detect_image_type(content: bytes) -> tuple[str, str]:
    """Identify a supported image using bytes, never client metadata."""
    if content.startswith(b"\xff\xd8\xff"):
        return "image/jpeg", ".jpg"
    if content.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png", ".png"
    if len(content) >= 12 and content.startswith(b"RIFF") and content[8:12] == b"WEBP":
        return "image/webp", ".webp"
    if content.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif", ".gif"
    raise MediaInputError("Unsupported or invalid image content")


def validate_image(content: bytes, *, declared_type: str | None = None) -> ValidatedImage:
    """Validate size, magic bytes, and an optional declared MIME type."""
    if not content:
        raise MediaInputError("Image is empty")
    if len(content) > max_image_bytes():
        raise MediaInputError("Image exceeds the configured size limit", status_code=413)

    content_type, suffix = detect_image_type(content)
    normalized_type = (declared_type or "").split(";", 1)[0].strip().lower()
    if normalized_type and normalized_type != content_type:
        raise MediaInputError("Declared content type does not match image content")
    return ValidatedImage(content=content, content_type=content_type, suffix=suffix)


async def read_validated_upload(file: UploadFile) -> ValidatedImage:
    """Read an upload incrementally and stop once it exceeds the ceiling."""
    limit = max_image_bytes()
    chunks: list[bytes] = []
    total = 0
    while chunk := await file.read(READ_CHUNK_BYTES):
        total += len(chunk)
        if total > limit:
            raise MediaInputError("Image exceeds the configured size limit", status_code=413)
        chunks.append(chunk)
    return validate_image(b"".join(chunks), declared_type=file.content_type)


async def download_validated_image(
    url: str,
    *,
    client: httpx.AsyncClient | None = None,
    ssrf: URLValidator = ssrf_protection,
) -> ValidatedImage:
    """Fetch a public image with SSRF, redirect, timeout, and size controls."""
    owns_client = client is None
    http_client = client or httpx.AsyncClient(
        follow_redirects=False,
        timeout=httpx.Timeout(10.0),
    )
    current_url = url

    try:
        for redirect_count in range(MAX_REMOTE_REDIRECTS + 1):
            try:
                ssrf.validate_url(current_url)
            except ValueError as exc:
                raise MediaInputError("Remote image URL is not allowed") from exc

            async with http_client.stream("GET", current_url) as response:
                if response.is_redirect:
                    location = response.headers.get("location")
                    if not location or redirect_count == MAX_REMOTE_REDIRECTS:
                        raise MediaInputError("Remote image redirect is not allowed")
                    current_url = urljoin(current_url, location)
                    continue

                response.raise_for_status()
                content_length = response.headers.get("content-length")
                if content_length:
                    try:
                        declared_length = int(content_length)
                    except ValueError as exc:
                        raise MediaInputError("Remote image has an invalid content length") from exc
                    if declared_length > max_image_bytes():
                        raise MediaInputError(
                            "Remote image exceeds the configured size limit",
                            status_code=413,
                        )

                chunks: list[bytes] = []
                total = 0
                async for chunk in response.aiter_bytes(READ_CHUNK_BYTES):
                    total += len(chunk)
                    if total > max_image_bytes():
                        raise MediaInputError(
                            "Remote image exceeds the configured size limit",
                            status_code=413,
                        )
                    chunks.append(chunk)
                return validate_image(
                    b"".join(chunks),
                    declared_type=response.headers.get("content-type"),
                )
    except httpx.HTTPError as exc:
        raise MediaInputError("Remote image could not be downloaded") from exc
    finally:
        if owns_client:
            await http_client.aclose()

    raise MediaInputError("Remote image redirect is not allowed")
