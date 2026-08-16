"""Regression tests for bounded and signature-verified media ingestion."""

from __future__ import annotations

from io import BytesIO

import httpx
import pytest
from fastapi import UploadFile

from security.media_input import (
    MediaInputError,
    detect_image_type,
    download_validated_image,
    read_validated_upload,
    validate_image,
)

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 32


def test_detect_image_type_uses_magic_bytes() -> None:
    assert detect_image_type(PNG) == ("image/png", ".png")


def test_validate_image_rejects_mime_spoofing() -> None:
    with pytest.raises(MediaInputError, match="does not match"):
        validate_image(PNG, declared_type="image/jpeg")


@pytest.mark.asyncio
async def test_upload_size_limit_stops_incremental_read(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MEDIA_MAX_IMAGE_BYTES", "16")
    upload = UploadFile(filename="photo.png", file=BytesIO(PNG), headers=None)

    with pytest.raises(MediaInputError) as exc_info:
        await read_validated_upload(upload)

    assert exc_info.value.status_code == 413


@pytest.mark.asyncio
async def test_remote_image_rejects_private_network_before_request() -> None:
    requested = False

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal requested
        requested = True
        return httpx.Response(200, content=PNG, headers={"content-type": "image/png"})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MediaInputError, match="not allowed"):
            await download_validated_image("http://127.0.0.1/image.png", client=client)

    assert requested is False


@pytest.mark.asyncio
async def test_remote_image_validates_redirect_target() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest"})

    class PublicOnlySSRF:
        def validate_url(self, url: str) -> bool:
            if "169.254.169.254" in url:
                raise ValueError("private")
            return True

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MediaInputError, match="not allowed"):
            await download_validated_image(
                "https://images.example.com/photo.png",
                client=client,
                ssrf=PublicOnlySSRF(),
            )


@pytest.mark.asyncio
async def test_remote_image_rejects_oversized_content_length() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=PNG,
            headers={"content-type": "image/png", "content-length": "999999999"},
        )

    class AllowAllSSRF:
        def validate_url(self, url: str) -> bool:
            return True

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(MediaInputError) as exc_info:
            await download_validated_image(
                "https://images.example.com/photo.png",
                client=client,
                ssrf=AllowAllSSRF(),
            )

    assert exc_info.value.status_code == 413
