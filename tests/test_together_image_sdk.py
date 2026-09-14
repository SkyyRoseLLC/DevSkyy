"""Exercise legacy Together consumers with the real SDK and an offline transport."""

from __future__ import annotations

import base64
import importlib.util
import io
import json
import socket
from collections.abc import Callable
from pathlib import Path

import httpx
import pytest
from PIL import Image
from together import Together

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(autouse=True)
def prohibit_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any accidental provider or image download must fail the test."""

    def refuse(*args: object, **kwargs: object) -> None:
        raise AssertionError("Network access is forbidden in SDK compatibility tests")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)


@pytest.fixture(params=["modular", "legacy"])
def consumer(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> Callable:
    """Import both actual consumers without invoking credential factories or CLIs."""
    monkeypatch.syspath_prepend(str(ROOT / "scripts"))
    if request.param == "modular":
        from nano_banana.generate import generate_flux

        return generate_flux
    spec = importlib.util.spec_from_file_location(
        "legacy_vton_sdk_test", ROOT / "scripts/nano-banana-vton.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate_image_flux


@pytest.mark.parametrize("use_free", [False, True])
def test_real_sdk_request_and_decoded_webp(consumer: Callable, use_free: bool) -> None:
    """Together requests base64 but returns the image in its b64_json field."""
    png = io.BytesIO()
    Image.new("RGB", (8, 6), (255, 0, 0)).save(png, format="PNG")
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append(body)
        assert request.url.path == "/v1/images/generations"
        assert body["response_format"] == "base64"
        assert body["prompt"] == "synthetic compatibility fixture"
        assert (body["width"], body["height"]) == (768, 1024)
        assert body["model"] == (
            "black-forest-labs/FLUX.1-schnell-Free" if use_free else "black-forest-labs/FLUX.2-pro"
        )
        return httpx.Response(
            200,
            json={
                "id": "offline-fixture",
                "object": "list",
                "model": body["model"],
                "data": [
                    {
                        "index": 0,
                        "type": "b64_json",
                        "b64_json": base64.b64encode(png.getvalue()).decode("ascii"),
                    }
                ],
            },
        )

    with Together(
        api_key="offline-dummy-key",
        base_url="https://together.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        result = consumer(client, "synthetic compatibility fixture", use_free=use_free)
    assert len(requests) == 1
    assert isinstance(result, bytes)
    with Image.open(io.BytesIO(result)) as decoded:
        assert decoded.format == "WEBP"
        assert decoded.size == (8, 6)
        red, green, blue = decoded.convert("RGB").getpixel((4, 3))
        assert red > 240 and green < 15 and blue < 15


@pytest.mark.parametrize("status", [400, 429, 500])
def test_sdk_errors_return_none_for_caller_fallback(consumer: Callable, status: int) -> None:
    """Real SDK HTTP exceptions preserve the consumer's fallback contract."""
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(status, json={"error": {"message": "offline failure fixture"}})

    with Together(
        api_key="offline-dummy-key",
        base_url="https://together.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        assert consumer(client, "synthetic compatibility fixture") is None
    assert len(requests) == 1


def test_empty_sdk_response_returns_none(consumer: Callable) -> None:
    """An HTTP success without images remains a generation failure."""

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"id": "offline-empty", "object": "list", "model": "fixture", "data": []}
        )

    with Together(
        api_key="offline-dummy-key",
        base_url="https://together.invalid/v1",
        max_retries=0,
        http_client=httpx.Client(transport=httpx.MockTransport(respond)),
    ) as client:
        assert consumer(client, "synthetic compatibility fixture") is None
