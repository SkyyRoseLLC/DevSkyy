"""JSON-only multimodal clients for OpenAI and local open-source VLMs."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI


class VisionClientError(RuntimeError):
    """A vision client failed to return a valid JSON object."""


class OpenAICompatibleVisionClient:
    """Calls OpenAI or an OpenAI-compatible local VLM endpoint.

    The same wire contract supports OpenAI, vLLM, Ollama and LocalAI.  The
    server must expose Chat Completions with multimodal ``image_url`` content.
    ``response_format=json_object`` can be disabled for compatible servers
    that guarantee JSON but do not implement that request field.
    """

    def __init__(
        self,
        *,
        id: str,
        model: str,
        api_key: str,
        base_url: str | None = None,
        timeout_seconds: float = 180.0,
        response_format: bool = True,
        max_image_bytes: int = 20_000_000,
    ) -> None:
        self.id = id
        self.model = model
        self.response_format = response_format
        self.max_image_bytes = max_image_bytes
        self._client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout_seconds,
        )

    @classmethod
    def openai(cls) -> OpenAICompatibleVisionClient:
        key = os.getenv("OPENAI_API_KEY")
        if not key:
            raise VisionClientError("OPENAI_API_KEY is not configured")
        return cls(
            id="openai-3d-vision",
            model=os.getenv("OPENAI_3D_VISION_MODEL", "gpt-5.5-pro"),
            api_key=key,
        )

    @classmethod
    def open_source(cls) -> OpenAICompatibleVisionClient:
        endpoint = os.getenv("OSS_MODEL_BASE_URL")
        model = os.getenv("OSS_3D_VISION_MODEL")
        if not endpoint or not model:
            raise VisionClientError(
                "OSS_MODEL_BASE_URL and OSS_3D_VISION_MODEL are required"
            )
        return cls(
            id="oss-3d-vision",
            model=model,
            api_key=os.getenv("OSS_MODEL_API_KEY", "local-not-secret"),
            base_url=endpoint,
            response_format=os.getenv("OSS_MODEL_JSON_RESPONSE_FORMAT", "true").lower()
            == "true",
        )

    async def generate_json(
        self,
        payload: dict[str, Any],
        images: tuple[Path, ...],
    ) -> dict:
        content: list[dict[str, Any]] = [
            {
                "type": "text",
                "text": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            }
        ]
        for path in images:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._data_url(path), "detail": "high"},
                }
            )

        kwargs: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a product-fidelity verifier. Return exactly one JSON object "
                        "matching output_schema. Never add Markdown or unsupported details."
                    ),
                },
                {"role": "user", "content": content},
            ],
        }
        if self.response_format:
            kwargs["response_format"] = {"type": "json_object"}

        try:
            response = await self._client.chat.completions.create(**kwargs)
        except Exception as exc:  # noqa: BLE001 - normalize provider SDK errors
            raise VisionClientError(f"{self.id} request failed: {exc}") from exc

        message = response.choices[0].message.content if response.choices else None
        if not isinstance(message, str) or not message.strip():
            raise VisionClientError(f"{self.id} returned no JSON content")
        try:
            parsed = json.loads(message)
        except json.JSONDecodeError as exc:
            raise VisionClientError(f"{self.id} returned invalid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise VisionClientError(f"{self.id} returned JSON {type(parsed).__name__}, expected object")
        return parsed

    async def close(self) -> None:
        await self._client.close()

    def _data_url(self, path: Path) -> str:
        if not path.is_file():
            raise VisionClientError(f"image not found: {path}")
        size = path.stat().st_size
        if size > self.max_image_bytes:
            raise VisionClientError(
                f"image exceeds {self.max_image_bytes} byte client limit: {path} ({size})"
            )
        mime = mimetypes.guess_type(path.name)[0]
        if mime not in {"image/jpeg", "image/png", "image/webp", "image/gif"}:
            raise VisionClientError(f"unsupported image MIME type for {path}: {mime}")
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"


__all__ = ["OpenAICompatibleVisionClient", "VisionClientError"]
