"""OpenAI gpt-image-2 image client — hardened, no silent fallback.

Wraps ``client.images.edit`` (reference-guided product renders) and
``client.images.generate`` (text-to-image scene backgrounds, no references)
with explicit timeout, bounded exponential-backoff retry on transient errors
(429 / timeout / connection / 5xx), and clear failure propagation. gpt-image
models always return base64 and never accept ``response_format`` (Context7-
verified against openai-python image_generate_params.py, 2026-06-24) — so it is
never sent; the returned base64 is decoded to image bytes.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from . import config

log = logging.getLogger(__name__)

_MIME = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".webp": "image/webp"}


@dataclass(frozen=True)
class ImageCallResult:
    """Decoded image plus the non-secret provider evidence for its API call."""

    data: bytes
    request_id: str
    requested_model: str
    sdk_version: str
    endpoint: str
    request_parameters: dict[str, object]
    ordered_input_sha256s: tuple[str, ...]
    mask_sha256: str | None
    prompt_sha256: str
    job_contract: dict[str, object]
    contract_sha256: str


class RenderClient(Protocol):
    """Structural type for an image render client (used for typing in pipeline)."""

    def edit_with_metadata(
        self,
        *,
        prompt: str,
        image_paths: list[Path],
        expected_input_sha256s: tuple[str, ...] | None = None,
    ) -> ImageCallResult: ...


def _as_upload(path: Path) -> tuple[str, bytes, str]:
    """Read an image into the (filename, bytes, content-type) tuple the SDK accepts."""
    mime = _MIME.get(path.suffix.lower())
    if mime is None:
        raise ValueError(f"Unsupported image type for edit reference: {path}")
    return (path.name, path.read_bytes(), mime)


class OAIImageClient:
    """Thin, retrying wrapper around OpenAI image edit + generate for gpt-image-2."""

    def __init__(self) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("openai SDK not installed. `pip install openai` (>=1.0).") from exc
        self._OpenAI = OpenAI
        self._client = OpenAI(
            api_key=config.get_api_key(),
            timeout=config.REQUEST_TIMEOUT_S,
            max_retries=0,
        )

    def _transient_errors(self) -> tuple[type[Exception], ...]:
        """Return the OpenAI exception classes worth retrying."""
        import openai

        names = ("RateLimitError", "APITimeoutError", "APIConnectionError", "InternalServerError")
        errs = tuple(getattr(openai, n) for n in names if hasattr(openai, n))
        return errs or (Exception,)

    def edit(self, *, prompt: str, image_paths: list[Path], mask_path: Path | None = None) -> bytes:
        """Run one gpt-image-2 edit and return decoded image bytes.

        Retries transient failures with exponential backoff + jitter; re-raises
        after ``config.MAX_RETRIES``. Never returns a partial/placeholder image.
        """
        if not image_paths:
            raise ValueError("edit() requires at least one reference image.")
        if len(image_paths) > config.MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"{len(image_paths)} references exceeds the {config.MAX_REFERENCE_IMAGES} limit."
            )

        return self.edit_with_metadata(
            prompt=prompt, image_paths=image_paths, mask_path=mask_path
        ).data

    def edit_with_metadata(
        self,
        *,
        prompt: str,
        image_paths: list[Path],
        mask_path: Path | None = None,
        expected_input_sha256s: tuple[str, ...] | None = None,
    ) -> ImageCallResult:
        """Run an edit and retain request metadata needed for an audit receipt."""
        if not image_paths:
            raise ValueError("edit_with_metadata() requires at least one reference image.")
        if len(image_paths) > config.MAX_REFERENCE_IMAGES:
            raise ValueError(
                f"{len(image_paths)} references exceeds the {config.MAX_REFERENCE_IMAGES} limit."
            )

        images = [_as_upload(p) for p in image_paths]
        ordered_input_sha256s = tuple(hashlib.sha256(image[1]).hexdigest() for image in images)
        if expected_input_sha256s is not None and ordered_input_sha256s != expected_input_sha256s:
            raise RuntimeError("reference bytes changed between pipeline preflight and upload")
        kwargs: dict = {
            "model": config.MODEL,
            "image": images,
            "prompt": prompt,
            "quality": config.QUALITY,
            "size": config.SIZE,
            "output_format": config.OUTPUT_FORMAT,
            "background": config.BACKGROUND,
            "n": config.N,
        }
        # input_fidelity is rejected outright (400) by models that don't
        # support it -- e.g. gpt-image-2 (bug-172). Only send it when the
        # configured model is known to accept it.
        if config.MODEL in config.INPUT_FIDELITY_SUPPORTED_MODELS:
            kwargs["input_fidelity"] = config.INPUT_FIDELITY
        mask_upload = _as_upload(mask_path) if mask_path is not None else None
        if mask_upload is not None:
            kwargs["mask"] = mask_upload

        public_parameters = {
            key: value for key, value in kwargs.items() if key not in {"image", "mask", "prompt"}
        }
        prompt_sha256 = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        mask_sha256 = hashlib.sha256(mask_upload[1]).hexdigest() if mask_upload else None
        job_contract: dict[str, object] = {
            "contract_id": "skyyrose-product-image-candidate-v1",
            "authority_state": "CANDIDATE_ONLY",
            "operation": "product_image_candidate",
            "provider": "openai",
            "api_surface": "images",
            "endpoint": "/v1/images/edits",
            "requested_model": config.MODEL,
            "prompt_sha256": prompt_sha256,
            "request_image_sha256s": list(ordered_input_sha256s),
            "mask_sha256": mask_sha256,
            "request_parameters": public_parameters,
            "release_authority": False,
        }
        contract_sha256 = hashlib.sha256(
            json.dumps(job_contract, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        return self._run_with_retry_raw(
            lambda: self._client.images.with_raw_response.edit(**kwargs),
            public_parameters=public_parameters,
            ordered_input_sha256s=ordered_input_sha256s,
            mask_sha256=mask_sha256,
            prompt_sha256=prompt_sha256,
            job_contract=job_contract,
            contract_sha256=contract_sha256,
        )

    def generate(
        self,
        *,
        prompt: str,
        size: str | None = None,
        quality: str | None = None,
        output_format: str | None = None,
        background: str = "opaque",
    ) -> bytes:
        """Run one gpt-image-2 text-to-image generation; return decoded bytes.

        Unlike :meth:`edit`, sends NO reference image — used for scene
        backgrounds. gpt-image models always return base64 and reject
        ``response_format`` (Context7-verified), so it is never sent. ``size``
        accepts any gpt-image-2 ``WIDTHxHEIGHT`` (both divisible by 16, aspect
        1:3..3:1); defaults to ``config.SIZE``. ``background`` defaults to
        ``"opaque"`` (scenes are full backdrops, not cut-outs). Retries
        transient failures with the same backoff as :meth:`edit`.
        """
        kwargs: dict = {
            "model": config.MODEL,
            "prompt": prompt,
            "size": size or config.SIZE,
            "quality": quality or config.QUALITY,
            "output_format": output_format or config.OUTPUT_FORMAT,
            "background": background,
            "n": config.N,
        }
        return self._run_with_retry(lambda: self._client.images.generate(**kwargs))

    def _decode_first(self, resp) -> bytes:
        """Decode the first image of an OpenAI images response to raw bytes."""
        b64 = resp.data[0].b64_json
        if not b64:
            raise RuntimeError("OpenAI returned an empty image payload.")
        return base64.b64decode(b64)

    def _run_with_retry(self, do_call) -> bytes:
        """Run ``do_call()`` (one image API call) with bounded backoff retry.

        Retries only transient OpenAI errors (429 / timeout / connection /
        5xx); re-raises after ``config.MAX_RETRIES``. Non-transient errors
        (including an empty payload) propagate immediately. Never returns a
        partial/placeholder image.
        """
        transient = self._transient_errors()
        attempt = 0
        while True:
            try:
                return self._decode_first(do_call())
            except transient as exc:
                attempt += 1
                if attempt > config.MAX_RETRIES:
                    log.error(
                        "gpt-image-2 call failed after %d retries: %s", config.MAX_RETRIES, exc
                    )
                    raise
                delay = min(
                    config.RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1))
                    + random.uniform(0, 1),  # nosec B311 — backoff jitter, not crypto
                    config.RETRY_BACKOFF_MAX_S,
                )
                log.warning(
                    "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    config.MAX_RETRIES,
                    exc,
                    delay,
                )
                time.sleep(delay)

    def _run_with_retry_raw(
        self,
        do_call,
        *,
        public_parameters: dict[str, object],
        ordered_input_sha256s: tuple[str, ...],
        mask_sha256: str | None,
        prompt_sha256: str,
        job_contract: dict[str, object],
        contract_sha256: str,
    ) -> ImageCallResult:
        """Run a raw-response call and preserve its provider request identifier."""
        import openai

        transient = self._transient_errors()
        attempt = 0
        while True:
            try:
                raw = do_call()
                parsed = raw.parse()
                request_id = raw.headers.get("x-request-id")
                if not request_id:
                    raise RuntimeError(
                        "OpenAI image response omitted x-request-id; refusing an untraceable result."
                    )
                return ImageCallResult(
                    data=self._decode_first(parsed),
                    request_id=request_id,
                    requested_model=config.MODEL,
                    sdk_version=openai.__version__,
                    endpoint="/v1/images/edits",
                    request_parameters=public_parameters,
                    ordered_input_sha256s=ordered_input_sha256s,
                    mask_sha256=mask_sha256,
                    prompt_sha256=prompt_sha256,
                    job_contract=job_contract,
                    contract_sha256=contract_sha256,
                )
            except transient as exc:
                attempt += 1
                if attempt > config.MAX_RETRIES:
                    log.error(
                        "gpt-image-2 call failed after %d retries: %s", config.MAX_RETRIES, exc
                    )
                    raise
                delay = min(
                    config.RETRY_BACKOFF_BASE_S * (2 ** (attempt - 1))
                    + random.uniform(0, 1),  # nosec B311 — backoff jitter, not crypto
                    config.RETRY_BACKOFF_MAX_S,
                )
                log.warning(
                    "Transient error (attempt %d/%d): %s — retrying in %.1fs",
                    attempt,
                    config.MAX_RETRIES,
                    exc,
                    delay,
                )
                time.sleep(delay)
