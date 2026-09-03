"""FASHN AI virtual try-on HTTP client — replaces the tryon_agent.py:53 stub.

Closes the silent-fallback bug: the stub returned the input image unchanged
with `success=False`, but the field convention `output_path=garment_image_path`
made downstream callers treat it as a tryon result. Per CLAUDE.md feedback
"No silent fallback on missing dossier" — same principle applies to FASHN:
either return a real transformed image, or raise loudly. Never quietly hand
back the input.

API reference (FASHN v1):
    POST /v1/run           → {"id": "abc123"}
    GET  /v1/status/{id}   → {"status": "completed", "output": ["https://..."]}

Auth: Bearer FASHN_API_KEY (Authorization header).

Billing:
    tryon-v1.6 uses the legacy USD estimate below.
    tryon-max is credit-priced by generation mode and resolution; the client
    reports credits and deliberately does not invent a USD conversion.

This module deliberately does NOT bypass the paid-api-stopgate hook — that
hook intercepts shell `python -m renders.fashn` invocations. In-process
calls from this module MUST be gated by the caller via
`RunBudget.ensure_within_budget(cost_estimate, stage="tryon")` BEFORE
calling `run_tryon`. The client itself does NOT enforce a budget — caller
responsibility. See `skyyrose/elite_studio/budget.py::RunBudget`.

Pipeline integration note: `skyyrose/elite_studio/graph/nodes.py::tryon_node`
currently feeds local file paths into `tryon_agent.execute_tryon`, which D7
correctly rejects with `FashnError`. The graph tryon stage will hard-fail
on every invocation until an image-upload pipeline (local render → public R2
URL) is wired in. This is intentional — the prior stub silently returned the
input image as a fake-success, which D7 closes.
"""

from __future__ import annotations

import asyncio
import os
import time
from dataclasses import dataclass, field
from typing import Any, Literal

import httpx

# Final/polling states from FASHN status endpoint.
_TERMINAL_STATES: frozenset[str] = frozenset({"completed", "failed", "canceled"})
_IN_FLIGHT_STATES: frozenset[str] = frozenset({"in_queue", "processing", "starting"})

# Per-sample cost in USD. Authoritative source: renders/preflight.py:25-29.
# When FASHN changes pricing, update this here AND in
# .claude/hooks/paid-api-stopgate.sh manifest text AND in
# mcp_tools/tools/elite_studio.py::_COST_PER_SAMPLE_USD.
COST_PER_SAMPLE_USD: dict[str, float] = {
    "tryon-v1.6": 0.075,
    "tryon-v1.5": 0.075,
    "bg-remove-v1": 0.025,
}

# First-party Try-On Max pricing, verified 2026-09-02:
# https://docs.fashn.ai/api-reference/tryon-max
TRYON_MAX_CREDITS: dict[tuple[str, str], int] = {
    ("fast", "1k"): 1,
    ("fast", "2k"): 2,
    ("fast", "4k"): 3,
    ("balanced", "1k"): 2,
    ("balanced", "2k"): 3,
    ("balanced", "4k"): 4,
    ("quality", "1k"): 3,
    ("quality", "2k"): 4,
    ("quality", "4k"): 5,
}
TRYON_MAX_MODEL = "tryon-max"
TRYON_MAX_LIFECYCLE = "preview"

TryOnMaxResolution = Literal["1k", "2k", "4k"]
TryOnMaxGenerationMode = Literal["fast", "balanced", "quality"]
TryOnMaxOutputFormat = Literal["png", "jpeg"]


class FashnError(RuntimeError):
    """Raised on any FASHN API failure (HTTP error, job failure, timeout)."""


@dataclass(frozen=True)
class FashnCredentials:
    """FASHN API credentials. `api_key` excluded from repr to avoid log leaks."""

    api_key: str = field(repr=False)

    @classmethod
    def from_env(cls) -> FashnCredentials:
        key = os.environ.get("FASHN_API_KEY", "")
        if not key:
            raise KeyError("FASHN_API_KEY not set in environment")
        return cls(api_key=key)

    def __str__(self) -> str:
        return "FashnCredentials(api_key=<redacted>)"


@dataclass(frozen=True)
class FashnResult:
    """Successful FASHN tryon result."""

    job_id: str
    output_urls: list[str]
    model_name: str
    cost_usd: float
    latency_s: float
    raw_status: dict[str, Any]
    credits_used: int | None = None


class FashnClient:
    """Async FASHN client with poll-until-complete semantics + retry on 429/503.

    Usage:
        async with FashnClient.from_env() as client:
            result = await client.run_tryon(
                model_image_url="https://...",
                garment_image_url="https://...",
                category="upper_body",
                num_samples=4,
            )
    """

    BASE_URL = "https://api.fashn.ai/v1"
    _RETRY_DELAYS_SECONDS = (1.0, 2.0, 4.0, 8.0)
    _RETRIABLE_STATUSES = frozenset({429, 503})
    _RETRIABLE_NETWORK = (httpx.ConnectError, httpx.ReadTimeout, httpx.RemoteProtocolError)

    def __init__(
        self,
        credentials: FashnCredentials,
        timeout_seconds: float = 60.0,
        poll_interval_seconds: float = 2.0,
        max_poll_seconds: float = 300.0,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._creds = credentials
        self._poll_interval = poll_interval_seconds
        self._max_poll = max_poll_seconds
        kwargs: dict[str, Any] = {
            "base_url": self.BASE_URL,
            "timeout": timeout_seconds,
            "headers": {
                "Authorization": f"Bearer {credentials.api_key}",
                "Content-Type": "application/json",
                "User-Agent": "skyyrose-fashn-client/1.0",
            },
        }
        if transport is not None:
            kwargs["transport"] = transport
        self._client = httpx.AsyncClient(**kwargs)

    @classmethod
    def from_env(cls, **kwargs: Any) -> FashnClient:
        return cls(FashnCredentials.from_env(), **kwargs)

    async def __aenter__(self) -> FashnClient:
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.close()

    async def close(self) -> None:
        await self._client.aclose()

    async def _request_with_retry(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
        sleep: Any = asyncio.sleep,
    ) -> httpx.Response:
        last_exc: Exception | None = None
        for attempt in range(len(self._RETRY_DELAYS_SECONDS) + 1):
            try:
                response = await self._client.request(method, path, json=json)
            except self._RETRIABLE_NETWORK as exc:
                last_exc = exc
                if attempt >= len(self._RETRY_DELAYS_SECONDS):
                    raise FashnError(f"network error after retries: {exc}") from exc
                await sleep(self._RETRY_DELAYS_SECONDS[attempt])
                continue
            if response.status_code not in self._RETRIABLE_STATUSES:
                return response
            if attempt >= len(self._RETRY_DELAYS_SECONDS):
                return response
            await sleep(self._RETRY_DELAYS_SECONDS[attempt])
        if last_exc:
            raise FashnError(str(last_exc))
        raise FashnError("unreachable: retry loop exited")

    async def _request_once(
        self,
        method: str,
        path: str,
        *,
        json: dict | None = None,
    ) -> httpx.Response:
        """Send a non-idempotent request exactly once.

        A connection failure after a provider accepted ``POST /run`` is
        ambiguous. Retrying it could create and bill a second prediction, so
        candidate-generation submissions use this path and require a human to
        reconcile provider history before any retry.
        """
        try:
            return await self._client.request(method, path, json=json)
        except self._RETRIABLE_NETWORK as exc:
            raise FashnError(
                "FASHN submission outcome is unknown after a network error; "
                "automatic retry is disabled to prevent duplicate paid jobs"
            ) from exc

    async def _start_job(
        self,
        model_name: str,
        inputs: dict[str, Any],
        *,
        retry_submission: bool = True,
        sleep: Any = asyncio.sleep,
    ) -> str:
        """POST /v1/run — returns job ID."""
        payload = {"model_name": model_name, "inputs": inputs}
        if retry_submission:
            response = await self._request_with_retry("POST", "/run", json=payload, sleep=sleep)
        else:
            response = await self._request_once("POST", "/run", json=payload)
        if response.status_code not in (200, 201, 202):
            raise FashnError(
                f"FASHN /run failed: HTTP {response.status_code} — {_safe_error_excerpt(response)}"
            )
        body = response.json()
        job_id = body.get("id")
        if not job_id:
            raise FashnError(f"FASHN /run returned no job id: {body}")
        return job_id

    async def _poll_until_done(
        self,
        job_id: str,
        *,
        sleep: Any = asyncio.sleep,
    ) -> dict[str, Any]:
        """GET /v1/status/{id} until terminal state or timeout."""
        deadline = time.monotonic() + self._max_poll
        while True:
            response = await self._request_with_retry("GET", f"/status/{job_id}", sleep=sleep)
            if response.status_code != 200:
                raise FashnError(
                    f"FASHN /status/{job_id} failed: HTTP {response.status_code} — "
                    f"{_safe_error_excerpt(response)}"
                )
            body = response.json()
            status = body.get("status", "")
            if status in _TERMINAL_STATES:
                return body
            if status not in _IN_FLIGHT_STATES:
                raise FashnError(f"unknown FASHN status: {status!r}; body: {body}")
            if time.monotonic() >= deadline:
                raise FashnError(
                    f"FASHN job {job_id} did not complete within {self._max_poll}s "
                    f"(last status: {status})"
                )
            await sleep(self._poll_interval)

    async def run_tryon(
        self,
        model_image_url: str,
        garment_image_url: str,
        category: Literal["upper_body", "lower_body", "full_body", "auto"] = "auto",
        num_samples: int = 1,
        model_name: str = "tryon-v1.6",
        *,
        sleep: Any = asyncio.sleep,
    ) -> FashnResult:
        """Run a try-on. Blocks until terminal state.

        Args:
            model_image_url: public URL of the model photo
            garment_image_url: public URL of the garment cutout
            category: garment category
            num_samples: how many output variations to generate (cost multiplier)
            model_name: FASHN model version

        Raises:
            FashnError: HTTP failure, job failure, or timeout. NEVER returns
                input image as a fake-success result.
        """
        if num_samples < 1 or num_samples > 16:
            raise ValueError(f"num_samples must be 1..16, got {num_samples}")

        start = time.monotonic()
        inputs = {
            "model_image": model_image_url,
            "garment_image": garment_image_url,
            "category": category,
            "num_samples": num_samples,
        }
        job_id = await self._start_job(model_name, inputs, sleep=sleep)
        status_body = await self._poll_until_done(job_id, sleep=sleep)

        if status_body.get("status") != "completed":
            err = status_body.get("error") or status_body.get("status", "unknown")
            raise FashnError(f"FASHN job {job_id} did not complete: {err}")

        # Strict validation: response shape `{"output": [""]}` or `{"output": [None]}`
        # is a non-empty list (passes naive `not outputs` check) but contains no
        # real URLs. Filter to https:// strings only. CR finding 2026-05-22 HIGH —
        # this was the same class of silent-success bug D7 was written to close.
        raw_outputs = status_body.get("output") or []
        outputs = [u for u in raw_outputs if isinstance(u, str) and u.startswith("https://")]
        if not outputs:
            raise FashnError(
                f"FASHN job {job_id} completed but returned no valid output URLs "
                f"(raw output: {raw_outputs!r})"
            )

        per_sample = COST_PER_SAMPLE_USD.get(model_name, 0.075)
        return FashnResult(
            job_id=job_id,
            output_urls=outputs,
            model_name=model_name,
            cost_usd=per_sample * num_samples,
            latency_s=time.monotonic() - start,
            raw_status=status_body,
        )

    async def run_tryon_max(
        self,
        *,
        model_image: str,
        product_image: str,
        prompt: str = "",
        resolution: TryOnMaxResolution = "2k",
        generation_mode: TryOnMaxGenerationMode = "quality",
        seed: int = 42,
        num_images: int = 1,
        output_format: TryOnMaxOutputFormat = "png",
        return_base64: bool = True,
        sleep: Any = asyncio.sleep,
    ) -> FashnResult:
        """Run one governed Try-On Max candidate request.

        ``model_image`` and ``product_image`` may be public URLs or image data
        URIs. SkyyRose's OODA runner uses data URIs so source files do not need
        to be uploaded to a public bucket. The provider submission is sent
        exactly once; only status polling is retried.

        Try-On Max is a Preview endpoint. A successful provider response is a
        candidate receipt, never product-fidelity approval or scene promotion.
        """
        if not isinstance(model_image, str) or not model_image.strip():
            raise ValueError("model_image must be a non-empty URL or data URI")
        if not isinstance(product_image, str) or not product_image.strip():
            raise ValueError("product_image must be a non-empty URL or data URI")
        if not isinstance(prompt, str):
            raise TypeError("prompt must be a string")
        if resolution not in {"1k", "2k", "4k"}:
            raise ValueError(f"unsupported Try-On Max resolution: {resolution!r}")
        if generation_mode not in {"fast", "balanced", "quality"}:
            raise ValueError(f"unsupported Try-On Max generation_mode: {generation_mode!r}")
        if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= 2**32 - 1:
            raise ValueError("seed must be an integer from 0 to 2^32 - 1")
        if (
            isinstance(num_images, bool)
            or not isinstance(num_images, int)
            or not 1 <= num_images <= 4
        ):
            raise ValueError("num_images must be an integer from 1 to 4")
        if output_format not in {"png", "jpeg"}:
            raise ValueError(f"unsupported Try-On Max output_format: {output_format!r}")
        if not isinstance(return_base64, bool):
            raise TypeError("return_base64 must be a boolean")

        inputs: dict[str, Any] = {
            "product_image": product_image,
            "model_image": model_image,
            "prompt": prompt,
            "resolution": resolution,
            "generation_mode": generation_mode,
            "seed": seed,
            "num_images": num_images,
            "output_format": output_format,
            "return_base64": return_base64,
        }

        start = time.monotonic()
        job_id = await self._start_job(
            TRYON_MAX_MODEL,
            inputs,
            retry_submission=False,
            sleep=sleep,
        )
        status_body = await self._poll_until_done(job_id, sleep=sleep)
        if status_body.get("status") != "completed":
            err = status_body.get("error") or status_body.get("status", "unknown")
            raise FashnError(f"FASHN job {job_id} did not complete: {err}")

        raw_outputs = status_body.get("output") or []
        expected_prefixes = (
            ("data:image/png;base64,", "data:image/jpeg;base64,")
            if return_base64
            else ("https://",)
        )
        outputs = [
            value
            for value in raw_outputs
            if isinstance(value, str) and value.startswith(expected_prefixes)
        ]
        if len(outputs) != num_images:
            expected = "base64 image data" if return_base64 else "HTTPS output URLs"
            raise FashnError(
                f"FASHN job {job_id} returned {len(outputs)} valid {expected}; "
                f"expected exactly {num_images} (raw output count: {len(raw_outputs)})"
            )

        credits_used = TRYON_MAX_CREDITS[(generation_mode, resolution)] * num_images
        return FashnResult(
            job_id=job_id,
            output_urls=outputs,
            model_name=TRYON_MAX_MODEL,
            cost_usd=0.0,
            latency_s=time.monotonic() - start,
            raw_status=status_body,
            credits_used=credits_used,
        )


_BEARER_TOKEN_RE = __import__("re").compile(
    r"Bearer\s+[A-Za-z0-9._\-+/=]+", __import__("re").IGNORECASE
)


def _scrub_credentials(text: str) -> str:
    """Strip Bearer-token patterns from error text — defense in depth.

    FASHN error bodies sometimes echo the request including the Authorization
    header. CR finding 2026-05-22 MEDIUM. Never trust an upstream not to leak.
    """
    return _BEARER_TOKEN_RE.sub("Bearer [REDACTED]", text or "")


def _safe_error_excerpt(response: httpx.Response) -> str:
    """Extract a clean error excerpt without leaking auth tokens or HTML pages."""
    try:
        body = response.json()
        if isinstance(body, dict):
            raw = str(body.get("error") or body.get("message") or body)[:200]
            return _scrub_credentials(raw)
    except Exception:
        pass
    text = response.text
    if "<" in text:
        return "error page returned (HTML, redacted)"
    return _scrub_credentials(text[:200])
