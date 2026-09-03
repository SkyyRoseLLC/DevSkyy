"""Tests for skyyrose/integrations/fashn_client.py.

Covers:
    - Credentials: env loading, repr/str redaction (no secret leak)
    - HTTP client: retry on 429/503, success path, poll-until-done
    - Error paths: HTTP error, job failure, timeout, no outputs
    - Cost calculation
    - run_tryon input validation
"""

from __future__ import annotations

import base64
import io
import json

import httpx
import pytest
from PIL import Image

from skyyrose.integrations.fashn_client import (
    COST_PER_SAMPLE_USD,
    FASHN_CREDITS_HEADER,
    FASHN_KEYCHAIN_SERVICE,
    KEYCHAIN_EXECUTABLE,
    MODEL_CREATE_MODEL,
    TRYON_MAX_CREDITS,
    TRYON_MAX_MODEL,
    FashnClient,
    FashnCredits,
    FashnCredentials,
    FashnError,
    FashnResult,
    _safe_error_excerpt,
)


def _png_data_uri(size: tuple[int, int] = (1536, 2736)) -> str:
    buffer = io.BytesIO()
    Image.new("RGB", size, (30, 30, 30)).save(buffer, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")


PNG_DATA_URI = _png_data_uri()


async def _zero_sleep(_: float) -> None:
    return None


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


class TestCredentials:
    def test_loads_from_env(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FASHN_API_KEY", "fa_test_secret_abc123")
        creds = FashnCredentials.from_env()
        assert creds.api_key == "fa_test_secret_abc123"

    def test_missing_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("FASHN_API_KEY", raising=False)
        with pytest.raises(KeyError, match="FASHN_API_KEY not set"):
            FashnCredentials.from_env()

    def test_empty_env_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("FASHN_API_KEY", "")
        with pytest.raises(KeyError):
            FashnCredentials.from_env()

    def test_repr_redacts_key(self) -> None:
        creds = FashnCredentials(api_key="fa_test_secret_abc123")
        assert "fa_test_secret_abc123" not in repr(creds)

    def test_str_redacts_key(self) -> None:
        creds = FashnCredentials(api_key="fa_test_secret_abc123")
        assert "fa_test_secret_abc123" not in str(creds)
        assert "<redacted>" in str(creds)

    def test_loads_from_keychain_without_exposing_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("skyyrose.integrations.fashn_client.sys.platform", "darwin")
        monkeypatch.setattr("skyyrose.integrations.fashn_client.os.path.isfile", lambda _: True)

        class Completed:
            returncode = 0
            stdout = "fa_keychain_secret\n"
            stderr = ""

        calls: list[list[str]] = []

        def fake_run(args: list[str], **_: object) -> Completed:
            calls.append(args)
            return Completed()

        monkeypatch.setattr("skyyrose.integrations.fashn_client.subprocess.run", fake_run)
        creds = FashnCredentials.from_keychain(account="theceo")

        assert creds.api_key == "fa_keychain_secret"
        assert calls == [
            [
                KEYCHAIN_EXECUTABLE,
                "find-generic-password",
                "-s",
                FASHN_KEYCHAIN_SERVICE,
                "-a",
                "theceo",
                "-w",
            ]
        ]
        assert "fa_keychain_secret" not in repr(creds)

    def test_missing_keychain_item_fails_without_secret_text(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("skyyrose.integrations.fashn_client.sys.platform", "darwin")
        monkeypatch.setattr("skyyrose.integrations.fashn_client.os.path.isfile", lambda _: True)

        class Completed:
            returncode = 44
            stdout = ""
            stderr = "security: SecKeychainSearchCopyNext: item not found"

        monkeypatch.setattr(
            "skyyrose.integrations.fashn_client.subprocess.run",
            lambda *args, **kwargs: Completed(),
        )
        with pytest.raises(KeyError, match="skyyrose-fashn-api") as exc:
            FashnCredentials.from_keychain(account="theceo")
        assert "SecKeychain" not in str(exc.value)


# ---------------------------------------------------------------------------
# FakeTransport + client builder
# ---------------------------------------------------------------------------


class FakeTransport(httpx.AsyncBaseTransport):
    """Returns canned responses in sequence; records every request."""

    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.calls: list[httpx.Request] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(request)
        if not self._responses:
            raise RuntimeError("FakeTransport exhausted")
        return self._responses.pop(0)


def _make_client(transport: FakeTransport, poll_interval: float = 0.001) -> FashnClient:
    return FashnClient(
        FashnCredentials(api_key="fa_test"),
        timeout_seconds=5.0,
        poll_interval_seconds=poll_interval,
        max_poll_seconds=10.0,
        transport=transport,
    )


def _ok_response(
    json_body: dict,
    status: int = 200,
    *,
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    return httpx.Response(status, content=json.dumps(json_body).encode(), headers=headers)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestRunTryonHappyPath:
    @pytest.mark.asyncio
    async def test_completes_on_first_poll(self) -> None:
        transport = FakeTransport(
            [
                _ok_response({"id": "job-1"}),  # /run
                _ok_response(
                    {"status": "completed", "output": ["https://output.example/img1.jpg"]}
                ),
            ]
        )
        async with _make_client(transport) as client:
            result = await client.run_tryon(
                model_image_url="https://x/model.jpg",
                garment_image_url="https://x/garment.jpg",
                sleep=_zero_sleep,
            )
        assert isinstance(result, FashnResult)
        assert result.job_id == "job-1"
        assert result.output_urls == ["https://output.example/img1.jpg"]
        assert result.cost_usd == 0.075  # 1 sample × $0.075
        assert len(transport.calls) == 2

    @pytest.mark.asyncio
    async def test_polls_until_complete(self) -> None:
        transport = FakeTransport(
            [
                _ok_response({"id": "job-2"}),
                _ok_response({"status": "in_queue"}),
                _ok_response({"status": "processing"}),
                _ok_response({"status": "completed", "output": ["https://x/out.jpg"]}),
            ]
        )
        async with _make_client(transport) as client:
            result = await client.run_tryon(
                model_image_url="https://x/m.jpg",
                garment_image_url="https://x/g.jpg",
                sleep=_zero_sleep,
            )
        assert result.job_id == "job-2"
        assert len(transport.calls) == 4

    @pytest.mark.asyncio
    async def test_cost_scales_with_samples(self) -> None:
        transport = FakeTransport(
            [
                _ok_response({"id": "j"}),
                _ok_response(
                    {
                        "status": "completed",
                        "output": [
                            "https://x/a.jpg",
                            "https://x/b.jpg",
                            "https://x/c.jpg",
                            "https://x/d.jpg",
                        ],
                    }
                ),
            ]
        )
        async with _make_client(transport) as client:
            result = await client.run_tryon(
                model_image_url="https://x/m",
                garment_image_url="https://x/g",
                num_samples=4,
                sleep=_zero_sleep,
            )
        assert result.cost_usd == 0.3  # 4 × 0.075


class TestRunTryonMax:
    @pytest.mark.asyncio
    async def test_uses_current_contract_and_reports_credits(self) -> None:
        output = PNG_DATA_URI
        transport = FakeTransport(
            [
                _ok_response({"id": "max-1"}),
                _ok_response(
                    {"status": "completed", "output": [output]},
                    headers={FASHN_CREDITS_HEADER: "4"},
                ),
            ]
        )
        async with _make_client(transport) as client:
            result = await client.run_tryon_max(
                model_image="data:image/png;base64,bW9kZWw=",
                product_image="data:image/png;base64,cHJvZHVjdA==",
                prompt="Replace only the lower-body garment.",
                resolution="2k",
                generation_mode="quality",
                seed=17,
                num_images=1,
                output_format="png",
                return_base64=True,
                sleep=_zero_sleep,
            )

        assert result.model_name == TRYON_MAX_MODEL
        assert result.output_urls == [output]
        assert result.cost_usd == 0.0
        assert result.credits_used == 4
        assert result.expected_credits == 4
        assert result.actual_credits == 4
        request = json.loads(transport.calls[0].content)
        assert request == {
            "model_name": "tryon-max",
            "inputs": {
                "product_image": "data:image/png;base64,cHJvZHVjdA==",
                "model_image": "data:image/png;base64,bW9kZWw=",
                "prompt": "Replace only the lower-body garment.",
                "resolution": "2k",
                "generation_mode": "quality",
                "seed": 17,
                "num_images": 1,
                "output_format": "png",
                "return_base64": True,
            },
        }

    @pytest.mark.asyncio
    async def test_submission_network_error_is_not_retried(self) -> None:
        class AmbiguousSubmissionTransport(httpx.AsyncBaseTransport):
            def __init__(self) -> None:
                self.calls = 0

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                self.calls += 1
                raise httpx.ConnectError("connection lost after submit", request=request)

        transport = AmbiguousSubmissionTransport()
        client = FashnClient(
            FashnCredentials(api_key="fa_test"),
            timeout_seconds=5.0,
            transport=transport,
        )
        async with client:
            with pytest.raises(FashnError, match="automatic retry is disabled"):
                await client.run_tryon_max(
                    model_image="data:image/png;base64,bW9kZWw=",
                    product_image="data:image/png;base64,cHJvZHVjdA==",
                )
        assert transport.calls == 1

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ("kwargs", "message"),
        [
            ({"resolution": "8k"}, "resolution"),
            ({"generation_mode": "automatic"}, "generation_mode"),
            ({"seed": -1}, "seed"),
            ({"num_images": 5}, "num_images"),
            ({"output_format": "webp"}, "output_format"),
        ],
    )
    async def test_rejects_invalid_parameters(
        self, kwargs: dict[str, object], message: str
    ) -> None:
        transport = FakeTransport([])
        async with _make_client(transport) as client:
            with pytest.raises((TypeError, ValueError), match=message):
                await client.run_tryon_max(
                    model_image="data:image/png;base64,bW9kZWw=",
                    product_image="data:image/png;base64,cHJvZHVjdA==",
                    **kwargs,  # type: ignore[arg-type]
                )
        assert not transport.calls

    @pytest.mark.asyncio
    async def test_missing_provider_credit_header_blocks_acceptance(self) -> None:
        transport = FakeTransport(
            [
                _ok_response({"id": "max-no-usage"}),
                _ok_response({"status": "completed", "output": [PNG_DATA_URI]}),
            ]
        )
        async with _make_client(transport) as client:
            with pytest.raises(FashnError, match=FASHN_CREDITS_HEADER):
                await client.run_tryon_max(
                    model_image="data:image/png;base64,bW9kZWw=",
                    product_image="data:image/png;base64,cHJvZHVjdA==",
                    sleep=_zero_sleep,
                )

    @pytest.mark.asyncio
    async def test_mismatched_provider_credit_header_blocks_acceptance(self) -> None:
        transport = FakeTransport(
            [
                _ok_response({"id": "max-wrong-usage"}),
                _ok_response(
                    {"status": "completed", "output": [PNG_DATA_URI]},
                    headers={FASHN_CREDITS_HEADER: "3"},
                ),
            ]
        )
        async with _make_client(transport) as client:
            with pytest.raises(FashnError, match="reported 3 credits"):
                await client.run_tryon_max(
                    model_image="data:image/png;base64,bW9kZWw=",
                    product_image="data:image/png;base64,cHJvZHVjdA==",
                    sleep=_zero_sleep,
                )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "outputs",
        [
            [PNG_DATA_URI, "https://example.test/unexpected.png"],
            ["data:image/png;base64,%%%"],
            ["data:image/jpeg;base64,/9j/4AAQ"],
            ["data:image/png;base64,aW1hZ2U="],
            [
                "data:image/png;base64,"
                + base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR" + b"\x00" * 8).decode(
                    "ascii"
                )
            ],
            [_png_data_uri((1, 1))],
        ],
    )
    async def test_base64_mode_requires_exact_exclusive_valid_png_outputs(
        self, outputs: list[str]
    ) -> None:
        transport = FakeTransport(
            [
                _ok_response({"id": "max-invalid-output"}),
                _ok_response(
                    {"status": "completed", "output": outputs},
                    headers={FASHN_CREDITS_HEADER: "4"},
                ),
            ]
        )
        async with _make_client(transport) as client:
            with pytest.raises(FashnError):
                await client.run_tryon_max(
                    model_image="data:image/png;base64,bW9kZWw=",
                    product_image="data:image/png;base64,cHJvZHVjdA==",
                    resolution="2k",
                    generation_mode="quality",
                    sleep=_zero_sleep,
                )


class TestCreditsAndModelCreate:
    @pytest.mark.asyncio
    async def test_get_credits_parses_typed_balance(self) -> None:
        transport = FakeTransport(
            [_ok_response({"credits": {"total": 12, "subscription": 8, "on_demand": 4}})]
        )
        async with _make_client(transport) as client:
            balance = await client.get_credits(sleep=_zero_sleep)
        assert balance == FashnCredits(total=12, subscription=8, on_demand=4)
        assert transport.calls[0].method == "GET"
        assert transport.calls[0].url.path == "/v1/credits"

    @pytest.mark.asyncio
    async def test_model_create_exact_request_and_credit_evidence(self) -> None:
        output = PNG_DATA_URI
        transport = FakeTransport(
            [
                _ok_response({"id": "model-1"}),
                _ok_response(
                    {"status": "completed", "output": [output]},
                    headers={FASHN_CREDITS_HEADER: "4"},
                ),
            ]
        )
        async with _make_client(transport) as client:
            result = await client.run_model_create(
                prompt="Full-body studio casting plate.",
                face_reference="data:image/png;base64,ZmFjZQ==",
                face_reference_mode="match_base",
                aspect_ratio="9:16",
                resolution="1k",
                generation_mode="fast",
                seed=42,
                num_images=1,
                output_format="png",
                return_base64=True,
                sleep=_zero_sleep,
            )

        assert result.model_name == MODEL_CREATE_MODEL
        assert result.expected_credits == 4
        assert result.actual_credits == 4
        request = json.loads(transport.calls[0].content)
        assert request == {
            "model_name": "model-create",
            "inputs": {
                "prompt": "Full-body studio casting plate.",
                "aspect_ratio": "9:16",
                "resolution": "1k",
                "generation_mode": "fast",
                "seed": 42,
                "num_images": 1,
                "output_format": "png",
                "return_base64": True,
                "face_reference": "data:image/png;base64,ZmFjZQ==",
                "face_reference_mode": "match_base",
            },
        }


# ---------------------------------------------------------------------------
# Failure modes — never silent
# ---------------------------------------------------------------------------


class TestRunTryonFailureModes:
    @pytest.mark.asyncio
    async def test_run_http_error_raises_fashn_error(self) -> None:
        transport = FakeTransport([_ok_response({"error": "invalid api key"}, status=401)])
        async with _make_client(transport) as client:
            with pytest.raises(FashnError, match="HTTP 401"):
                await client.run_tryon(
                    model_image_url="https://x/m", garment_image_url="https://x/g"
                )

    @pytest.mark.asyncio
    async def test_status_failed_raises(self) -> None:
        transport = FakeTransport(
            [
                _ok_response({"id": "j"}),
                _ok_response({"status": "failed", "error": "garment_too_small"}),
            ]
        )
        async with _make_client(transport) as client:
            with pytest.raises(FashnError, match="garment_too_small|did not complete"):
                await client.run_tryon(
                    model_image_url="https://x/m",
                    garment_image_url="https://x/g",
                    sleep=_zero_sleep,
                )

    @pytest.mark.asyncio
    async def test_empty_output_raises(self) -> None:
        transport = FakeTransport(
            [
                _ok_response({"id": "j"}),
                _ok_response({"status": "completed", "output": []}),
            ]
        )
        async with _make_client(transport) as client:
            with pytest.raises(
                FashnError, match="returned no valid output URLs|returned no outputs"
            ):
                await client.run_tryon(
                    model_image_url="https://x/m",
                    garment_image_url="https://x/g",
                    sleep=_zero_sleep,
                )

    @pytest.mark.asyncio
    async def test_no_job_id_raises(self) -> None:
        transport = FakeTransport([_ok_response({"weird": "response"})])
        async with _make_client(transport) as client:
            with pytest.raises(FashnError, match="no job id"):
                await client.run_tryon(
                    model_image_url="https://x/m", garment_image_url="https://x/g"
                )

    @pytest.mark.asyncio
    async def test_unknown_status_raises(self) -> None:
        transport = FakeTransport(
            [
                _ok_response({"id": "j"}),
                _ok_response({"status": "frobnicating"}),  # Not in any state set
            ]
        )
        async with _make_client(transport) as client:
            with pytest.raises(FashnError, match="unknown FASHN status"):
                await client.run_tryon(
                    model_image_url="https://x/m",
                    garment_image_url="https://x/g",
                    sleep=_zero_sleep,
                )

    @pytest.mark.asyncio
    async def test_num_samples_validation(self) -> None:
        transport = FakeTransport([])
        async with _make_client(transport) as client:
            with pytest.raises(ValueError, match="num_samples"):
                await client.run_tryon(
                    model_image_url="https://x/m",
                    garment_image_url="https://x/g",
                    num_samples=0,
                )
            with pytest.raises(ValueError, match="num_samples"):
                await client.run_tryon(
                    model_image_url="https://x/m",
                    garment_image_url="https://x/g",
                    num_samples=17,
                )


# ---------------------------------------------------------------------------
# Retry semantics
# ---------------------------------------------------------------------------


class TestRetrySemantics:
    @pytest.mark.asyncio
    async def test_retries_429_on_run(self) -> None:
        transport = FakeTransport(
            [
                httpx.Response(429),
                httpx.Response(429),
                _ok_response({"id": "j"}),
                _ok_response({"status": "completed", "output": ["https://x/out.jpg"]}),
            ]
        )
        async with _make_client(transport) as client:
            result = await client.run_tryon(
                model_image_url="https://x/m",
                garment_image_url="https://x/g",
                sleep=_zero_sleep,
            )
        assert result.job_id == "j"
        assert len(transport.calls) == 4

    @pytest.mark.asyncio
    async def test_429_exhausted_returns_response(self) -> None:
        transport = FakeTransport([httpx.Response(429) for _ in range(10)])
        async with _make_client(transport) as client:
            with pytest.raises(FashnError, match="HTTP 429"):
                await client.run_tryon(
                    model_image_url="https://x/m",
                    garment_image_url="https://x/g",
                    sleep=_zero_sleep,
                )

    @pytest.mark.asyncio
    async def test_network_error_retried(self) -> None:
        class FlakyTransport(httpx.AsyncBaseTransport):
            def __init__(self) -> None:
                self.calls = 0
                self.responses = [
                    _ok_response({"id": "j"}),
                    _ok_response({"status": "completed", "output": ["https://x/out.jpg"]}),
                ]

            async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
                self.calls += 1
                if self.calls == 1:
                    raise httpx.ConnectError("connection refused")
                return self.responses.pop(0)

        transport = FlakyTransport()
        client = FashnClient(
            FashnCredentials(api_key="fa_test"),
            timeout_seconds=5.0,
            poll_interval_seconds=0.001,
            max_poll_seconds=10.0,
            transport=transport,
        )
        async with client:
            result = await client.run_tryon(
                model_image_url="https://x/m",
                garment_image_url="https://x/g",
                sleep=_zero_sleep,
            )
        assert result.job_id == "j"
        assert transport.calls >= 2


# ---------------------------------------------------------------------------
# Cost constants + error helper
# ---------------------------------------------------------------------------


class TestCostConstants:
    def test_tryon_cost(self) -> None:
        assert COST_PER_SAMPLE_USD["tryon-v1.6"] == 0.075

    def test_bg_remove_cost(self) -> None:
        assert COST_PER_SAMPLE_USD["bg-remove-v1"] == 0.025

    def test_tryon_max_credit_matrix(self) -> None:
        assert TRYON_MAX_CREDITS[("fast", "1k")] == 1
        assert TRYON_MAX_CREDITS[("quality", "4k")] == 5


class TestSafeErrorExcerpt:
    def test_json_error_message(self) -> None:
        response = _ok_response({"error": "invalid api key"}, status=401)
        assert "invalid api key" in _safe_error_excerpt(response)

    def test_html_redacted(self) -> None:
        response = httpx.Response(
            500, content=b"<html><body>Stacktrace at /usr/local/.../</body></html>"
        )
        result = _safe_error_excerpt(response)
        assert "stacktrace" not in result.lower()
        assert "redacted" in result.lower()

    def test_plain_text_truncated(self) -> None:
        response = httpx.Response(500, content=b"X" * 500)
        result = _safe_error_excerpt(response)
        assert len(result) <= 200
