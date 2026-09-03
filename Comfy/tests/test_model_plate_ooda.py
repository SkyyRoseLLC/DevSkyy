from __future__ import annotations

import base64
import importlib.util
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "model_plate_ooda.py"
SPEC = importlib.util.spec_from_file_location("model_plate_ooda", SCRIPT)
assert SPEC and SPEC.loader
model_plate_ooda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(model_plate_ooda)


def _png_bytes(size: tuple[int, int] = (576, 1024)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (112, 112, 112)).save(buffer, format="PNG")
    return buffer.getvalue()


MODEL_PLATE_PNG = _png_bytes()
MODEL_PLATE_DATA_URI = "data:image/png;base64," + base64.b64encode(MODEL_PLATE_PNG).decode("ascii")


def _write(path: Path, content: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return model_plate_ooda.sha256_file(path)


def _contract(root: Path, *, include_approval: bool) -> Path:
    bindings: dict[str, dict[str, str]] = {}
    for name, suffix in (
        ("identity_manifest", ".json"),
        ("founder_identity_approval", ".json"),
        ("front_bust", ".png"),
        ("rear_profile", ".png"),
    ):
        path = root / f"inputs/{name}{suffix}"
        payload = _png_bytes() if suffix == ".png" else f"{name}-bytes".encode()
        digest = _write(path, payload)
        bindings[name] = {"path": str(path.relative_to(root)), "sha256": digest}
    contract = {
        "schema": model_plate_ooda.CONTRACT_SCHEMA,
        "plate_id": "LH-MODEL-01-FULL-BODY-A1",
        "identity_id": "LH-MODEL-01",
        "model": {"id": "model-create", "lifecycle": "experimental"},
        "inputs": bindings,
        "request": {
            "face_reference_input": "front_bust",
            "face_reference_mode": "match_base",
            "prompt": "Full-body straight-on studio casting plate.",
            "aspect_ratio": "9:16",
            "resolution": "1k",
            "generation_mode": "fast",
            "seed": 42,
            "num_images": 1,
            "output_format": "png",
            "return_base64": True,
        },
        "credit_control": {
            "max_paid_generations": 1,
            "paid_generations_recorded": 0,
            "automatic_paid_retries": False,
            "max_credits": 4,
            "approval_receipt": None,
        },
        "review_gate": {
            "candidate_author": "product-fidelity-image-edits",
            "required_checks": ["identity", "framing", "anatomy", "pose", "garment", "background"],
        },
        "output": {
            "candidate_path": "outputs/model.png",
            "execution_receipt": "receipts/model-execution.json",
        },
    }
    if include_approval:
        approval_path = root / "receipts/model-approval.json"
        approval = {
            "schema": model_plate_ooda.APPROVAL_SCHEMA,
            "plate_id": contract["plate_id"],
            "execution_fingerprint": model_plate_ooda.execution_fingerprint(contract),
            "max_credits": 4,
            "approved": True,
            "approved_by": "founder",
            "approved_at": "2026-09-02T00:00:00Z",
        }
        approval_path.parent.mkdir(parents=True, exist_ok=True)
        approval_path.write_text(json.dumps(approval), encoding="utf-8")
        contract["credit_control"]["approval_receipt"] = {
            "path": str(approval_path.relative_to(root)),
            "sha256": model_plate_ooda.sha256_file(approval_path),
        }
    contract_path = root / "contract.json"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path


def _complete_execution_receipt(
    root: Path, contract_path: Path, contract: dict, candidate_hash: str
) -> dict:
    source_hashes = {
        name: contract["inputs"][name]["sha256"]
        for name in ("identity_manifest", "founder_identity_approval", "front_bust", "rear_profile")
    }
    marker_path = model_plate_ooda._write_attempt_marker(
        contract=contract,
        contract_sha256=model_plate_ooda.sha256_file(contract_path),
        fingerprint=model_plate_ooda.execution_fingerprint(contract),
        source_sha256s=source_hashes,
    )
    return {
        "schema": model_plate_ooda.RECEIPT_SCHEMA,
        "created_at": "2026-09-02T00:00:01+00:00",
        "plate_id": contract["plate_id"],
        "identity_id": contract["identity_id"],
        "contract_sha256": model_plate_ooda.sha256_file(contract_path),
        "execution_fingerprint": model_plate_ooda.execution_fingerprint(contract),
        "approval_receipt": contract["credit_control"]["approval_receipt"]["path"],
        "approval_receipt_sha256": model_plate_ooda.sha256_file(
            root / contract["credit_control"]["approval_receipt"]["path"]
        ),
        "provider": "fashn",
        "requested_model": model_plate_ooda.MODEL_CREATE_MODEL,
        "model_lifecycle": model_plate_ooda.MODEL_CREATE_LIFECYCLE,
        "job_id": "model-review-job",
        "request_parameters": contract["request"],
        "source_sha256s": source_hashes,
        "output_path": contract["output"]["candidate_path"],
        "output_sha256": candidate_hash,
        "output_dimensions": {"width": 576, "height": 1024},
        "contracted_credits": 4,
        "provider_reported_credits": 4,
        "balance_before": 8,
        "balance_after": 4,
        "balance_after_status": "RECORDED",
        "provider_latency_seconds": 1.0,
        "candidate_only": True,
        "founder_authority": None,
        "output_transport": "base64_png",
        "attempt_marker": str(marker_path.relative_to(root)),
        "attempt_marker_sha256": model_plate_ooda.sha256_file(marker_path),
    }


@pytest.mark.asyncio
async def test_preflight_is_zero_credit_and_pending_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_plate_ooda, "PROJECT_ROOT", tmp_path)
    contract_path = _contract(tmp_path, include_approval=False)

    @dataclass
    class Balance:
        total: int = 8
        subscription: int = 8
        on_demand: int = 0

    class FakeClient:
        paid_calls = 0

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get_credits(self) -> Balance:
            return Balance()

        async def run_model_create(self, **_: object) -> None:
            self.paid_calls += 1

    fake = FakeClient()

    class Factory:
        @classmethod
        def from_default(cls) -> FakeClient:
            return fake

    monkeypatch.setattr(model_plate_ooda, "FashnClient", Factory)
    result = await model_plate_ooda.preflight(contract_path)

    assert result["status"] == "PASS_PENDING_APPROVAL"
    assert result["execution_ready"] is False
    assert result["expected_credits"] == 4
    assert result["fashn_balance"]["total"] == 8
    assert fake.paid_calls == 0


@pytest.mark.asyncio
async def test_execute_writes_candidate_and_provider_credit_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_plate_ooda, "PROJECT_ROOT", tmp_path)
    contract_path = _contract(tmp_path, include_approval=True)

    @dataclass
    class Balance:
        total: int
        subscription: int
        on_demand: int

    @dataclass
    class Result:
        job_id: str = "model-job-1"
        output_urls: tuple[str, ...] = (MODEL_PLATE_DATA_URI,)
        expected_credits: int = 4
        actual_credits: int = 4
        latency_s: float = 2.5

    class FakeClient:
        credit_calls = 0
        request: dict[str, object] | None = None

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get_credits(self) -> Balance:
            self.credit_calls += 1
            total = 8 if self.credit_calls == 1 else 4
            return Balance(total=total, subscription=total, on_demand=0)

        async def run_model_create(self, **kwargs: object) -> Result:
            self.request = kwargs
            return Result()

    fake = FakeClient()

    class Factory:
        @classmethod
        def from_default(cls) -> FakeClient:
            return fake

    monkeypatch.setattr(model_plate_ooda, "FashnClient", Factory)
    receipt = await model_plate_ooda.execute(contract_path, allow_spend=True)

    assert receipt["job_id"] == "model-job-1"
    assert receipt["contracted_credits"] == 4
    assert receipt["provider_reported_credits"] == 4
    assert receipt["balance_before"] == 8
    assert receipt["balance_after"] == 4
    assert receipt["output_dimensions"] == {"width": 576, "height": 1024}
    assert receipt["candidate_only"] is True
    assert fake.request is not None
    assert fake.request["face_reference_mode"] == "match_base"
    assert fake.request["aspect_ratio"] == "9:16"
    assert fake.request["resolution"] == "1k"
    assert fake.request["generation_mode"] == "fast"
    assert (tmp_path / "receipts/model-execution-paid-attempt.json").is_file()


@pytest.mark.asyncio
async def test_model_create_attempt_is_consumed_before_ambiguous_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_plate_ooda, "PROJECT_ROOT", tmp_path)
    contract_path = _contract(tmp_path, include_approval=True)

    @dataclass
    class Balance:
        total: int = 8
        subscription: int = 8
        on_demand: int = 0

    class FakeClient:
        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get_credits(self) -> Balance:
            return Balance()

        async def run_model_create(self, **_: object) -> None:
            raise RuntimeError("ambiguous provider outcome")

    class Factory:
        @classmethod
        def from_default(cls) -> FakeClient:
            return FakeClient()

    monkeypatch.setattr(model_plate_ooda, "FashnClient", Factory)
    with pytest.raises(RuntimeError, match="ambiguous"):
        await model_plate_ooda.execute(contract_path, allow_spend=True)

    marker = tmp_path / "receipts/model-execution-paid-attempt.json"
    assert marker.is_file()
    assert model_plate_ooda.validate(contract_path, require_approval=True)["status"] == "BLOCKED"


def test_review_requires_independent_candidate_bound_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(model_plate_ooda, "PROJECT_ROOT", tmp_path)
    contract_path = _contract(tmp_path, include_approval=True)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    candidate = tmp_path / contract["output"]["candidate_path"]
    _write(candidate, MODEL_PLATE_PNG)
    receipt_path = tmp_path / contract["output"]["execution_receipt"]
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(
        json.dumps(
            _complete_execution_receipt(
                tmp_path,
                contract_path,
                contract,
                model_plate_ooda.sha256_file(candidate),
            )
        ),
        encoding="utf-8",
    )
    review_path = tmp_path / "receipts/review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema": model_plate_ooda.REVIEW_SCHEMA,
                "plate_id": contract["plate_id"],
                "contract_sha256": model_plate_ooda.sha256_file(contract_path),
                "candidate_sha256": model_plate_ooda.sha256_file(candidate),
                "reviewer": "visual-commerce-qa",
                "verdict": "PASS",
                "findings": [],
                "checks": dict.fromkeys(contract["review_gate"]["required_checks"], "PASS"),
            }
        ),
        encoding="utf-8",
    )

    result = model_plate_ooda.review(
        contract_path,
        candidate_path=candidate,
        execution_receipt_path=receipt_path,
        review_path=review_path,
    )

    assert result["status"] == "PASS"
    assert result["authority_state"] == "PENDING_FOUNDER_APPROVAL"
    assert result["candidate_only"] is True

    receipt_path.write_text(
        json.dumps(
            {
                "schema": model_plate_ooda.RECEIPT_SCHEMA,
                "contract_sha256": model_plate_ooda.sha256_file(contract_path),
                "output_sha256": model_plate_ooda.sha256_file(candidate),
            }
        ),
        encoding="utf-8",
    )
    forged = model_plate_ooda.review(
        contract_path,
        candidate_path=candidate,
        execution_receipt_path=receipt_path,
        review_path=review_path,
    )
    assert forged["status"] == "BLOCKED"
    assert any("execution receipt" in failure for failure in forged["failures"])
