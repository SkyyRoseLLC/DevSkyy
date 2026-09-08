from __future__ import annotations

import base64
import importlib.util
import io
import json
from dataclasses import dataclass
from pathlib import Path

import pytest
from PIL import Image

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vto_ooda.py"
SPEC = importlib.util.spec_from_file_location("vto_ooda", SCRIPT)
assert SPEC and SPEC.loader
vto_ooda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vto_ooda)


def _png_bytes(size: tuple[int, int] = (1536, 2736)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", size, (112, 112, 112)).save(buffer, format="PNG")
    return buffer.getvalue()


VTO_CANDIDATE_PNG = _png_bytes()
VTO_CANDIDATE_DATA_URI = "data:image/png;base64," + base64.b64encode(VTO_CANDIDATE_PNG).decode(
    "ascii"
)


def _write(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return vto_ooda.sha256_file(path)


def _binding(root: Path, relative: str, payload: bytes) -> dict[str, object]:
    path = root / relative
    digest = _write(path, payload)
    receipt_path = root / f"receipts/{path.stem}-authority.json"
    receipt_hash = _write(receipt_path, b"{}")
    return {
        "path": relative,
        "sha256": digest,
        "authority_receipt": {
            "path": str(receipt_path.relative_to(root)),
            "sha256": receipt_hash,
        },
    }


def _contract(root: Path, *, include_model: bool, include_approval: bool) -> tuple[Path, dict]:
    registry_path = root / "registry.json"
    registry = {
        "models": [
            {
                "id": "fashn-tryon-max",
                "aliases": ["tryon-max"],
                "allowed_operations": ["virtual_tryon_candidate"],
                "fidelity_tier": "candidate_only",
            }
        ]
    }
    registry_path.write_text(json.dumps(registry), encoding="utf-8")
    product = _binding(root, "inputs/product.png", b"product")
    supporting = []
    for index, view in enumerate(("front", "back", "wearer_left", "wearer_right")):
        binding = _binding(root, f"inputs/view-{index}.jpg", f"view-{index}".encode())
        binding["view"] = view
        supporting.append(binding)
    model = (
        {
            **_binding(root, "inputs/model.png", _png_bytes()),
            "authority_state": vto_ooda.APPROVED_MODEL_AUTHORITY,
        }
        if include_model
        else {
            "path": None,
            "sha256": None,
            "authority_state": "PENDING_FOUNDER_APPROVED_FULL_BODY_MODEL",
            "authority_receipt": None,
        }
    )
    contract = {
        "schema": vto_ooda.CONTRACT_SCHEMA,
        "pilot_id": "TEST-VTO-A1",
        "scene_id": "TEST-SCENE",
        "sku": "test-001",
        "model": {
            "id": "tryon-max",
            "lifecycle": "preview",
            "registry": {
                "path": "registry.json",
                "sha256": vto_ooda.sha256_file(registry_path),
            },
        },
        "inputs": {
            "product": product,
            "supporting_product_views": supporting,
            "model": model,
        },
        "request": {
            "prompt": "Replace only the lower-body garment.",
            "resolution": "2k",
            "generation_mode": "quality",
            "seed": 42,
            "num_images": 1,
            "output_format": "png",
            "return_base64": True,
        },
        "product_invariants": ["a", "b", "c", "d"],
        "credit_control": {
            "max_paid_generations": 1,
            "paid_generations_recorded": 0,
            "automatic_paid_retries": False,
            "max_credits": 4,
            "approval_receipt": None,
        },
        "review_gate": {
            "candidate_author": "product-fidelity-image-edits",
            "required_checks": ["identity", "construction", "pockets", "artwork"],
            "pocket_evidence": {
                "wearer_left_side": {
                    "candidate_proof": "DIRECTLY_VISIBLE_ZIPPERED",
                    "required_disposition": "PASS",
                },
                "wearer_right_side": {
                    "candidate_proof": "DIRECTLY_VISIBLE_ZIPPERED",
                    "required_disposition": "PASS",
                },
                "wearer_left_rear": {
                    "candidate_proof": "NOT_OBSERVABLE_IN_FRONT_CANDIDATE",
                    "source_authority": "PASS",
                },
                "wearer_right_rear": {
                    "candidate_proof": "NOT_OBSERVABLE_IN_FRONT_CANDIDATE",
                    "source_authority": "PASS",
                },
            },
        },
        "output": {
            "candidate_path": "outputs/candidate.png",
            "execution_receipt": "receipts/execution.json",
        },
    }
    contract_path = root / "contract.json"
    if include_approval:
        approval_path = root / "receipts/approval.json"
        approval = {
            "schema": vto_ooda.APPROVAL_SCHEMA,
            "pilot_id": contract["pilot_id"],
            "execution_fingerprint": vto_ooda.execution_fingerprint(contract),
            "max_credits": 4,
            "approved": True,
            "approved_by": "founder",
            "approved_at": "2026-09-02T00:00:00Z",
        }
        approval_path.parent.mkdir(parents=True, exist_ok=True)
        approval_path.write_text(json.dumps(approval), encoding="utf-8")
        contract["credit_control"]["approval_receipt"] = {
            "path": str(approval_path.relative_to(root)),
            "sha256": vto_ooda.sha256_file(approval_path),
        }
    contract_path.write_text(json.dumps(contract), encoding="utf-8")
    return contract_path, contract


def _complete_execution_receipt(
    root: Path, contract_path: Path, contract: dict, candidate_hash: str
) -> dict:
    source_hashes = {
        "product": contract["inputs"]["product"]["sha256"],
        "model": contract["inputs"]["model"]["sha256"],
        **{
            f"supporting_{binding['view']}": binding["sha256"]
            for binding in contract["inputs"]["supporting_product_views"]
        },
    }
    marker_path = vto_ooda._write_attempt_marker(
        contract=contract,
        contract_sha256=vto_ooda.sha256_file(contract_path),
        fingerprint=vto_ooda.execution_fingerprint(contract),
        source_sha256s=source_hashes,
    )
    return {
        "schema": vto_ooda.RECEIPT_SCHEMA,
        "created_at": "2026-09-02T00:00:01+00:00",
        "pilot_id": contract["pilot_id"],
        "scene_id": contract["scene_id"],
        "sku": contract["sku"],
        "contract_sha256": vto_ooda.sha256_file(contract_path),
        "execution_fingerprint": vto_ooda.execution_fingerprint(contract),
        "approval_receipt": contract["credit_control"]["approval_receipt"]["path"],
        "approval_receipt_sha256": vto_ooda.sha256_file(
            root / contract["credit_control"]["approval_receipt"]["path"]
        ),
        "provider": "fashn",
        "requested_model": vto_ooda.TRYON_MAX_MODEL,
        "model_lifecycle": vto_ooda.TRYON_MAX_LIFECYCLE,
        "job_id": "review-job",
        "request_parameters": contract["request"],
        "product_sha256": source_hashes["product"],
        "model_sha256": source_hashes["model"],
        "source_sha256s": source_hashes,
        "output_path": contract["output"]["candidate_path"],
        "output_sha256": candidate_hash,
        "output_dimensions": {"width": 1536, "height": 2736},
        "contracted_credits": 4,
        "provider_reported_credits": 4,
        "credits_used": 4,
        "balance_before": 8,
        "balance_after": 4,
        "balance_after_status": "RECORDED",
        "provider_latency_seconds": 1.0,
        "candidate_only": True,
        "independent_review_required": True,
        "founder_approval_required": True,
        "promotion_authorized": False,
        "output_transport": "base64_png",
        "attempt_marker": str(marker_path.relative_to(root)),
        "attempt_marker_sha256": vto_ooda.sha256_file(marker_path),
    }


def test_validate_blocks_missing_full_body_model_and_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, _ = _contract(tmp_path, include_model=False, include_approval=False)

    result = vto_ooda.validate(contract_path)

    assert result["status"] == "BLOCKED"
    assert result["execution_ready"] is False
    assert "MISSING_PATH" in result["blockers"]
    assert "FULL_BODY_MODEL_NOT_APPROVED" in result["blockers"]
    assert "MISSING_PAID_APPROVAL_RECEIPT" in result["blockers"]


def test_validate_passes_complete_candidate_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, _ = _contract(tmp_path, include_model=True, include_approval=True)

    result = vto_ooda.validate(contract_path)

    assert result["status"] == "PASS"
    assert result["execution_ready"] is True
    assert result["technical_ready"] is True
    assert result["estimated_credits"] == 4
    assert result["blockers"] == []


@pytest.mark.asyncio
async def test_prepare_passes_technical_checks_without_approval_or_paid_post(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, _ = _contract(tmp_path, include_model=True, include_approval=False)

    @dataclass
    class FakeBalance:
        total: int = 8
        subscription: int = 8
        on_demand: int = 0

    class FakeClient:
        paid_calls = 0

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get_credits(self) -> FakeBalance:
            return FakeBalance()

        async def run_tryon_max(self, **_: object) -> None:
            self.paid_calls += 1

    fake = FakeClient()

    class FakeClientFactory:
        @classmethod
        def from_default(cls) -> FakeClient:
            return fake

    monkeypatch.setattr(vto_ooda, "FashnClient", FakeClientFactory)

    result = await vto_ooda.prepare(contract_path)

    assert result["status"] == "PASS_PENDING_APPROVAL"
    assert result["technical_ready"] is True
    assert result["execution_ready"] is False
    assert result["execution_fingerprint"]
    assert result["fashn_balance"]["total"] == 8
    assert fake.paid_calls == 0


@pytest.mark.asyncio
async def test_prepare_blocks_insufficient_credits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, _ = _contract(tmp_path, include_model=True, include_approval=False)

    @dataclass
    class FakeBalance:
        total: int = 3
        subscription: int = 3
        on_demand: int = 0

    class FakeClient:
        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get_credits(self) -> FakeBalance:
            return FakeBalance()

    class FakeClientFactory:
        @classmethod
        def from_default(cls) -> FakeClient:
            return FakeClient()

    monkeypatch.setattr(vto_ooda, "FashnClient", FakeClientFactory)
    result = await vto_ooda.prepare(contract_path)

    assert result["status"] == "BLOCKED"
    assert "INSUFFICIENT_FASHN_CREDITS" in result["blockers"]


def test_validate_rejects_duplicate_directional_product_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, contract = _contract(tmp_path, include_model=True, include_approval=True)
    contract["inputs"]["supporting_product_views"][3]["view"] = "front"
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    result = vto_ooda.validate(contract_path)

    assert result["status"] == "BLOCKED"
    assert "INVALID_DIRECTIONAL_PRODUCT_VIEW_SET" in result["blockers"]


def test_contract_change_invalidates_existing_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, contract = _contract(tmp_path, include_model=True, include_approval=True)
    contract["request"]["seed"] = 43
    contract_path.write_text(json.dumps(contract), encoding="utf-8")

    result = vto_ooda.validate(contract_path)

    assert result["status"] == "BLOCKED"
    assert "INVALID_PAID_APPROVAL_RECEIPT" in result["blockers"]


def test_existing_candidate_path_blocks_execution_preflight(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, _ = _contract(tmp_path, include_model=True, include_approval=True)
    _write(tmp_path / "outputs/candidate.png", b"existing")

    result = vto_ooda.validate(contract_path)

    assert result["status"] == "BLOCKED"
    assert "OUTPUT_OR_RECEIPT_ALREADY_EXISTS" in result["blockers"]


def test_execute_requires_explicit_spend_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow-spend"):
        vto_ooda.asyncio.run(vto_ooda.execute(tmp_path / "contract.json", allow_spend=False))


@pytest.mark.asyncio
async def test_execute_writes_candidate_and_hash_bound_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, _ = _contract(tmp_path, include_model=True, include_approval=True)
    png = VTO_CANDIDATE_PNG

    @dataclass
    class FakeResult:
        job_id: str = "job-123"
        output_urls: tuple[str, ...] = (VTO_CANDIDATE_DATA_URI,)
        expected_credits: int = 4
        actual_credits: int = 4
        latency_s: float = 1.25

        @property
        def credits_used(self) -> int:
            return self.actual_credits

    @dataclass
    class FakeBalance:
        total: int
        subscription: int
        on_demand: int

    class FakeClient:
        credit_calls = 0
        request: dict[str, object] | None = None

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get_credits(self) -> FakeBalance:
            self.credit_calls += 1
            return (
                FakeBalance(total=8, subscription=8, on_demand=0)
                if self.credit_calls == 1
                else FakeBalance(total=4, subscription=4, on_demand=0)
            )

        async def run_tryon_max(self, **kwargs: object) -> FakeResult:
            self.request = kwargs
            return FakeResult()

    fake = FakeClient()

    class FakeClientFactory:
        @classmethod
        def from_default(cls) -> FakeClient:
            return fake

    monkeypatch.setattr(vto_ooda, "FashnClient", FakeClientFactory)

    receipt = await vto_ooda.execute(contract_path, allow_spend=True)

    candidate = tmp_path / "outputs/candidate.png"
    assert candidate.read_bytes() == png
    assert receipt["job_id"] == "job-123"
    assert receipt["credits_used"] == 4
    assert receipt["contracted_credits"] == 4
    assert receipt["provider_reported_credits"] == 4
    assert receipt["balance_before"] == 8
    assert receipt["balance_after"] == 4
    assert receipt["output_sha256"] == vto_ooda.sha256_file(candidate)
    assert receipt["output_dimensions"] == {"width": 1536, "height": 2736}
    assert receipt["candidate_only"] is True
    assert receipt["promotion_authorized"] is False
    marker = tmp_path / "receipts/execution-paid-attempt.json"
    assert marker.is_file()
    assert receipt["attempt_marker_sha256"] == vto_ooda.sha256_file(marker)
    assert fake.request is not None
    assert base64.b64decode(str(fake.request["product_image"]).split(",", 1)[1]) == b"product"
    assert base64.b64decode(str(fake.request["model_image"]).split(",", 1)[1]) == _png_bytes()
    assert receipt["product_sha256"] == vto_ooda.sha256_bytes(b"product")
    assert receipt["model_sha256"] == vto_ooda.sha256_bytes(_png_bytes())
    saved = json.loads((tmp_path / "receipts/execution.json").read_text(encoding="utf-8"))
    assert saved == receipt


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider_error",
    ["ambiguous provider outcome", "missing x-fashn-credits-used evidence"],
)
async def test_paid_attempt_is_consumed_before_provider_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider_error: str
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, _ = _contract(tmp_path, include_model=True, include_approval=True)

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

        async def run_tryon_max(self, **_: object) -> None:
            raise RuntimeError(provider_error)

    class Factory:
        @classmethod
        def from_default(cls) -> FakeClient:
            return FakeClient()

    monkeypatch.setattr(vto_ooda, "FashnClient", Factory)
    with pytest.raises(RuntimeError, match=provider_error):
        await vto_ooda.execute(contract_path, allow_spend=True)

    marker = tmp_path / "receipts/execution-paid-attempt.json"
    assert marker.is_file()
    assert vto_ooda.validate(contract_path)["status"] == "BLOCKED"
    with pytest.raises(ValueError, match="preflight blocked"):
        await vto_ooda.execute(contract_path, allow_spend=True)


def test_paid_attempt_marker_is_atomic_across_executors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, contract = _contract(tmp_path, include_model=True, include_approval=True)
    contract_hash = vto_ooda.sha256_file(contract_path)
    fingerprint = vto_ooda.execution_fingerprint(contract)
    source_hashes = {"product": contract["inputs"]["product"]["sha256"]}

    vto_ooda._write_attempt_marker(
        contract=contract,
        contract_sha256=contract_hash,
        fingerprint=fingerprint,
        source_sha256s=source_hashes,
    )
    with pytest.raises(ValueError, match="already been consumed"):
        vto_ooda._write_attempt_marker(
            contract=contract,
            contract_sha256=contract_hash,
            fingerprint=fingerprint,
            source_sha256s=source_hashes,
        )


@pytest.mark.asyncio
async def test_post_call_balance_failure_preserves_candidate_and_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, _ = _contract(tmp_path, include_model=True, include_approval=True)

    @dataclass
    class Balance:
        total: int = 8
        subscription: int = 8
        on_demand: int = 0

    @dataclass
    class Result:
        job_id: str = "job-balance-reconcile"
        output_urls: tuple[str, ...] = (VTO_CANDIDATE_DATA_URI,)
        expected_credits: int = 4
        actual_credits: int = 4
        latency_s: float = 1.0

    class FakeClient:
        calls = 0

        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def get_credits(self) -> Balance:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("balance unavailable")
            return Balance()

        async def run_tryon_max(self, **_: object) -> Result:
            return Result()

    class Factory:
        @classmethod
        def from_default(cls) -> FakeClient:
            return FakeClient()

    monkeypatch.setattr(vto_ooda, "FashnClient", Factory)
    receipt = await vto_ooda.execute(contract_path, allow_spend=True)

    assert receipt["balance_after"] is None
    assert receipt["balance_after_status"] == "RECONCILIATION_REQUIRED"
    assert (tmp_path / "outputs/candidate.png").is_file()


def test_review_requires_independent_exact_check_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, contract = _contract(tmp_path, include_model=True, include_approval=True)
    candidate_path = tmp_path / "outputs/candidate.png"
    _write(candidate_path, VTO_CANDIDATE_PNG)
    candidate_hash = vto_ooda.sha256_file(candidate_path)
    contract_hash = vto_ooda.sha256_file(contract_path)

    execution_path = tmp_path / "receipts/execution.json"
    execution_path.write_text(
        json.dumps(_complete_execution_receipt(tmp_path, contract_path, contract, candidate_hash)),
        encoding="utf-8",
    )
    review_path = tmp_path / "receipts/review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema": vto_ooda.REVIEW_SCHEMA,
                "pilot_id": contract["pilot_id"],
                "contract_sha256": contract_hash,
                "candidate_sha256": candidate_hash,
                "reviewer": "visual-commerce-qa",
                "verdict": "PASS",
                "findings": [],
                "checks": dict.fromkeys(contract["review_gate"]["required_checks"], "PASS"),
                "pocket_evidence": contract["review_gate"]["pocket_evidence"],
            }
        ),
        encoding="utf-8",
    )

    result = vto_ooda.verify_review(
        contract_path,
        candidate_path=candidate_path,
        execution_receipt_path=execution_path,
        review_path=review_path,
    )

    assert result["status"] == "PASS"
    assert result["candidate_only"] is True
    assert result["scene_input_authorized"] is False

    review = json.loads(review_path.read_text(encoding="utf-8"))
    review["reviewer"] = "product-fidelity-image-edits"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    blocked = vto_ooda.verify_review(
        contract_path,
        candidate_path=candidate_path,
        execution_receipt_path=execution_path,
        review_path=review_path,
    )
    assert blocked["status"] == "BLOCKED"
    assert "reviewer must be independent" in " ".join(blocked["failures"])

    execution_path.write_text(
        json.dumps(
            {
                "schema": vto_ooda.RECEIPT_SCHEMA,
                "contract_sha256": contract_hash,
                "output_sha256": candidate_hash,
                "candidate_only": True,
            }
        ),
        encoding="utf-8",
    )
    review["reviewer"] = "visual-commerce-qa"
    review_path.write_text(json.dumps(review), encoding="utf-8")
    forged = vto_ooda.verify_review(
        contract_path,
        candidate_path=candidate_path,
        execution_receipt_path=execution_path,
        review_path=review_path,
    )
    assert forged["status"] == "BLOCKED"
    assert any("execution receipt" in failure for failure in forged["failures"])


def test_review_cannot_claim_rear_pockets_are_rendered_in_front_candidate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, contract = _contract(tmp_path, include_model=True, include_approval=True)
    candidate_path = tmp_path / "outputs/candidate.png"
    _write(candidate_path, VTO_CANDIDATE_PNG)
    candidate_hash = vto_ooda.sha256_file(candidate_path)
    contract_hash = vto_ooda.sha256_file(contract_path)
    execution_path = tmp_path / "receipts/execution.json"
    execution_path.write_text(
        json.dumps(_complete_execution_receipt(tmp_path, contract_path, contract, candidate_hash)),
        encoding="utf-8",
    )
    pocket_evidence = json.loads(json.dumps(contract["review_gate"]["pocket_evidence"]))
    pocket_evidence["wearer_left_rear"] = {
        "candidate_proof": "DIRECTLY_VISIBLE_ZIPPERED",
        "required_disposition": "PASS",
    }
    review_path = tmp_path / "receipts/review.json"
    review_path.write_text(
        json.dumps(
            {
                "schema": vto_ooda.REVIEW_SCHEMA,
                "pilot_id": contract["pilot_id"],
                "contract_sha256": contract_hash,
                "candidate_sha256": candidate_hash,
                "reviewer": "visual-commerce-qa",
                "verdict": "PASS",
                "findings": [],
                "checks": dict.fromkeys(contract["review_gate"]["required_checks"], "PASS"),
                "pocket_evidence": pocket_evidence,
            }
        ),
        encoding="utf-8",
    )

    result = vto_ooda.verify_review(
        contract_path,
        candidate_path=candidate_path,
        execution_receipt_path=execution_path,
        review_path=review_path,
    )

    assert result["status"] == "BLOCKED"
    assert "NOT_OBSERVABLE_IN_FRONT_CANDIDATE" in " ".join(result["failures"])
