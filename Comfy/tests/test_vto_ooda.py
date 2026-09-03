from __future__ import annotations

import importlib.util
import json
from dataclasses import dataclass
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "vto_ooda.py"
SPEC = importlib.util.spec_from_file_location("vto_ooda", SCRIPT)
assert SPEC and SPEC.loader
vto_ooda = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(vto_ooda)


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
            **_binding(root, "inputs/model.png", b"model"),
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
    assert result["estimated_credits"] == 4
    assert result["blockers"] == []


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


def test_execute_requires_explicit_spend_flag(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="allow-spend"):
        vto_ooda.asyncio.run(vto_ooda.execute(tmp_path / "contract.json", allow_spend=False))


@pytest.mark.asyncio
async def test_execute_writes_candidate_and_hash_bound_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, _ = _contract(tmp_path, include_model=True, include_approval=True)
    png = b"\x89PNG\r\n\x1a\nunit-test-png"

    @dataclass
    class FakeResult:
        job_id: str = "job-123"
        output_urls: tuple[str, ...] = ("data:image/png;base64,iVBORw0KGgp1bml0LXRlc3QtcG5n",)
        credits_used: int = 4

    class FakeClient:
        async def __aenter__(self) -> FakeClient:
            return self

        async def __aexit__(self, *_: object) -> None:
            return None

        async def run_tryon_max(self, **_: object) -> FakeResult:
            return FakeResult()

    class FakeClientFactory:
        @classmethod
        def from_env(cls) -> FakeClient:
            return FakeClient()

    monkeypatch.setattr(vto_ooda, "FashnClient", FakeClientFactory)

    receipt = await vto_ooda.execute(contract_path, allow_spend=True)

    candidate = tmp_path / "outputs/candidate.png"
    assert candidate.read_bytes() == png
    assert receipt["job_id"] == "job-123"
    assert receipt["credits_used"] == 4
    assert receipt["output_sha256"] == vto_ooda.sha256_file(candidate)
    assert receipt["candidate_only"] is True
    assert receipt["promotion_authorized"] is False
    saved = json.loads((tmp_path / "receipts/execution.json").read_text(encoding="utf-8"))
    assert saved == receipt


def test_review_requires_independent_exact_check_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(vto_ooda, "PROJECT_ROOT", tmp_path)
    contract_path, contract = _contract(tmp_path, include_model=True, include_approval=True)
    candidate_path = tmp_path / "outputs/candidate.png"
    _write(candidate_path, b"\x89PNG\r\n\x1a\nreview")
    candidate_hash = vto_ooda.sha256_file(candidate_path)
    contract_hash = vto_ooda.sha256_file(contract_path)

    execution_path = tmp_path / "receipts/execution.json"
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
