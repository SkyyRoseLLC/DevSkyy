from __future__ import annotations

from skyyrose.elite_studio.pipeline3d.fidelity import (
    AuthorityKind,
    FidelityCertifier,
    GateReceipt,
    GateStatus,
)
from skyyrose.elite_studio.pipeline3d.platform_contracts import QualityPolicy


def _receipt(gate: str, authority: AuthorityKind, verifier: str) -> GateReceipt:
    return GateReceipt(
        gate=gate,
        status=GateStatus.PASS,
        authority=authority,
        producer_id="trellis-worker-1",
        verifier_id=verifier,
        evidence=(f"receipts/{gate}.json",),
        metrics={"score": 1.0},
    )


def _passing_receipts() -> list[GateReceipt]:
    deterministic = [
        "source_integrity",
        "glb_structure",
        "mesh_topology",
        "materials_pbr",
        "dimensions",
        "proof_views",
    ]
    receipts = [
        _receipt(gate, AuthorityKind.DETERMINISTIC, f"det-{gate}")
        for gate in deterministic
    ]
    receipts.extend(
        [
            _receipt("openai_vision", AuthorityKind.OPENAI_VISION, "openai-judge"),
            _receipt("oss_vision", AuthorityKind.OSS_VISION, "oss-judge"),
            _receipt("founder_approval", AuthorityKind.HUMAN, "founder"),
        ]
    )
    return receipts


def test_certifier_requires_independent_dual_vision_and_founder() -> None:
    report = FidelityCertifier(QualityPolicy()).certify(_passing_receipts())

    assert report.certified is True
    assert report.blockers == ()


def test_certifier_fails_closed_when_oss_vision_is_missing() -> None:
    receipts = [r for r in _passing_receipts() if r.gate != "oss_vision"]
    report = FidelityCertifier(QualityPolicy()).certify(receipts)

    assert report.certified is False
    assert "missing required gate: oss_vision" in report.blockers


def test_certifier_rejects_self_grading_receipt() -> None:
    receipts = _passing_receipts()
    receipts[0] = receipts[0].model_copy(update={"verifier_id": "trellis-worker-1"})
    report = FidelityCertifier(QualityPolicy()).certify(receipts)

    assert report.certified is False
    assert any("self-grading" in blocker for blocker in report.blockers)


def test_skip_is_not_a_pass() -> None:
    receipts = _passing_receipts()
    receipts[0] = receipts[0].model_copy(update={"status": GateStatus.SKIP})
    report = FidelityCertifier(QualityPolicy()).certify(receipts)

    assert report.certified is False
    assert any("did not pass" in blocker for blocker in report.blockers)
