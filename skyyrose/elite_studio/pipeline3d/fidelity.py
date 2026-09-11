"""Independent, receipt-based fidelity certification.

The certifier never invokes a generator or a judge.  It consumes immutable
receipts emitted by independent authorities and evaluates the release policy.
That separation prevents the artifact-producing harness from grading itself.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from .platform_contracts import QualityPolicy


class GateStatus(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    BLOCKED = "blocked"
    SKIP = "skip"


class AuthorityKind(StrEnum):
    DETERMINISTIC = "deterministic"
    OPENAI_VISION = "openai_vision"
    OSS_VISION = "oss_vision"
    HUMAN = "human"


class GateReceipt(BaseModel):
    """Evidence emitted by a named independent verifier."""

    model_config = ConfigDict(frozen=True)

    gate: str = Field(min_length=1)
    status: GateStatus
    authority: AuthorityKind
    producer_id: str = Field(min_length=1)
    verifier_id: str = Field(min_length=1)
    evidence: tuple[str, ...] = Field(min_length=1)
    metrics: dict[str, float | int | str | bool] = Field(default_factory=dict)
    artifact_sha256: str | None = Field(default=None, pattern=r"^[a-fA-F0-9]{64}$")
    source_revision: str | None = None


class FidelityReport(BaseModel):
    """Final certification decision with every blocking reason preserved."""

    model_config = ConfigDict(frozen=True)

    certified: bool
    blockers: tuple[str, ...]
    passed_gates: tuple[str, ...]
    receipts: tuple[GateReceipt, ...]


class FidelityCertifier:
    """Evaluate deterministic, dual-vision and founder approval receipts."""

    _BASE_REQUIRED: tuple[tuple[str, AuthorityKind], ...] = (
        ("source_integrity", AuthorityKind.DETERMINISTIC),
        ("glb_structure", AuthorityKind.DETERMINISTIC),
        ("mesh_topology", AuthorityKind.DETERMINISTIC),
        ("materials_pbr", AuthorityKind.DETERMINISTIC),
        ("dimensions", AuthorityKind.DETERMINISTIC),
        ("proof_views", AuthorityKind.DETERMINISTIC),
    )

    def __init__(self, policy: QualityPolicy) -> None:
        self.policy = policy

    def certify(
        self,
        receipts: list[GateReceipt] | tuple[GateReceipt, ...],
        *,
        require_rig: bool = False,
        require_animation: bool = False,
    ) -> FidelityReport:
        by_gate: dict[str, GateReceipt] = {}
        blockers: list[str] = []

        for receipt in receipts:
            if receipt.gate in by_gate:
                blockers.append(f"duplicate gate receipt: {receipt.gate}")
                continue
            by_gate[receipt.gate] = receipt
            if receipt.producer_id == receipt.verifier_id:
                blockers.append(
                    f"self-grading receipt rejected: {receipt.gate} by {receipt.verifier_id}"
                )

        required = list(self._BASE_REQUIRED)
        if self.policy.require_openai_vision:
            required.append(("openai_vision", AuthorityKind.OPENAI_VISION))
        if self.policy.require_open_source_vision:
            required.append(("oss_vision", AuthorityKind.OSS_VISION))
        if require_rig:
            required.append(("rig", AuthorityKind.DETERMINISTIC))
        if require_animation:
            required.append(("animation", AuthorityKind.DETERMINISTIC))
        if self.policy.require_founder_approval:
            required.append(("founder_approval", AuthorityKind.HUMAN))

        passed: list[str] = []
        for gate, authority in required:
            receipt = by_gate.get(gate)
            if receipt is None:
                blockers.append(f"missing required gate: {gate}")
                continue
            if receipt.authority != authority:
                blockers.append(
                    f"gate {gate} used {receipt.authority.value}, expected {authority.value}"
                )
            if receipt.status != GateStatus.PASS:
                blockers.append(f"gate {gate} did not pass: {receipt.status.value}")
            if (
                receipt.authority in {AuthorityKind.OPENAI_VISION, AuthorityKind.OSS_VISION}
                and isinstance(receipt.metrics.get("score"), (float, int))
                and float(receipt.metrics["score"]) < self.policy.minimum_fidelity_score
            ):
                blockers.append(
                    f"gate {gate} score {receipt.metrics['score']} below "
                    f"{self.policy.minimum_fidelity_score}"
                )
            if receipt.status == GateStatus.PASS and receipt.authority == authority:
                passed.append(gate)

        openai = by_gate.get("openai_vision")
        oss = by_gate.get("oss_vision")
        if openai and oss and openai.verifier_id == oss.verifier_id:
            blockers.append("dual-vision gates must use independent verifier identities")

        artifact_hashes = {
            receipt.artifact_sha256 for receipt in by_gate.values() if receipt.artifact_sha256
        }
        if len(artifact_hashes) > 1:
            blockers.append("gate receipts refer to different artifact hashes")

        revisions = {
            receipt.source_revision for receipt in by_gate.values() if receipt.source_revision
        }
        if len(revisions) > 1:
            blockers.append("gate receipts refer to different source revisions")

        return FidelityReport(
            certified=not blockers,
            blockers=tuple(blockers),
            passed_gates=tuple(passed),
            receipts=tuple(receipts),
        )


__all__ = [
    "AuthorityKind",
    "FidelityCertifier",
    "FidelityReport",
    "GateReceipt",
    "GateStatus",
]
