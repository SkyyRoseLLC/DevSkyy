from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from skyyrose.elite_studio.pipeline3d.execution import (
    ExecutionStatus,
    HandlerKind,
    HandlerRegistry,
    PlatformExecutionEngine,
    StageExecutionResult,
)
from skyyrose.elite_studio.pipeline3d.fidelity import (
    AuthorityKind,
    GateReceipt,
    GateStatus,
)
from skyyrose.elite_studio.pipeline3d.model_registry import (
    Capability,
    DeploymentKind,
    LicenseClass,
    ModelRegistration,
    ModelRegistry,
)
from skyyrose.elite_studio.pipeline3d.platform_contracts import (
    EvidenceAsset,
    EvidenceBundle,
    EvidenceRole,
    GenerationMode,
    PlatformRequest,
    PlatformStage,
    QualityPolicy,
)
from skyyrose.elite_studio.pipeline3d.production import SkyyRose3DPlatform


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _request(tmp_path: Path) -> PlatformRequest:
    front = tmp_path / "front.png"
    back = tmp_path / "back.png"
    dossier = tmp_path / "dossier.json"
    front.write_bytes(b"front")
    back.write_bytes(b"back")
    dossier.write_text("{}")
    return PlatformRequest(
        sku="br-001",
        generation_mode=GenerationMode.MULTIVIEW,
        evidence=EvidenceBundle(
            sku="br-001",
            revision="r1",
            dossier_uri=str(dossier),
            dossier_sha256=_hash(dossier),
            assets=(
                EvidenceAsset(
                    role=EvidenceRole.FRONT,
                    uri=str(front),
                    sha256=_hash(front),
                    founder_approved=True,
                ),
                EvidenceAsset(
                    role=EvidenceRole.BACK,
                    uri=str(back),
                    sha256=_hash(back),
                    founder_approved=True,
                ),
            ),
        ),
        quality=QualityPolicy(
            require_openai_vision=False,
            require_open_source_vision=False,
            require_founder_approval=False,
        ),
    )


def _platform() -> SkyyRose3DPlatform:
    capabilities = (
        Capability.MULTIVIEW_TO_3D,
        Capability.TEXTURE,
        Capability.MESH_PROCESSING,
        Capability.GLB_VALIDATION,
    )
    registry = ModelRegistry(
        [
            ModelRegistration(
                id=f"model-{capability.value}",
                provider="test",
                model="test",
                deployment=DeploymentKind.OPEN_SOURCE_LOCAL,
                license_class=LicenseClass.OPEN_SOURCE,
                capabilities=(capability,),
                available=True,
            )
            for capability in capabilities
        ]
    )
    return SkyyRose3DPlatform(registry)


class PassingHandler:
    def __init__(self, id_: str, kind: HandlerKind) -> None:
        self.id = id_
        self.kind = kind

    def supports(self, stage: PlatformStage) -> bool:
        return True

    async def run(self, *, stage, request, plan, previous):
        gates: tuple[str, ...] = ()
        if stage == PlatformStage.EVIDENCE_GATE:
            gates = ("source_integrity",)
        elif stage == PlatformStage.DETERMINISTIC_QC:
            gates = (
                "glb_structure",
                "mesh_topology",
                "materials_pbr",
                "dimensions",
                "proof_views",
            )
        receipts = tuple(
            GateReceipt(
                gate=gate,
                status=GateStatus.PASS,
                authority=AuthorityKind.DETERMINISTIC,
                producer_id="generation-worker",
                verifier_id=f"deterministic-{gate}",
                evidence=(f"receipts/{gate}.json",),
            )
            for gate in gates
        )
        return StageExecutionResult(
            stage=stage,
            success=True,
            detail={"stage": stage.value},
            receipts=receipts,
        )


def _handlers() -> HandlerRegistry:
    return HandlerRegistry(
        [
            PassingHandler("control", HandlerKind.CONTROL),
            PassingHandler("producer", HandlerKind.PRODUCER),
            PassingHandler("deterministic", HandlerKind.DETERMINISTIC_VERIFIER),
            PassingHandler("openai", HandlerKind.OPENAI_JUDGE),
            PassingHandler("oss", HandlerKind.OSS_JUDGE),
            PassingHandler("founder", HandlerKind.HUMAN_APPROVER),
            PassingHandler("publisher", HandlerKind.PUBLISHER),
        ]
    )


@pytest.mark.asyncio
async def test_execution_persists_a_complete_manifest(tmp_path: Path) -> None:
    engine = PlatformExecutionEngine(
        platform=_platform(),
        handlers=_handlers(),
        manifest_dir=tmp_path / "manifests",
    )

    manifest = await engine.run(_request(tmp_path))

    assert manifest.status == ExecutionStatus.SUCCEEDED
    assert manifest.fidelity is not None and manifest.fidelity.certified
    assert len(manifest.stages) == len(manifest.workflow.stages)
    assert manifest.manifest_path is not None
    assert Path(manifest.manifest_path).is_file()


@pytest.mark.asyncio
async def test_missing_stage_handler_blocks_instead_of_skipping(tmp_path: Path) -> None:
    engine = PlatformExecutionEngine(
        platform=_platform(),
        handlers=HandlerRegistry([]),
        manifest_dir=tmp_path / "manifests",
    )

    manifest = await engine.run(_request(tmp_path))

    assert manifest.status == ExecutionStatus.BLOCKED
    assert "no handler" in (manifest.error or "")


class ReceiptlessHandler(PassingHandler):

    async def run(self, *, stage, request, plan, previous):
        return StageExecutionResult(stage=stage, success=True)


@pytest.mark.asyncio
async def test_successful_handlers_cannot_bypass_certification(tmp_path: Path) -> None:
    engine = PlatformExecutionEngine(
        platform=_platform(),
        handlers=HandlerRegistry(
            [
                ReceiptlessHandler("control", HandlerKind.CONTROL),
                ReceiptlessHandler("producer", HandlerKind.PRODUCER),
                ReceiptlessHandler("deterministic", HandlerKind.DETERMINISTIC_VERIFIER),
            ]
        ),
        manifest_dir=tmp_path / "manifests",
    )

    manifest = await engine.run(_request(tmp_path))

    assert manifest.status == ExecutionStatus.BLOCKED
    assert "certification failed" in (manifest.error or "")


def test_generation_handler_cannot_grade_deterministic_qc() -> None:
    registry = HandlerRegistry([PassingHandler("producer", HandlerKind.PRODUCER)])

    assert registry.pick(PlatformStage.GENERATE) is not None
    assert registry.pick(PlatformStage.DETERMINISTIC_QC) is None
