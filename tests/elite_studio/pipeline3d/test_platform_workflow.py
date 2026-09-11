from __future__ import annotations

from pathlib import Path

from skyyrose.elite_studio.pipeline3d.platform_contracts import (
    EvidenceAsset,
    EvidenceBundle,
    EvidenceRole,
    GenerationMode,
    PlatformRequest,
    PlatformStage,
    QualityPolicy,
)
from skyyrose.elite_studio.pipeline3d.workflow import compile_workflow


def _request(tmp_path: Path, **updates) -> PlatformRequest:
    bundle = EvidenceBundle(
        sku="br-001",
        revision="r1",
        dossier_uri=str(tmp_path / "dossier.json"),
        dossier_sha256="a" * 64,
        assets=(
            EvidenceAsset(
                role=EvidenceRole.FRONT,
                uri=str(tmp_path / "front.png"),
                sha256="b" * 64,
                founder_approved=True,
            ),
            EvidenceAsset(
                role=EvidenceRole.BACK,
                uri=str(tmp_path / "back.png"),
                sha256="c" * 64,
                founder_approved=True,
            ),
        ),
    )
    values = {
        "sku": "br-001",
        "generation_mode": GenerationMode.MULTIVIEW,
        "evidence": bundle,
    }
    values.update(updates)
    return PlatformRequest(**values)


def test_production_workflow_places_rig_after_topology_changes(tmp_path: Path) -> None:
    plan = compile_workflow(_request(tmp_path, require_rig=True))
    stages = [item.stage for item in plan.stages]

    assert stages.index(PlatformStage.REMESH) < stages.index(PlatformStage.RIG)
    assert stages.index(PlatformStage.UV) < stages.index(PlatformStage.RIG)
    assert stages.index(PlatformStage.RIG) < stages.index(PlatformStage.EXPORT)


def test_publish_is_after_all_certification_gates(tmp_path: Path) -> None:
    plan = compile_workflow(_request(tmp_path, publish=True))
    stages = [item.stage for item in plan.stages]

    assert stages[-1] == PlatformStage.PUBLISH
    assert stages.index(PlatformStage.DETERMINISTIC_QC) < stages.index(
        PlatformStage.FOUNDER_APPROVAL
    )
    assert stages.index(PlatformStage.OPENAI_VISION_QC) < stages.index(
        PlatformStage.FOUNDER_APPROVAL
    )
    assert stages.index(PlatformStage.OSS_VISION_QC) < stages.index(
        PlatformStage.FOUNDER_APPROVAL
    )


def test_workflow_has_no_optional_quality_bypass(tmp_path: Path) -> None:
    plan = compile_workflow(_request(tmp_path))
    stages = {item.stage for item in plan.stages}

    assert PlatformStage.EVIDENCE_GATE in stages
    assert PlatformStage.DETERMINISTIC_QC in stages
    assert PlatformStage.OPENAI_VISION_QC in stages
    assert PlatformStage.OSS_VISION_QC in stages
    assert PlatformStage.FOUNDER_APPROVAL in stages


def test_explicit_noncommercial_policy_can_omit_model_and_founder_gates(
    tmp_path: Path,
) -> None:
    plan = compile_workflow(
        _request(
            tmp_path,
            quality=QualityPolicy(
                require_openai_vision=False,
                require_open_source_vision=False,
                require_founder_approval=False,
            ),
        )
    )
    stages = {item.stage for item in plan.stages}

    assert PlatformStage.OPENAI_VISION_QC not in stages
    assert PlatformStage.OSS_VISION_QC not in stages
    assert PlatformStage.FOUNDER_APPROVAL not in stages
