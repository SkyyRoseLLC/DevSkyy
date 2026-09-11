from __future__ import annotations

import hashlib
from pathlib import Path

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
    bundle = EvidenceBundle(
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
    )
    return PlatformRequest(
        sku="br-001",
        generation_mode=GenerationMode.MULTIVIEW,
        evidence=bundle,
    )


def _registration(
    id_: str,
    capability: Capability,
    deployment: DeploymentKind,
    license_class: LicenseClass,
) -> ModelRegistration:
    return ModelRegistration(
        id=id_,
        provider=id_,
        model=id_,
        deployment=deployment,
        license_class=license_class,
        capabilities=(capability,),
        available=True,
    )


def _ready_registry() -> ModelRegistry:
    return ModelRegistry(
        [
            _registration(
                "trellis",
                Capability.MULTIVIEW_TO_3D,
                DeploymentKind.OPEN_SOURCE_LOCAL,
                LicenseClass.OPEN_SOURCE,
            ),
            _registration(
                "texture",
                Capability.TEXTURE,
                DeploymentKind.OPEN_SOURCE_LOCAL,
                LicenseClass.OPEN_SOURCE,
            ),
            _registration(
                "blender",
                Capability.MESH_PROCESSING,
                DeploymentKind.INTERNAL_TOOL,
                LicenseClass.OPEN_SOURCE,
            ),
            _registration(
                "validator",
                Capability.GLB_VALIDATION,
                DeploymentKind.INTERNAL_TOOL,
                LicenseClass.OPEN_SOURCE,
            ),
            _registration(
                "openai",
                Capability.VISION_FIDELITY,
                DeploymentKind.OPENAI_API,
                LicenseClass.PROPRIETARY_API,
            ),
            _registration(
                "oss-vision",
                Capability.VISION_FIDELITY,
                DeploymentKind.OPEN_SOURCE_LOCAL,
                LicenseClass.OPEN_SOURCE,
            ),
        ]
    )


def test_preflight_passes_only_with_hashes_and_independent_models(tmp_path: Path) -> None:
    report = SkyyRose3DPlatform(_ready_registry()).preflight(_request(tmp_path))

    assert report.ready is True
    assert report.selected_models["openai_vision_fidelity"] == "openai"
    assert report.selected_models["oss_vision_fidelity"] == "oss-vision"


def test_preflight_blocks_stale_evidence(tmp_path: Path) -> None:
    request = _request(tmp_path)
    Path(request.evidence.assets[0].uri).write_bytes(b"changed")  # type: ignore[union-attr]

    report = SkyyRose3DPlatform(_ready_registry()).preflight(request)

    assert report.ready is False
    assert any(issue.code == "evidence.hash.mismatch" for issue in report.issues)
