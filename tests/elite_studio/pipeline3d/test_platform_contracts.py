from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from skyyrose.elite_studio.pipeline3d.platform_contracts import (
    AssetIntent,
    EvidenceAsset,
    EvidenceBundle,
    EvidenceRole,
    GenerationMode,
    PlatformRequest,
)


SHA = "a" * 64


def _bundle(tmp_path: Path) -> EvidenceBundle:
    front = tmp_path / "front.png"
    back = tmp_path / "back.png"
    dossier = tmp_path / "dossier.json"
    front.write_bytes(b"front")
    back.write_bytes(b"back")
    dossier.write_text("{}")
    return EvidenceBundle(
        sku="br-001",
        revision="founder-approved-v1",
        dossier_uri=str(dossier),
        dossier_sha256=SHA,
        assets=(
            EvidenceAsset(
                role=EvidenceRole.FRONT,
                uri=str(front),
                sha256=SHA,
                founder_approved=True,
            ),
            EvidenceAsset(
                role=EvidenceRole.BACK,
                uri=str(back),
                sha256=SHA,
                founder_approved=True,
            ),
        ),
    )


def test_replica_requires_front_and_back_evidence(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    request = PlatformRequest(
        sku="br-001",
        intent=AssetIntent.REPLICA,
        generation_mode=GenerationMode.MULTIVIEW,
        evidence=bundle,
    )

    assert request.evidence.roles == {EvidenceRole.FRONT, EvidenceRole.BACK}


def test_replica_rejects_text_only_generation(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="replica jobs cannot use text-only"):
        PlatformRequest(
            sku="br-001",
            intent=AssetIntent.REPLICA,
            generation_mode=GenerationMode.TEXT,
            prompt="a garment",
            evidence=_bundle(tmp_path),
        )


def test_replica_rejects_missing_back_view(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    missing_back = bundle.model_copy(update={"assets": bundle.assets[:1]})
    with pytest.raises(ValidationError, match="front and back"):
        PlatformRequest(
            sku="br-001",
            intent=AssetIntent.REPLICA,
            generation_mode=GenerationMode.IMAGE,
            evidence=missing_back,
        )


def test_sku_must_match_evidence_bundle(tmp_path: Path) -> None:
    with pytest.raises(ValidationError, match="does not match evidence"):
        PlatformRequest(
            sku="lh-001",
            generation_mode=GenerationMode.MULTIVIEW,
            evidence=_bundle(tmp_path),
        )
