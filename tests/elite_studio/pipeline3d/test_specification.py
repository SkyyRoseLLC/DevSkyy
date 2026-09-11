from __future__ import annotations

from pathlib import Path

import pytest

from skyyrose.elite_studio.pipeline3d.platform_contracts import (
    EvidenceAsset,
    EvidenceBundle,
    EvidenceRole,
    GenerationMode,
    PlatformRequest,
)
from skyyrose.elite_studio.pipeline3d.specification import (
    CollaborativeSpecCompiler,
    SpecConsensusError,
)


class FakeVisionClient:
    def __init__(self, id_: str, responses: list[dict]) -> None:
        self.id = id_
        self.responses = list(responses)
        self.calls: list[tuple[dict, tuple[Path, ...]]] = []

    async def generate_json(self, payload: dict, images: tuple[Path, ...]) -> dict:
        self.calls.append((payload, images))
        return self.responses.pop(0)


def _request(tmp_path: Path) -> PlatformRequest:
    dossier = tmp_path / "dossier.json"
    front = tmp_path / "front.png"
    back = tmp_path / "back.png"
    dossier.write_text('{"material":"satin","front":"rose patch"}')
    front.write_bytes(b"front")
    back.write_bytes(b"back")
    evidence = EvidenceBundle(
        sku="lh-001",
        revision="approved-r1",
        dossier_uri=str(dossier),
        dossier_sha256="a" * 64,
        assets=(
            EvidenceAsset(
                role=EvidenceRole.FRONT,
                uri=str(front),
                sha256="b" * 64,
                founder_approved=True,
            ),
            EvidenceAsset(
                role=EvidenceRole.BACK,
                uri=str(back),
                sha256="c" * 64,
                founder_approved=True,
            ),
        ),
    )
    return PlatformRequest(
        sku="lh-001",
        generation_mode=GenerationMode.MULTIVIEW,
        evidence=evidence,
    )


def _spec() -> dict:
    return {
        "sku": "lh-001",
        "evidence_revision": "approved-r1",
        "silhouette": ["waist-length bomber"],
        "construction": ["raglan sleeves"],
        "materials": ["satin shell"],
        "front_identity": ["rose patch at wearer-left chest"],
        "back_identity": ["preserve approved back artwork"],
        "side_identity": ["contrast side stripe"],
        "branding": ["use only source-bound logo art"],
        "colors": ["black", "white", "red"],
        "must_not_change": ["logo geometry", "material class"],
        "generation_prompt": "Reconstruct the approved product exactly.",
        "negative_prompt": "No invented logos or construction.",
    }


@pytest.mark.asyncio
async def test_openai_draft_requires_independent_oss_approval(tmp_path: Path) -> None:
    openai = FakeVisionClient("openai", [_spec()])
    oss = FakeVisionClient(
        "oss",
        [
            {
                "approved": True,
                "fidelity_risk": 0.02,
                "missing_evidence": [],
                "conflicts": [],
                "required_corrections": [],
                "reason": "all identity-bearing details are source-bound",
            }
        ],
    )

    result = await CollaborativeSpecCompiler(openai=openai, open_source=oss).compile(
        _request(tmp_path)
    )

    assert result.review.approved is True
    assert result.draft_model == "openai"
    assert result.review_model == "oss"
    assert len(openai.calls[0][1]) == 2
    assert len(oss.calls[0][1]) == 2


@pytest.mark.asyncio
async def test_rejected_review_gets_one_repair_then_fails_closed(tmp_path: Path) -> None:
    rejected = {
        "approved": False,
        "fidelity_risk": 0.5,
        "missing_evidence": ["back logo dimensions"],
        "conflicts": [],
        "required_corrections": ["bind back logo to approved evidence"],
        "reason": "back identity is underspecified",
    }
    openai = FakeVisionClient("openai", [_spec(), _spec()])
    oss = FakeVisionClient("oss", [rejected, rejected])

    with pytest.raises(SpecConsensusError, match="open-source review rejected"):
        await CollaborativeSpecCompiler(openai=openai, open_source=oss).compile(
            _request(tmp_path)
        )

    assert len(openai.calls) == 2
    assert len(oss.calls) == 2
