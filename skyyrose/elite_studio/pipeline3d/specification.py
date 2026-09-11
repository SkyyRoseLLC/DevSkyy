"""Evidence-bound generation specification compiled by two vision models.

OpenAI drafts the specification from the same approved images and dossier that
will enter generation.  An independently hosted open-source vision model then
audits every identity-bearing claim.  One bounded repair is allowed; a second
rejection blocks generation.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .platform_contracts import PlatformRequest


@runtime_checkable
class JsonVisionClient(Protocol):
    """Minimal port implemented by OpenAI and OpenAI-compatible VLM clients."""

    id: str

    async def generate_json(self, payload: dict[str, Any], images: tuple[Path, ...]) -> dict:
        """Return one parsed JSON object for the payload and source images."""
        ...


class GenerationSpecification(BaseModel):
    model_config = ConfigDict(frozen=True)

    sku: str
    evidence_revision: str
    silhouette: tuple[str, ...] = Field(min_length=1)
    construction: tuple[str, ...] = Field(min_length=1)
    materials: tuple[str, ...] = Field(min_length=1)
    front_identity: tuple[str, ...] = Field(min_length=1)
    back_identity: tuple[str, ...] = Field(min_length=1)
    side_identity: tuple[str, ...] = Field(min_length=1)
    branding: tuple[str, ...] = Field(min_length=1)
    colors: tuple[str, ...] = Field(min_length=1)
    must_not_change: tuple[str, ...] = Field(min_length=1)
    generation_prompt: str = Field(min_length=1)
    negative_prompt: str = Field(min_length=1)


class SpecReview(BaseModel):
    model_config = ConfigDict(frozen=True)

    approved: bool
    fidelity_risk: float = Field(ge=0.0, le=1.0)
    missing_evidence: tuple[str, ...] = ()
    conflicts: tuple[str, ...] = ()
    required_corrections: tuple[str, ...] = ()
    reason: str = Field(min_length=1)


class SpecificationResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    specification: GenerationSpecification
    review: SpecReview
    draft_model: str
    review_model: str
    rounds: int
    evidence_hashes: tuple[str, ...]


class SpecConsensusError(RuntimeError):
    """The independent reviewer would not approve a bounded repair."""


class CollaborativeSpecCompiler:
    """OpenAI draft + open-source visual audit with one repair round."""

    def __init__(self, *, openai: JsonVisionClient, open_source: JsonVisionClient) -> None:
        if openai.id == open_source.id:
            raise ValueError("draft and review clients must have independent identities")
        self.openai = openai
        self.open_source = open_source

    async def compile(self, request: PlatformRequest) -> SpecificationResult:
        if request.evidence is None:
            raise SpecConsensusError("specification requires an evidence bundle")

        images = self._image_paths(request)
        dossier = self._read_dossier(request.evidence.dossier_uri)
        base_payload = {
            "contract": "skyyrose-3d-spec/v1",
            "task": (
                "Describe only product attributes proven by the dossier and attached approved "
                "views. Never invent a logo, patch, material, seam, color, back design or trim."
            ),
            "sku": request.sku,
            "intent": request.intent.value,
            "generation_mode": request.generation_mode.value,
            "evidence_revision": request.evidence.revision,
            "dossier": dossier,
            "evidence": [
                {
                    "role": asset.role.value,
                    "sha256": asset.sha256,
                    "label": asset.label,
                    "metadata": asset.metadata,
                }
                for asset in request.evidence.assets
            ],
            "output_schema": GenerationSpecification.model_json_schema(),
        }

        specification = await self._draft(base_payload, images)
        review = await self._review(request, specification, images)
        rounds = 1

        if not review.approved:
            repair_payload = {
                **base_payload,
                "task": "Repair the specification using every required correction. Do not invent.",
                "rejected_specification": specification.model_dump(mode="json"),
                "independent_review": review.model_dump(mode="json"),
            }
            specification = await self._draft(repair_payload, images)
            review = await self._review(request, specification, images)
            rounds = 2

        if not review.approved:
            raise SpecConsensusError(
                "open-source review rejected the specification after one repair: " + review.reason
            )

        return SpecificationResult(
            specification=specification,
            review=review,
            draft_model=self.openai.id,
            review_model=self.open_source.id,
            rounds=rounds,
            evidence_hashes=tuple(asset.sha256 for asset in request.evidence.assets),
        )

    async def _draft(
        self,
        payload: dict[str, Any],
        images: tuple[Path, ...],
    ) -> GenerationSpecification:
        try:
            raw = await self.openai.generate_json(payload, images)
            return GenerationSpecification.model_validate(raw)
        except (ValidationError, TypeError, ValueError) as exc:
            raise SpecConsensusError(f"OpenAI specification was invalid: {exc}") from exc

    async def _review(
        self,
        request: PlatformRequest,
        specification: GenerationSpecification,
        images: tuple[Path, ...],
    ) -> SpecReview:
        payload = {
            "contract": "skyyrose-3d-spec-review/v1",
            "task": (
                "Independently compare every specification claim with the attached approved "
                "product views. Reject unsupported, missing, stale, contradictory or invented "
                "identity-bearing details."
            ),
            "sku": request.sku,
            "specification": specification.model_dump(mode="json"),
            "output_schema": SpecReview.model_json_schema(),
        }
        try:
            raw = await self.open_source.generate_json(payload, images)
            return SpecReview.model_validate(raw)
        except (ValidationError, TypeError, ValueError) as exc:
            raise SpecConsensusError(f"open-source review was invalid: {exc}") from exc

    @staticmethod
    def _image_paths(request: PlatformRequest) -> tuple[Path, ...]:
        assert request.evidence is not None  # guarded by compile
        paths: list[Path] = []
        for asset in request.evidence.assets:
            path = asset.local_path
            if path is None or not path.is_file():
                raise SpecConsensusError(
                    f"evidence must be materialized before visual specification: {asset.uri}"
                )
            paths.append(path)
        return tuple(paths)

    @staticmethod
    def _read_dossier(uri: str) -> dict[str, Any] | str:
        path = Path(uri).expanduser()
        if not path.is_file():
            raise SpecConsensusError(f"dossier not found: {path}")
        if path.stat().st_size > 256_000:
            raise SpecConsensusError(f"dossier exceeds 256 KB input limit: {path}")
        text = path.read_text(encoding="utf-8")
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return text
        return value if isinstance(value, dict) else text


__all__ = [
    "CollaborativeSpecCompiler",
    "GenerationSpecification",
    "JsonVisionClient",
    "SpecConsensusError",
    "SpecReview",
    "SpecificationResult",
]
