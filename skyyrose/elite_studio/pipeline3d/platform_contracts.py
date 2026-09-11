"""Canonical contracts for the SkyyRose product-to-3D platform.

The original :mod:`pipeline3d.models` module remains the low-level provider
chaining contract.  This module owns the product-evidence, workflow and
certification contract that sits above every generator.  It is deliberately
provider-neutral: TRELLIS, Tripo and future in-house engines must all consume
the same immutable evidence bundle and produce the same gate receipts.
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator


class AssetIntent(StrEnum):
    """Whether identity must match an existing SKU or can be exploratory."""

    REPLICA = "replica"
    CONCEPT = "concept"


class GenerationMode(StrEnum):
    """Supported generation entry points."""

    IMAGE = "image"
    MULTIVIEW = "multiview"
    TEXT = "text"


class EvidenceRole(StrEnum):
    """Semantic role of a source asset in a product evidence bundle."""

    FRONT = "front"
    BACK = "back"
    LEFT = "left"
    RIGHT = "right"
    DETAIL = "detail"
    TECH_FLAT = "tech_flat"
    MATERIAL = "material"
    LOGO = "logo"
    PATCH = "patch"


class PlatformStage(StrEnum):
    """Complete production DAG, including evidence and release gates."""

    INGEST = "ingest"
    EVIDENCE_GATE = "evidence_gate"
    SPEC = "spec"
    PREPROCESS = "preprocess"
    GENERATE = "generate"
    TEXTURE = "texture"
    SEGMENT = "segment"
    COMPLETE = "complete"
    REMESH = "remesh"
    UV = "uv"
    RIGGABILITY = "riggability"
    RIG = "rig"
    RETARGET = "retarget"
    EXPORT = "export"
    RENDER_PROOFS = "render_proofs"
    DETERMINISTIC_QC = "deterministic_qc"
    OPENAI_VISION_QC = "openai_vision_qc"
    OSS_VISION_QC = "oss_vision_qc"
    CONSENSUS = "consensus"
    FOUNDER_APPROVAL = "founder_approval"
    PUBLISH = "publish"


class EvidenceAsset(BaseModel):
    """Hash-bound, founder-reviewed source material.

    ``sha256`` is mandatory even for remote assets.  Remote assets must be
    downloaded and hashed by ingestion before they can enter this contract.
    This prevents a URL from silently serving different pixels on a later run.
    """

    model_config = ConfigDict(frozen=True)

    role: EvidenceRole
    uri: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    founder_approved: bool = False
    label: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def local_path(self) -> Path | None:
        if self.uri.startswith(("http://", "https://", "s3://", "r2://")):
            return None
        return Path(self.uri).expanduser()


class EvidenceBundle(BaseModel):
    """Versioned source of truth for one SKU generation run."""

    model_config = ConfigDict(frozen=True)

    sku: str = Field(min_length=1)
    revision: str = Field(min_length=1)
    dossier_uri: str = Field(min_length=1)
    dossier_sha256: str = Field(pattern=r"^[a-fA-F0-9]{64}$")
    assets: tuple[EvidenceAsset, ...] = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def roles(self) -> set[EvidenceRole]:
        return {asset.role for asset in self.assets}

    def for_role(self, role: EvidenceRole) -> tuple[EvidenceAsset, ...]:
        return tuple(asset for asset in self.assets if asset.role == role)


class QualityPolicy(BaseModel):
    """Fail-closed thresholds and authorities required for certification."""

    model_config = ConfigDict(frozen=True)

    minimum_fidelity_score: float = Field(default=0.98, ge=0.0, le=1.0)
    max_triangles: int = Field(default=150_000, gt=0)
    max_glb_bytes: int = Field(default=25_000_000, gt=0)
    min_texture_size: int = Field(default=1024, ge=256)
    required_proof_views: tuple[str, ...] = (
        "front",
        "back",
        "left",
        "right",
        "front_three_quarter",
        "back_three_quarter",
        "material_macro",
        "branding_macro",
    )
    require_openai_vision: bool = True
    require_open_source_vision: bool = True
    require_founder_approval: bool = True


class ProviderPolicy(BaseModel):
    """Routing boundary for owned, OpenAI and optional external providers."""

    model_config = ConfigDict(frozen=True)

    prefer_open_source_generation: bool = True
    allow_openai: bool = True
    allow_external_3d_apis: bool = False
    require_reproducible_seed: bool = True


class PlatformRequest(BaseModel):
    """Canonical request accepted by the SkyyRose 3D platform."""

    model_config = ConfigDict(frozen=True)

    sku: str = Field(min_length=1)
    intent: AssetIntent = AssetIntent.REPLICA
    generation_mode: GenerationMode = GenerationMode.MULTIVIEW
    evidence: EvidenceBundle | None = None
    prompt: str | None = None
    require_rig: bool = False
    animation_clip: str | None = None
    publish: bool = False
    output_formats: tuple[str, ...] = ("glb",)
    quality: QualityPolicy = Field(default_factory=QualityPolicy)
    providers: ProviderPolicy = Field(default_factory=ProviderPolicy)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_identity_contract(self) -> PlatformRequest:
        if self.evidence is not None and self.evidence.sku != self.sku:
            raise ValueError(
                f"request sku {self.sku!r} does not match evidence sku {self.evidence.sku!r}"
            )

        if self.generation_mode == GenerationMode.TEXT and not self.prompt:
            raise ValueError("text generation requires prompt")

        if self.intent == AssetIntent.REPLICA:
            if self.generation_mode == GenerationMode.TEXT:
                raise ValueError("replica jobs cannot use text-only generation")
            if self.evidence is None:
                raise ValueError("replica jobs require a hash-bound evidence bundle")
            required = {EvidenceRole.FRONT, EvidenceRole.BACK}
            if not required.issubset(self.evidence.roles):
                raise ValueError("replica evidence must contain front and back views")
            unapproved = [asset.uri for asset in self.evidence.assets if not asset.founder_approved]
            if unapproved:
                raise ValueError(
                    "replica evidence contains assets without founder approval: "
                    + ", ".join(unapproved)
                )

        if self.animation_clip and not self.require_rig:
            raise ValueError("animation retargeting requires require_rig=true")
        if not self.output_formats:
            raise ValueError("at least one output format is required")
        return self


class PlannedStage(BaseModel):
    """One compiled stage and its explicit dependencies."""

    model_config = ConfigDict(frozen=True)

    stage: PlatformStage
    requires: tuple[PlatformStage, ...] = ()
    blocking: bool = True
    description: str = ""


class WorkflowPlan(BaseModel):
    """Immutable, inspectable execution plan."""

    model_config = ConfigDict(frozen=True)

    version: str = "skyyrose-3d/v1"
    sku: str
    intent: AssetIntent
    generation_mode: GenerationMode
    stages: tuple[PlannedStage, ...]
    provider_policy: ProviderPolicy
    quality_policy: QualityPolicy
    evidence_revision: str | None = None


__all__ = [
    "AssetIntent",
    "EvidenceAsset",
    "EvidenceBundle",
    "EvidenceRole",
    "GenerationMode",
    "PlatformRequest",
    "PlatformStage",
    "PlannedStage",
    "ProviderPolicy",
    "QualityPolicy",
    "WorkflowPlan",
]
