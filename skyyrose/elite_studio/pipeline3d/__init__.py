"""SkyyRose-owned, provider-neutral product-to-3D platform."""

from .models import (
    STAGE_ORDER,
    Artifact,
    JobSpec,
    PipelineResult,
    Stage,
    StageResult,
    StageSpec,
    TaskStatus,
    ordered_stages,
)
from .deterministic_gates import ArtifactGateRunner
from .execution import ExecutionManifest, HandlerKind, HandlerRegistry, PlatformExecutionEngine
from .platform_contracts import (
    AssetIntent,
    EvidenceAsset,
    EvidenceBundle,
    EvidenceRole,
    GenerationMode,
    PlatformRequest,
    PlatformStage,
    ProviderPolicy,
    QualityPolicy,
    WorkflowPlan,
)
from .production import SkyyRose3DPlatform
from .specification import CollaborativeSpecCompiler, GenerationSpecification, SpecReview
from .vision_clients import OpenAICompatibleVisionClient
from .workflow import compile_workflow

__all__ = [
    "Artifact",
    "ArtifactGateRunner",
    "ExecutionManifest",
    "HandlerRegistry",
    "HandlerKind",
    "PlatformExecutionEngine",
    "JobSpec",
    "PipelineResult",
    "Stage",
    "StageResult",
    "StageSpec",
    "TaskStatus",
    "STAGE_ORDER",
    "ordered_stages",
    "AssetIntent",
    "EvidenceAsset",
    "EvidenceBundle",
    "EvidenceRole",
    "GenerationMode",
    "PlatformRequest",
    "PlatformStage",
    "ProviderPolicy",
    "QualityPolicy",
    "SkyyRose3DPlatform",
    "WorkflowPlan",
    "compile_workflow",
    "CollaborativeSpecCompiler",
    "GenerationSpecification",
    "OpenAICompatibleVisionClient",
    "SpecReview",
]
