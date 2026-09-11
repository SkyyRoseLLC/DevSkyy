"""Compile a product request into the canonical production DAG."""

from __future__ import annotations

from .platform_contracts import (
    PlannedStage,
    PlatformRequest,
    PlatformStage,
    WorkflowPlan,
)


def _stage(
    stage: PlatformStage,
    *requires: PlatformStage,
    description: str,
) -> PlannedStage:
    return PlannedStage(stage=stage, requires=requires, description=description)


def compile_workflow(request: PlatformRequest) -> WorkflowPlan:
    """Create a fail-closed workflow for ``request``.

    Topology-changing operations are deliberately completed before rigging.
    Publishing is never inserted unless all deterministic, dual-vision and
    founder gates precede it.
    """

    stages: list[PlannedStage] = [
        _stage(PlatformStage.INGEST, description="Resolve immutable inputs"),
        _stage(
            PlatformStage.EVIDENCE_GATE,
            PlatformStage.INGEST,
            description="Verify evidence hashes, roles, revision and approval",
        ),
        _stage(
            PlatformStage.SPEC,
            PlatformStage.EVIDENCE_GATE,
            description="Compile dossier and visual evidence into a generation specification",
        ),
        _stage(
            PlatformStage.PREPROCESS,
            PlatformStage.SPEC,
            description="Normalize approved views without changing product identity",
        ),
        _stage(
            PlatformStage.GENERATE,
            PlatformStage.PREPROCESS,
            description=f"Run {request.generation_mode.value}-to-3D generation",
        ),
        _stage(
            PlatformStage.TEXTURE,
            PlatformStage.GENERATE,
            description="Generate and bind PBR textures from approved evidence",
        ),
        _stage(
            PlatformStage.SEGMENT,
            PlatformStage.TEXTURE,
            description="Segment garment parts and trim",
        ),
        _stage(
            PlatformStage.COMPLETE,
            PlatformStage.SEGMENT,
            description="Repair occluded geometry from multiview evidence",
        ),
        _stage(
            PlatformStage.REMESH,
            PlatformStage.COMPLETE,
            description="Create production topology and LOD budget",
        ),
        _stage(
            PlatformStage.UV,
            PlatformStage.REMESH,
            description="Validate UVs and rebake source-bound textures",
        ),
    ]

    export_dependency = PlatformStage.UV
    if request.require_rig:
        stages.extend(
            [
                _stage(
                    PlatformStage.RIGGABILITY,
                    PlatformStage.UV,
                    description="Gate topology and rest pose before rigging",
                ),
                _stage(
                    PlatformStage.RIG,
                    PlatformStage.RIGGABILITY,
                    description="Author and independently validate the production rig",
                ),
            ]
        )
        export_dependency = PlatformStage.RIG
        if request.animation_clip:
            stages.append(
                _stage(
                    PlatformStage.RETARGET,
                    PlatformStage.RIG,
                    description="Run rest-direction gate, then retarget animation",
                )
            )
            export_dependency = PlatformStage.RETARGET

    stages.extend(
        [
            _stage(
                PlatformStage.EXPORT,
                export_dependency,
                description="Export, compress and normalize GLB/USDZ artifacts",
            ),
            _stage(
                PlatformStage.RENDER_PROOFS,
                PlatformStage.EXPORT,
                description="Render fresh fixed-camera proof views from the exported artifact",
            ),
            _stage(
                PlatformStage.DETERMINISTIC_QC,
                PlatformStage.EXPORT,
                PlatformStage.RENDER_PROOFS,
                description="Parse final artifacts with independent numeric validators",
            ),
        ]
    )

    consensus_dependencies = [PlatformStage.DETERMINISTIC_QC]
    if request.quality.require_openai_vision:
        stages.append(
            _stage(
                PlatformStage.OPENAI_VISION_QC,
                PlatformStage.RENDER_PROOFS,
                description="OpenAI vision comparison against the approved evidence bundle",
            )
        )
        consensus_dependencies.append(PlatformStage.OPENAI_VISION_QC)
    if request.quality.require_open_source_vision:
        stages.append(
            _stage(
                PlatformStage.OSS_VISION_QC,
                PlatformStage.RENDER_PROOFS,
                description="Independent open-source vision comparison",
            )
        )
        consensus_dependencies.append(PlatformStage.OSS_VISION_QC)

    stages.append(
        _stage(
            PlatformStage.CONSENSUS,
            *consensus_dependencies,
            description="Fail closed on deterministic failures or judge disagreement",
        )
    )

    release_dependency = PlatformStage.CONSENSUS
    if request.quality.require_founder_approval:
        stages.append(
            _stage(
                PlatformStage.FOUNDER_APPROVAL,
                PlatformStage.CONSENSUS,
                description="Bind founder approval to artifact and evidence hashes",
            )
        )
        release_dependency = PlatformStage.FOUNDER_APPROVAL

    if request.publish:
        stages.append(
            _stage(
                PlatformStage.PUBLISH,
                release_dependency,
                description="Publish only the hash-approved artifact",
            )
        )

    _validate_plan(stages)
    return WorkflowPlan(
        sku=request.sku,
        intent=request.intent,
        generation_mode=request.generation_mode,
        stages=tuple(stages),
        provider_policy=request.providers,
        quality_policy=request.quality,
        evidence_revision=request.evidence.revision if request.evidence else None,
    )


def _validate_plan(stages: list[PlannedStage]) -> None:
    seen: set[PlatformStage] = set()
    for item in stages:
        missing = set(item.requires) - seen
        if missing:
            names = ", ".join(sorted(stage.value for stage in missing))
            raise ValueError(f"stage {item.stage.value} has unresolved dependencies: {names}")
        if item.stage in seen:
            raise ValueError(f"duplicate stage in workflow: {item.stage.value}")
        seen.add(item.stage)


__all__ = ["compile_workflow"]
