"""Production facade for planning, preflight and fidelity certification.

This facade is intentionally compute-free.  Generation workers consume its
``WorkflowPlan`` and must return receipts; the facade decides whether a job is
allowed to start and whether its output may be released.  Keeping policy out of
workers prevents a provider implementation from weakening its own gates.
"""

from __future__ import annotations

import hashlib
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from .fidelity import FidelityCertifier, FidelityReport, GateReceipt
from .model_registry import (
    Capability,
    DeploymentKind,
    LicenseClass,
    ModelRegistry,
    NoCapableModelError,
    ReadinessLevel,
)
from .platform_contracts import GenerationMode, PlatformRequest, WorkflowPlan
from .workflow import compile_workflow


class PreflightSeverity(StrEnum):
    BLOCKER = "blocker"
    WARNING = "warning"


class PreflightIssue(BaseModel):
    model_config = ConfigDict(frozen=True)

    code: str
    severity: PreflightSeverity
    message: str


class PreflightReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    ready: bool
    issues: tuple[PreflightIssue, ...]
    selected_models: dict[str, str]
    workflow: WorkflowPlan


class SkyyRose3DPlatform:
    """Canonical entry point shared by API, workers, CLI and tests."""

    def __init__(self, registry: ModelRegistry | None = None) -> None:
        self.registry = registry or ModelRegistry.from_environment()

    def plan(self, request: PlatformRequest) -> WorkflowPlan:
        return compile_workflow(request)

    def preflight(self, request: PlatformRequest) -> PreflightReport:
        workflow = self.plan(request)
        issues = self._evidence_issues(request)
        selected: dict[str, str] = {}

        for capability in self._required_capabilities(request):
            try:
                registration = self.registry.select(
                    capability,
                    allow_external=request.providers.allow_external_3d_apis,
                )
            except NoCapableModelError as exc:
                issues.append(
                    PreflightIssue(
                        code=f"model.{capability.value}.unavailable",
                        severity=PreflightSeverity.BLOCKER,
                        message=str(exc),
                    )
                )
            else:
                selected[capability.value] = registration.id
                if registration.readiness != ReadinessLevel.VERIFIED:
                    issues.append(
                        PreflightIssue(
                            code=f"model.{capability.value}.not_live_verified",
                            severity=PreflightSeverity.WARNING,
                            message=(
                                f"{registration.id} is configured but must pass a live worker "
                                "health check before dispatch"
                            ),
                        )
                    )

        vision_candidates = self.registry.candidates(Capability.VISION_FIDELITY)
        if request.quality.require_openai_vision:
            openai = next(
                (
                    item
                    for item in vision_candidates
                    if item.deployment == DeploymentKind.OPENAI_API
                ),
                None,
            )
            if not request.providers.allow_openai:
                openai = None
                message = "provider policy disables OpenAI required by the quality policy"
            else:
                message = "no available OpenAI vision model"
            if openai is None:
                issues.append(
                    PreflightIssue(
                        code="model.openai_vision.unavailable",
                        severity=PreflightSeverity.BLOCKER,
                        message=message,
                    )
                )
            else:
                selected["openai_vision_fidelity"] = openai.id
                if openai.readiness != ReadinessLevel.VERIFIED:
                    issues.append(
                        PreflightIssue(
                            code="model.openai_vision.not_live_verified",
                            severity=PreflightSeverity.WARNING,
                            message="OpenAI vision is configured but not live-verified",
                        )
                    )

        if request.quality.require_open_source_vision:
            oss = next(
                (
                    item
                    for item in vision_candidates
                    if item.license_class == LicenseClass.OPEN_SOURCE
                    and item.deployment
                    in {
                        DeploymentKind.OPEN_SOURCE_LOCAL,
                        DeploymentKind.OPEN_SOURCE_HOSTED,
                    }
                ),
                None,
            )
            if oss is None:
                issues.append(
                    PreflightIssue(
                        code="model.oss_vision.unavailable",
                        severity=PreflightSeverity.BLOCKER,
                        message="no available open-source vision model",
                    )
                )
            else:
                selected["oss_vision_fidelity"] = oss.id
                if oss.readiness != ReadinessLevel.VERIFIED:
                    issues.append(
                        PreflightIssue(
                            code="model.oss_vision.not_live_verified",
                            severity=PreflightSeverity.WARNING,
                            message="open-source vision is configured but not live-verified",
                        )
                    )

        return PreflightReport(
            ready=not any(issue.severity == PreflightSeverity.BLOCKER for issue in issues),
            issues=tuple(issues),
            selected_models=selected,
            workflow=workflow,
        )

    def certify(
        self,
        request: PlatformRequest,
        receipts: list[GateReceipt] | tuple[GateReceipt, ...],
    ) -> FidelityReport:
        return FidelityCertifier(request.quality).certify(
            receipts,
            require_rig=request.require_rig,
            require_animation=bool(request.animation_clip),
        )

    @staticmethod
    def _evidence_issues(request: PlatformRequest) -> list[PreflightIssue]:
        issues: list[PreflightIssue] = []
        bundle = request.evidence
        if bundle is None:
            return issues

        dossier = _local_path(bundle.dossier_uri)
        if dossier is None:
            issues.append(
                PreflightIssue(
                    code="evidence.dossier.not_materialized",
                    severity=PreflightSeverity.BLOCKER,
                    message="dossier must be materialized locally before generation",
                )
            )
        else:
            issues.extend(_file_hash_issues(dossier, bundle.dossier_sha256, "dossier"))

        for asset in bundle.assets:
            path = asset.local_path
            if path is None:
                issues.append(
                    PreflightIssue(
                        code="evidence.asset.not_materialized",
                        severity=PreflightSeverity.BLOCKER,
                        message=f"{asset.role.value} evidence must be downloaded before generation",
                    )
                )
                continue
            issues.extend(_file_hash_issues(path, asset.sha256, asset.role.value))
        return issues

    @staticmethod
    def _required_capabilities(request: PlatformRequest) -> tuple[Capability, ...]:
        generation = {
            GenerationMode.IMAGE: Capability.IMAGE_TO_3D,
            GenerationMode.MULTIVIEW: Capability.MULTIVIEW_TO_3D,
            GenerationMode.TEXT: Capability.TEXT_TO_3D,
        }[request.generation_mode]
        required = [
            generation,
            Capability.TEXTURE,
            Capability.MESH_PROCESSING,
            Capability.GLB_VALIDATION,
        ]
        if request.require_rig:
            required.append(Capability.RIGGING)
        if request.animation_clip:
            required.append(Capability.RETARGETING)
        return tuple(dict.fromkeys(required))


def _local_path(uri: str) -> Path | None:
    if uri.startswith(("http://", "https://", "s3://", "r2://")):
        return None
    return Path(uri).expanduser()


def _file_hash_issues(path: Path, expected: str, label: str) -> list[PreflightIssue]:
    if not path.is_file():
        return [
            PreflightIssue(
                code="evidence.file.missing",
                severity=PreflightSeverity.BLOCKER,
                message=f"{label} evidence file not found: {path}",
            )
        ]
    actual = _sha256(path)
    if actual.lower() != expected.lower():
        return [
            PreflightIssue(
                code="evidence.hash.mismatch",
                severity=PreflightSeverity.BLOCKER,
                message=f"{label} hash mismatch: expected {expected}, got {actual}",
            )
        ]
    return []


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


__all__ = [
    "PreflightIssue",
    "PreflightReport",
    "PreflightSeverity",
    "SkyyRose3DPlatform",
]
