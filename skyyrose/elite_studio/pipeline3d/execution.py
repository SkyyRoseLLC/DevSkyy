"""Resumable execution kernel for the canonical production workflow."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from .fidelity import FidelityReport, GateReceipt
from .platform_contracts import PlatformRequest, PlatformStage, WorkflowPlan
from .production import PreflightReport, SkyyRose3DPlatform


class ExecutionStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    BLOCKED = "blocked"
    FAILED = "failed"
    SUCCEEDED = "succeeded"


class HandlerKind(StrEnum):
    CONTROL = "control"
    PRODUCER = "producer"
    DETERMINISTIC_VERIFIER = "deterministic_verifier"
    OPENAI_JUDGE = "openai_judge"
    OSS_JUDGE = "oss_judge"
    HUMAN_APPROVER = "human_approver"
    PUBLISHER = "publisher"


class StageExecutionResult(BaseModel):
    """One handler result. Failure is explicit and never converted to a skip."""

    model_config = ConfigDict(frozen=True)

    stage: PlatformStage
    success: bool
    artifacts: tuple[str, ...] = ()
    receipts: tuple[GateReceipt, ...] = ()
    detail: dict[str, Any] = Field(default_factory=dict)
    error: str | None = None


class ExecutionManifest(BaseModel):
    """Durable job state written atomically after every stage."""

    model_config = ConfigDict(frozen=True)

    version: str = "skyyrose-3d-execution/v1"
    job_id: str
    request_fingerprint: str
    status: ExecutionStatus
    workflow: WorkflowPlan
    preflight: PreflightReport
    stages: tuple[StageExecutionResult, ...] = ()
    created_at: datetime
    updated_at: datetime
    error: str | None = None
    manifest_path: str | None = None
    fidelity: FidelityReport | None = None


@runtime_checkable
class StageHandler(Protocol):
    id: str
    kind: HandlerKind

    def supports(self, stage: PlatformStage) -> bool:
        ...

    async def run(
        self,
        *,
        stage: PlatformStage,
        request: PlatformRequest,
        plan: WorkflowPlan,
        previous: tuple[StageExecutionResult, ...],
    ) -> StageExecutionResult:
        ...


class HandlerRegistry:
    """Priority-ordered runtime handlers; no silent fallback or no-op stages."""

    def __init__(self, handlers: list[StageHandler] | tuple[StageHandler, ...]) -> None:
        ids = [handler.id for handler in handlers]
        if len(ids) != len(set(ids)):
            raise ValueError("stage handler ids must be unique")
        self.handlers = tuple(handlers)

    def pick(self, stage: PlatformStage) -> StageHandler | None:
        required_kind = _required_handler_kind(stage)
        return next(
            (
                handler
                for handler in self.handlers
                if handler.kind == required_kind and handler.supports(stage)
            ),
            None,
        )


class PlatformExecutionEngine:
    """Execute every compiled stage and persist a resumable manifest."""

    def __init__(
        self,
        *,
        platform: SkyyRose3DPlatform,
        handlers: HandlerRegistry,
        manifest_dir: Path,
    ) -> None:
        self.platform = platform
        self.handlers = handlers
        self.manifest_dir = manifest_dir

    async def run(self, request: PlatformRequest, *, job_id: str | None = None) -> ExecutionManifest:
        plan = self.platform.plan(request)
        preflight = self.platform.preflight(request)
        now = datetime.now(UTC)
        job_id = job_id or f"3d_{uuid.uuid4().hex[:16]}"
        manifest_path = self.manifest_dir / f"{job_id}.json"
        manifest = ExecutionManifest(
            job_id=job_id,
            request_fingerprint=_request_fingerprint(request),
            status=ExecutionStatus.PENDING,
            workflow=plan,
            preflight=preflight,
            created_at=now,
            updated_at=now,
            manifest_path=str(manifest_path),
        )

        if not preflight.ready:
            blockers = [issue.message for issue in preflight.issues if issue.severity == "blocker"]
            manifest = manifest.model_copy(
                update={
                    "status": ExecutionStatus.BLOCKED,
                    "error": "preflight blocked: " + " | ".join(blockers),
                    "updated_at": datetime.now(UTC),
                }
            )
            self._persist(manifest)
            return manifest

        manifest = manifest.model_copy(
            update={"status": ExecutionStatus.RUNNING, "updated_at": datetime.now(UTC)}
        )
        self._persist(manifest)

        completed: list[StageExecutionResult] = []
        for planned in plan.stages:
            if planned.stage == PlatformStage.PUBLISH:
                fidelity = self._certify(request, completed)
                if not fidelity.certified:
                    return self._halt(
                        manifest,
                        completed,
                        ExecutionStatus.BLOCKED,
                        "publication blocked by fidelity certification: "
                        + " | ".join(fidelity.blockers),
                        fidelity=fidelity,
                    )
            handler = self.handlers.pick(planned.stage)
            if handler is None:
                return self._halt(
                    manifest,
                    completed,
                    ExecutionStatus.BLOCKED,
                    f"no handler registered for stage={planned.stage.value}",
                )
            try:
                result = await handler.run(
                    stage=planned.stage,
                    request=request,
                    plan=plan,
                    previous=tuple(completed),
                )
            except Exception as exc:  # noqa: BLE001 - handler isolation boundary
                return self._halt(
                    manifest,
                    completed,
                    ExecutionStatus.FAILED,
                    f"handler {handler.id} failed at {planned.stage.value}: {exc}",
                )
            if result.stage != planned.stage:
                return self._halt(
                    manifest,
                    completed,
                    ExecutionStatus.FAILED,
                    f"handler {handler.id} returned {result.stage.value} for {planned.stage.value}",
                )
            completed.append(result)
            if not result.success:
                return self._halt(
                    manifest,
                    completed,
                    ExecutionStatus.FAILED,
                    result.error or f"stage failed: {planned.stage.value}",
                )
            manifest = manifest.model_copy(
                update={"stages": tuple(completed), "updated_at": datetime.now(UTC)}
            )
            self._persist(manifest)

        fidelity = self._certify(request, completed)
        if not fidelity.certified:
            return self._halt(
                manifest,
                completed,
                ExecutionStatus.BLOCKED,
                "execution finished but certification failed: "
                + " | ".join(fidelity.blockers),
                fidelity=fidelity,
            )

        manifest = manifest.model_copy(
            update={
                "status": ExecutionStatus.SUCCEEDED,
                "stages": tuple(completed),
                "updated_at": datetime.now(UTC),
                "fidelity": fidelity,
            }
        )
        self._persist(manifest)
        return manifest

    def _halt(
        self,
        manifest: ExecutionManifest,
        completed: list[StageExecutionResult],
        status: ExecutionStatus,
        error: str,
        *,
        fidelity: FidelityReport | None = None,
    ) -> ExecutionManifest:
        halted = manifest.model_copy(
            update={
                "status": status,
                "stages": tuple(completed),
                "error": error,
                "updated_at": datetime.now(UTC),
                "fidelity": fidelity,
            }
        )
        self._persist(halted)
        return halted

    def _persist(self, manifest: ExecutionManifest) -> None:
        self.manifest_dir.mkdir(parents=True, exist_ok=True)
        target = Path(manifest.manifest_path or self.manifest_dir / f"{manifest.job_id}.json")
        temporary = target.with_suffix(f"{target.suffix}.{os.getpid()}.tmp")
        temporary.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
        os.replace(temporary, target)

    def _certify(
        self,
        request: PlatformRequest,
        completed: list[StageExecutionResult],
    ) -> FidelityReport:
        receipts = tuple(receipt for result in completed for receipt in result.receipts)
        return self.platform.certify(request, receipts)


def _request_fingerprint(request: PlatformRequest) -> str:
    payload = json.dumps(
        request.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _required_handler_kind(stage: PlatformStage) -> HandlerKind:
    if stage in {
        PlatformStage.INGEST,
        PlatformStage.SPEC,
        PlatformStage.PREPROCESS,
        PlatformStage.CONSENSUS,
    }:
        return HandlerKind.CONTROL
    if stage in {
        PlatformStage.GENERATE,
        PlatformStage.TEXTURE,
        PlatformStage.SEGMENT,
        PlatformStage.COMPLETE,
        PlatformStage.REMESH,
        PlatformStage.UV,
        PlatformStage.RIG,
        PlatformStage.RETARGET,
        PlatformStage.EXPORT,
        PlatformStage.RENDER_PROOFS,
    }:
        return HandlerKind.PRODUCER
    if stage in {
        PlatformStage.EVIDENCE_GATE,
        PlatformStage.RIGGABILITY,
        PlatformStage.DETERMINISTIC_QC,
    }:
        return HandlerKind.DETERMINISTIC_VERIFIER
    if stage == PlatformStage.OPENAI_VISION_QC:
        return HandlerKind.OPENAI_JUDGE
    if stage == PlatformStage.OSS_VISION_QC:
        return HandlerKind.OSS_JUDGE
    if stage == PlatformStage.FOUNDER_APPROVAL:
        return HandlerKind.HUMAN_APPROVER
    if stage == PlatformStage.PUBLISH:
        return HandlerKind.PUBLISHER
    raise ValueError(f"unclassified platform stage: {stage.value}")


__all__ = [
    "ExecutionManifest",
    "ExecutionStatus",
    "HandlerKind",
    "HandlerRegistry",
    "PlatformExecutionEngine",
    "StageExecutionResult",
    "StageHandler",
]
