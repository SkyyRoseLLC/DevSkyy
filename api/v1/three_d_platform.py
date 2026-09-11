"""Governed SkyyRose product-to-3D platform API.

These endpoints expose planning and evidence gates without pretending that a
configured model is operational.  Generation workers may consume the returned
workflow only after ``/preflight`` reports ``ready=true``.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, ConfigDict

from security.jwt_oauth2_auth import RoleChecker, UserRole
from skyyrose.elite_studio.pipeline3d.fidelity import FidelityReport, GateReceipt
from skyyrose.elite_studio.pipeline3d.platform_contracts import PlatformRequest, WorkflowPlan
from skyyrose.elite_studio.pipeline3d.production import PreflightReport, SkyyRose3DPlatform
from skyyrose.elite_studio.pipeline3d.specification import (
    CollaborativeSpecCompiler,
    SpecConsensusError,
    SpecificationResult,
)
from skyyrose.elite_studio.pipeline3d.vision_clients import (
    OpenAICompatibleVisionClient,
    VisionClientError,
)

require_operator = RoleChecker([UserRole.ADMIN, UserRole.DEVELOPER])
router = APIRouter(
    prefix="/3d-platform",
    tags=["SkyyRose 3D Platform"],
    dependencies=[Depends(require_operator)],
)


def get_platform() -> SkyyRose3DPlatform:
    return SkyyRose3DPlatform()


PlatformDep = Annotated[SkyyRose3DPlatform, Depends(get_platform)]


class CertificationRequest(BaseModel):
    model_config = ConfigDict(frozen=True)

    request: PlatformRequest
    receipts: tuple[GateReceipt, ...]


@router.get("/capabilities", summary="Inspect configured model and tool capabilities")
async def capabilities(platform: PlatformDep) -> dict:
    return {
        "ok": True,
        "platform_version": "skyyrose-3d/v1",
        "models": [item.model_dump(mode="json") for item in platform.registry.registrations],
    }


@router.post("/plan", response_model=WorkflowPlan, summary="Compile the production workflow")
async def plan(request: PlatformRequest, platform: PlatformDep) -> WorkflowPlan:
    return platform.plan(request)


@router.post(
    "/preflight",
    response_model=PreflightReport,
    summary="Verify evidence, models and deterministic tools before compute",
)
async def preflight(request: PlatformRequest, platform: PlatformDep) -> PreflightReport:
    return platform.preflight(request)


@router.post(
    "/specification",
    response_model=SpecificationResult,
    summary="Compile and independently audit an evidence-bound generation specification",
)
async def specification(request: PlatformRequest) -> SpecificationResult:
    openai = None
    oss = None
    try:
        openai = OpenAICompatibleVisionClient.openai()
        oss = OpenAICompatibleVisionClient.open_source()
        compiler = CollaborativeSpecCompiler(openai=openai, open_source=oss)
        return await compiler.compile(request)
    except (SpecConsensusError, VisionClientError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc
    finally:
        if openai is not None:
            await openai.close()
        if oss is not None:
            await oss.close()


@router.post(
    "/certify",
    response_model=FidelityReport,
    summary="Evaluate independent release receipts",
)
async def certify(body: CertificationRequest, platform: PlatformDep) -> FidelityReport:
    return platform.certify(body.request, body.receipts)


__all__ = ["router", "get_platform"]
