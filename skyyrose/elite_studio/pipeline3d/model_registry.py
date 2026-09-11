"""Capability registry for OpenAI, open-source models and deterministic tools.

Availability is resolved from the current runtime rather than inferred from a
model name.  A configured model and a callable model are different facts; the
registry preserves that distinction for preflight and routing.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

class Capability(StrEnum):
    ARCHITECTURE_PLANNING = "architecture_planning"
    SPEC_COMPILATION = "spec_compilation"
    IMAGE_TO_3D = "image_to_3d"
    MULTIVIEW_TO_3D = "multiview_to_3d"
    TEXT_TO_3D = "text_to_3d"
    TEXTURE = "texture"
    VISION_FIDELITY = "vision_fidelity"
    MESH_PROCESSING = "mesh_processing"
    RIGGING = "rigging"
    RETARGETING = "retargeting"
    GLB_VALIDATION = "glb_validation"


class DeploymentKind(StrEnum):
    OPENAI_API = "openai_api"
    OPEN_SOURCE_LOCAL = "open_source_local"
    OPEN_SOURCE_HOSTED = "open_source_hosted"
    INTERNAL_TOOL = "internal_tool"
    EXTERNAL_API = "external_api"


class LicenseClass(StrEnum):
    OPEN_SOURCE = "open_source"
    INTERNAL = "internal"
    PROPRIETARY_API = "proprietary_api"


class ReadinessLevel(StrEnum):
    UNCONFIGURED = "unconfigured"
    CONFIGURED = "configured"
    VERIFIED = "verified"


class ModelRegistration(BaseModel):
    """One routable model or deterministic production tool."""

    model_config = ConfigDict(frozen=True)

    id: str
    provider: str
    model: str
    deployment: DeploymentKind
    license_class: LicenseClass
    capabilities: tuple[Capability, ...]
    available: bool
    readiness: ReadinessLevel = ReadinessLevel.CONFIGURED
    priority: int = Field(default=100, ge=0)
    endpoint: str | None = None
    reason_unavailable: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)

    @property
    def is_external(self) -> bool:
        return self.deployment == DeploymentKind.EXTERNAL_API


class NoCapableModelError(RuntimeError):
    """Raised when no currently available model can satisfy a capability."""


class ModelRegistry:
    """Deterministic capability routing with owned/open-source preference."""

    def __init__(self, registrations: list[ModelRegistration] | tuple[ModelRegistration, ...]):
        ids = [item.id for item in registrations]
        if len(ids) != len(set(ids)):
            raise ValueError("model registration ids must be unique")
        self._registrations = tuple(registrations)

    @property
    def registrations(self) -> tuple[ModelRegistration, ...]:
        return self._registrations

    def candidates(
        self,
        capability: Capability,
        *,
        allow_external: bool = False,
    ) -> tuple[ModelRegistration, ...]:
        matches = [
            item
            for item in self._registrations
            if item.available
            and capability in item.capabilities
            and (allow_external or not item.is_external)
        ]
        return tuple(sorted(matches, key=lambda item: (item.priority, item.id)))

    def select(
        self,
        capability: Capability,
        *,
        allow_external: bool = False,
    ) -> ModelRegistration:
        candidates = self.candidates(capability, allow_external=allow_external)
        if not candidates:
            raise NoCapableModelError(
                f"no available model for capability={capability.value!r} "
                f"allow_external={allow_external}"
            )
        return candidates[0]

    @classmethod
    def from_environment(cls) -> ModelRegistry:
        """Build the production registry from runtime configuration.

        Open-source LLM/VLM entries use an OpenAI-compatible endpoint so the
        same contract works with vLLM, Ollama, LocalAI or a managed endpoint.
        Model identifiers are explicit environment settings; no arbitrary
        model is silently downloaded during application startup.
        """

        openai_key = bool(os.getenv("OPENAI_API_KEY"))
        oss_endpoint = os.getenv("OSS_MODEL_BASE_URL")
        oss_planner = os.getenv("OSS_3D_PLANNER_MODEL")
        oss_vision = os.getenv("OSS_3D_VISION_MODEL")
        trellis_repo = Path(os.getenv("TRELLIS2_REPO", "vendor/TRELLIS.2")).expanduser()
        trellis_model = os.getenv("TRELLIS2_MODEL", "microsoft/TRELLIS.2-4B")
        blender = shutil.which("blender")
        gltf_transform_version = _command_version(
            ["npx", "--no-install", "@gltf-transform/cli", "--version"]
        )
        blender_version = _command_version([blender, "--version"]) if blender else None

        registrations = [
            ModelRegistration(
                id="openai-3d-planner",
                provider="openai",
                model=os.getenv("OPENAI_3D_PLANNER_MODEL", "gpt-5.5-pro"),
                deployment=DeploymentKind.OPENAI_API,
                license_class=LicenseClass.PROPRIETARY_API,
                capabilities=(Capability.ARCHITECTURE_PLANNING, Capability.SPEC_COMPILATION),
                available=openai_key,
                readiness=(
                    ReadinessLevel.CONFIGURED if openai_key else ReadinessLevel.UNCONFIGURED
                ),
                priority=20,
                reason_unavailable=None if openai_key else "OPENAI_API_KEY is not configured",
            ),
            ModelRegistration(
                id="openai-3d-vision",
                provider="openai",
                model=os.getenv("OPENAI_3D_VISION_MODEL", "gpt-5.5-pro"),
                deployment=DeploymentKind.OPENAI_API,
                license_class=LicenseClass.PROPRIETARY_API,
                capabilities=(Capability.VISION_FIDELITY,),
                available=openai_key,
                readiness=(
                    ReadinessLevel.CONFIGURED if openai_key else ReadinessLevel.UNCONFIGURED
                ),
                priority=20,
                reason_unavailable=None if openai_key else "OPENAI_API_KEY is not configured",
            ),
            ModelRegistration(
                id="oss-3d-planner",
                provider="openai-compatible",
                model=oss_planner or "unconfigured",
                deployment=DeploymentKind.OPEN_SOURCE_LOCAL,
                license_class=LicenseClass.OPEN_SOURCE,
                capabilities=(Capability.ARCHITECTURE_PLANNING, Capability.SPEC_COMPILATION),
                available=bool(oss_endpoint and oss_planner),
                readiness=(
                    ReadinessLevel.CONFIGURED
                    if oss_endpoint and oss_planner
                    else ReadinessLevel.UNCONFIGURED
                ),
                priority=10,
                endpoint=oss_endpoint,
                reason_unavailable=(
                    None
                    if oss_endpoint and oss_planner
                    else "OSS_MODEL_BASE_URL and OSS_3D_PLANNER_MODEL are required"
                ),
            ),
            ModelRegistration(
                id="oss-3d-vision",
                provider="openai-compatible",
                model=oss_vision or "unconfigured",
                deployment=DeploymentKind.OPEN_SOURCE_LOCAL,
                license_class=LicenseClass.OPEN_SOURCE,
                capabilities=(Capability.VISION_FIDELITY,),
                available=bool(oss_endpoint and oss_vision),
                readiness=(
                    ReadinessLevel.CONFIGURED
                    if oss_endpoint and oss_vision
                    else ReadinessLevel.UNCONFIGURED
                ),
                priority=10,
                endpoint=oss_endpoint,
                reason_unavailable=(
                    None
                    if oss_endpoint and oss_vision
                    else "OSS_MODEL_BASE_URL and OSS_3D_VISION_MODEL are required"
                ),
            ),
            ModelRegistration(
                id="trellis2-local",
                provider="trellis2",
                model=trellis_model,
                deployment=DeploymentKind.OPEN_SOURCE_LOCAL,
                license_class=LicenseClass.OPEN_SOURCE,
                capabilities=(
                    Capability.IMAGE_TO_3D,
                    Capability.MULTIVIEW_TO_3D,
                    Capability.TEXTURE,
                ),
                available=trellis_repo.is_dir(),
                readiness=(
                    ReadinessLevel.CONFIGURED
                    if trellis_repo.is_dir()
                    else ReadinessLevel.UNCONFIGURED
                ),
                priority=10,
                reason_unavailable=(
                    None if trellis_repo.is_dir() else f"TRELLIS.2 repo not found: {trellis_repo}"
                ),
                metadata={"repo": str(trellis_repo)},
            ),
            ModelRegistration(
                id="blender-authoring",
                provider="blender",
                model="blender-headless",
                deployment=DeploymentKind.INTERNAL_TOOL,
                license_class=LicenseClass.OPEN_SOURCE,
                capabilities=(
                    Capability.MESH_PROCESSING,
                    Capability.RIGGING,
                    Capability.RETARGETING,
                ),
                available=bool(blender),
                readiness=(
                    ReadinessLevel.VERIFIED
                    if blender_version is not None
                    else ReadinessLevel.UNCONFIGURED
                ),
                priority=10,
                reason_unavailable=None if blender else "blender executable not found",
                metadata={"executable": blender or "", "version": blender_version or ""},
            ),
            ModelRegistration(
                id="gltf-transform-validator",
                provider="gltf-transform",
                model="@gltf-transform/cli",
                deployment=DeploymentKind.INTERNAL_TOOL,
                license_class=LicenseClass.OPEN_SOURCE,
                capabilities=(Capability.GLB_VALIDATION,),
                available=gltf_transform_version is not None,
                readiness=(
                    ReadinessLevel.VERIFIED
                    if gltf_transform_version is not None
                    else ReadinessLevel.UNCONFIGURED
                ),
                priority=10,
                reason_unavailable=(
                    None
                    if gltf_transform_version is not None
                    else "npx --no-install @gltf-transform/cli --version failed"
                ),
                metadata={
                    "executable": shutil.which("npx") or "",
                    "version": gltf_transform_version or "",
                },
            ),
        ]
        return cls(registrations)


def _command_version(command: list[str | None]) -> str | None:
    if not command or any(part is None for part in command):
        return None
    try:
        result = subprocess.run(
            [str(part) for part in command],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    output = (result.stdout or result.stderr).strip()
    return output.splitlines()[0] if output else "verified"


__all__ = [
    "Capability",
    "DeploymentKind",
    "LicenseClass",
    "ModelRegistration",
    "ModelRegistry",
    "NoCapableModelError",
    "ReadinessLevel",
]
