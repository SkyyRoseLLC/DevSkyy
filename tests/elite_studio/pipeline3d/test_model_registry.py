from __future__ import annotations

from skyyrose.elite_studio.pipeline3d.model_registry import (
    Capability,
    DeploymentKind,
    LicenseClass,
    ModelRegistration,
    ModelRegistry,
    ReadinessLevel,
)


def test_registry_prefers_owned_open_source_generator() -> None:
    registry = ModelRegistry(
        [
            ModelRegistration(
                id="external",
                provider="external",
                model="paid-api",
                deployment=DeploymentKind.EXTERNAL_API,
                license_class=LicenseClass.PROPRIETARY_API,
                capabilities=(Capability.IMAGE_TO_3D,),
                available=True,
                priority=20,
            ),
            ModelRegistration(
                id="owned",
                provider="trellis",
                model="microsoft/TRELLIS.2-4B",
                deployment=DeploymentKind.OPEN_SOURCE_LOCAL,
                license_class=LicenseClass.OPEN_SOURCE,
                capabilities=(Capability.IMAGE_TO_3D,),
                available=True,
                priority=10,
            ),
        ]
    )

    assert registry.select(Capability.IMAGE_TO_3D).id == "owned"


def test_external_provider_is_excluded_by_default() -> None:
    registry = ModelRegistry(
        [
            ModelRegistration(
                id="external",
                provider="external",
                model="paid-api",
                deployment=DeploymentKind.EXTERNAL_API,
                license_class=LicenseClass.PROPRIETARY_API,
                capabilities=(Capability.IMAGE_TO_3D,),
                available=True,
            )
        ]
    )

    assert registry.candidates(Capability.IMAGE_TO_3D) == ()
    assert registry.candidates(Capability.IMAGE_TO_3D, allow_external=True)[0].id == "external"


def test_default_registration_is_configured_not_falsely_live_verified() -> None:
    registration = ModelRegistration(
        id="configured",
        provider="local",
        model="model",
        deployment=DeploymentKind.OPEN_SOURCE_LOCAL,
        license_class=LicenseClass.OPEN_SOURCE,
        capabilities=(Capability.VISION_FIDELITY,),
        available=True,
    )

    assert registration.readiness == ReadinessLevel.CONFIGURED
