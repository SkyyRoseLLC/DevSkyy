from __future__ import annotations

from api.v1.three_d_platform import router


def test_platform_router_exposes_governed_surface() -> None:
    paths = {route.path for route in router.routes}

    assert paths == {
        "/3d-platform/capabilities",
        "/3d-platform/plan",
        "/3d-platform/preflight",
        "/3d-platform/specification",
        "/3d-platform/certify",
    }


def test_platform_router_requires_operator_authentication() -> None:
    assert router.dependencies
