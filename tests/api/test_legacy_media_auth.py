"""Legacy media routers must reject unauthenticated requests."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from api.three_d import three_d_router
from api.visual import visual_router


def test_three_d_router_requires_authentication() -> None:
    app = FastAPI()
    app.include_router(three_d_router, prefix="/api/v1")

    response = TestClient(app).get("/api/v1/3d/status")

    assert response.status_code == 401


def test_visual_router_requires_authentication() -> None:
    app = FastAPI()
    app.include_router(visual_router, prefix="/api/v1")

    response = TestClient(app).get("/api/v1/visual/providers")

    assert response.status_code == 401
