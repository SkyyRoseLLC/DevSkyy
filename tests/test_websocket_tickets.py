import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from jwt.exceptions import InvalidTokenError
from starlette.websockets import WebSocketDisconnect

from api import websocket
from security import jwt_oauth2_auth
from security.jwt_oauth2_auth import JWTConfig, JWTManager, TokenType, UserRole

TEST_SECRET = "s" * 64
TEST_REFRESH_SECRET = "r" * 64


def make_manager() -> JWTManager:
    return JWTManager(JWTConfig(secret_key=TEST_SECRET, refresh_secret_key=TEST_REFRESH_SECRET))


def test_websocket_ticket_is_channel_bound_and_not_an_access_token() -> None:
    manager = make_manager()
    ticket = manager.create_websocket_ticket("owner-1", [UserRole.SUPER_ADMIN.value], "round_table")

    payload = manager.validate_token(ticket, TokenType.WEBSOCKET)

    assert payload.sub == "owner-1"
    assert payload.channel == "round_table"
    assert payload.has_role(UserRole.SUPER_ADMIN)


def test_websocket_ticket_cannot_be_used_as_an_access_token() -> None:
    manager = make_manager()
    ticket = manager.create_websocket_ticket("owner-1", [UserRole.SUPER_ADMIN.value], "round_table")

    with pytest.raises(InvalidTokenError, match="Invalid token type"):
        manager.validate_token(ticket, TokenType.ACCESS)


def test_websocket_ticket_rejects_a_different_channel(monkeypatch) -> None:
    manager = make_manager()
    ticket = manager.create_websocket_ticket("owner-1", [UserRole.SUPER_ADMIN.value], "round_table")
    monkeypatch.setattr(websocket, "jwt_manager", manager)

    assert websocket.validate_websocket_ticket(ticket, "round_table") is not None
    assert websocket.validate_websocket_ticket(ticket, "3d_pipeline") is None


def test_websocket_requires_ticket_and_accepts_matching_subprotocol(monkeypatch) -> None:
    manager = make_manager()
    monkeypatch.setattr(websocket, "jwt_manager", manager)
    app = FastAPI()
    app.include_router(websocket.ws_router)
    client = TestClient(app)

    with pytest.raises(WebSocketDisconnect) as missing_ticket:
        with client.websocket_connect(
            "/api/ws/agents", headers={"Origin": "http://localhost:3000"}
        ):
            pass
    assert missing_ticket.value.code == 4401

    ticket = manager.create_websocket_ticket("owner-1", [UserRole.SUPER_ADMIN.value], "agents")
    with client.websocket_connect(
        "/api/ws/agents",
        subprotocols=[websocket.WEBSOCKET_TICKET_PROTOCOL, ticket],
        headers={"Origin": "http://localhost:3000"},
    ) as connection:
        connection.send_json({"type": "ping"})
        assert connection.receive_json()["type"] == "pong"

    other_channel_ticket = manager.create_websocket_ticket(
        "owner-1", [UserRole.SUPER_ADMIN.value], "round_table"
    )
    with pytest.raises(WebSocketDisconnect) as channel_mismatch:
        with client.websocket_connect(
            "/api/ws/agents",
            subprotocols=[websocket.WEBSOCKET_TICKET_PROTOCOL, other_channel_ticket],
            headers={"Origin": "http://localhost:3000"},
        ):
            pass
    assert channel_mismatch.value.code == 4401

    with pytest.raises(WebSocketDisconnect) as bad_origin:
        with client.websocket_connect(
            "/api/ws/agents",
            subprotocols=[websocket.WEBSOCKET_TICKET_PROTOCOL, ticket],
            headers={"Origin": "https://untrusted.example"},
        ):
            pass
    assert bad_origin.value.code == 4403


def test_ticket_issuance_requires_admin_and_known_channel(monkeypatch) -> None:
    manager = make_manager()
    monkeypatch.setattr(jwt_oauth2_auth, "jwt_manager", manager)
    app = FastAPI()
    app.include_router(jwt_oauth2_auth.auth_router)
    client = TestClient(app)

    assert client.post("/api/v1/auth/ws-ticket", json={"channel": "agents"}).status_code == 401

    api_user_token = manager.create_access_token("reader-1", [UserRole.API_USER.value])
    denied = client.post(
        "/api/v1/auth/ws-ticket",
        headers={"Authorization": f"Bearer {api_user_token}"},
        json={"channel": "agents"},
    )
    assert denied.status_code == 403

    operator_token = manager.create_access_token("owner-1", [UserRole.SUPER_ADMIN.value])
    unknown_channel = client.post(
        "/api/v1/auth/ws-ticket",
        headers={"Authorization": f"Bearer {operator_token}"},
        json={"channel": "not_a_channel"},
    )
    assert unknown_channel.status_code == 422

    issued = client.post(
        "/api/v1/auth/ws-ticket",
        headers={"Authorization": f"Bearer {operator_token}"},
        json={"channel": "agents"},
    )
    assert issued.status_code == 200
    payload = manager.validate_token(issued.json()["ticket"], TokenType.WEBSOCKET)
    assert payload.channel == "agents"
