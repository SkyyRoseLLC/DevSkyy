"""Authentication dependencies for the dashboard-facing v2 control plane."""

from __future__ import annotations

import hmac
import os

from fastapi import Header, HTTPException, status
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError

from security.jwt_oauth2_auth import TokenPayload, UserRole, jwt_manager

_DASHBOARD_ROLES = frozenset({UserRole.ADMIN, UserRole.SUPER_ADMIN})


async def require_dashboard_operator_or_api_key(
    authorization: str | None = Header(default=None, alias="Authorization"),
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> TokenPayload | None:
    """Authorize an operator JWT or an existing server-to-server API key.

    Dashboard requests arrive through the Next.js relay with a short-lived
    bearer token. Internal automation can continue using ``X-API-Key``.
    Both credentials are valid alternatives in every environment.
    """
    expected_api_key = os.getenv("API_KEY", "")
    if expected_api_key and x_api_key and hmac.compare_digest(x_api_key, expected_api_key):
        return None

    if authorization and authorization.startswith("Bearer "):
        try:
            user = jwt_manager.validate_token(authorization[7:])
        except ExpiredSignatureError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Token has expired",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc
        except InvalidTokenError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid authentication token",
                headers={"WWW-Authenticate": "Bearer"},
            ) from exc

        if not user.has_any_role(_DASHBOARD_ROLES):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Dashboard operator role required",
            )
        return user

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Dashboard operator authentication required",
        headers={"WWW-Authenticate": "Bearer"},
    )
