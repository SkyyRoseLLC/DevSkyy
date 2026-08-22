"""Provision exactly one DevSkyy owner account through a trusted terminal.

Usage (interactive password prompt):
    ADMIN_EMAIL=owner@skyyrose.co ADMIN_USERNAME=founder python -m database.seed_admin

For non-interactive automation, provide ADMIN_PASSWORD only through the host's
secret store. This command never emits the password and refuses to overwrite an
existing account.
"""

from __future__ import annotations

import asyncio
import getpass
import os
import sys
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass

from pydantic import ValidationError

from security.jwt_oauth2_auth import UserCreate, UserRole, password_manager


@dataclass(frozen=True)
class OwnerConfig:
    """Validated owner configuration kept only for the duration of provisioning."""

    email: str
    username: str
    password: str


def load_owner_config(
    env: Mapping[str, str] | None = None,
    *,
    password_prompt: Callable[[], str] | None = None,
) -> OwnerConfig:
    """Load and validate explicit owner input without accepting insecure defaults."""
    values = env if env is not None else os.environ
    email = values.get("ADMIN_EMAIL", "").strip().lower()
    username = values.get("ADMIN_USERNAME", "").strip()
    password = values.get("ADMIN_PASSWORD", "")

    if not email:
        raise SystemExit("ADMIN_EMAIL is required; no default owner email is permitted.")
    if not username:
        raise SystemExit("ADMIN_USERNAME is required; no default owner username is permitted.")
    if not password and password_prompt is not None:
        password = password_prompt()
    if not password:
        raise SystemExit("ADMIN_PASSWORD is required outside an interactive terminal.")

    try:
        validated = UserCreate(
            email=email,
            username=username,
            password=password,
            roles=[UserRole.SUPER_ADMIN],
        )
    except ValidationError as error:
        messages = "; ".join(issue["msg"] for issue in error.errors())
        raise SystemExit(messages) from error

    return OwnerConfig(
        email=str(validated.email),
        username=validated.username,
        password=validated.password,
    )


def prompt_owner_password() -> str:
    """Prompt twice on a TTY so the owner password never reaches shell history."""
    if not sys.stdin.isatty():
        raise SystemExit("ADMIN_PASSWORD is required outside an interactive terminal.")

    password = getpass.getpass("Owner dashboard password: ")
    confirmation = getpass.getpass("Confirm owner dashboard password: ")
    if password != confirmation:
        raise SystemExit("Password confirmation does not match.")
    return password


async def seed_admin(config: OwnerConfig | None = None) -> None:
    """Create a new super-admin account and fail closed on every duplicate."""
    from database.db import User, UserRepository, db_manager

    owner = config or load_owner_config(
        password_prompt=prompt_owner_password if not os.getenv("ADMIN_PASSWORD") else None
    )
    await db_manager.initialize()

    async with db_manager.session() as db:
        repo = UserRepository(db)
        duplicate_username = await repo.get_by_username(owner.username)
        duplicate_email = await repo.get_by_email(owner.email)
        if duplicate_username or duplicate_email:
            raise SystemExit(
                "Refusing to overwrite an existing owner or operator account. "
                "Use the account recovery process instead."
            )

        admin = User(
            id=str(uuid.uuid4()),
            email=owner.email,
            username=owner.username,
            hashed_password=password_manager.hash_password(owner.password),
            role=UserRole.SUPER_ADMIN.value,
            is_active=True,
            is_verified=True,
        )
        db.add(admin)
        await db.commit()

    print(f"Owner account created: {owner.username} / {owner.email}")
    print("Role: super_admin")
    print("Password: not displayed")


if __name__ == "__main__":
    asyncio.run(seed_admin())
