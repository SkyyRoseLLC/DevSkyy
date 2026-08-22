"""Regression tests for the fail-closed owner provisioning command."""

from __future__ import annotations

import pytest

from database.seed_admin import load_owner_config


def test_owner_config_requires_explicit_email() -> None:
    with pytest.raises(SystemExit, match="ADMIN_EMAIL"):
        load_owner_config(
            {"ADMIN_USERNAME": "founder", "ADMIN_PASSWORD": "StrongP@ss1"},
            password_prompt=lambda: "unused",
        )


def test_owner_config_requires_explicit_username() -> None:
    with pytest.raises(SystemExit, match="ADMIN_USERNAME"):
        load_owner_config(
            {"ADMIN_EMAIL": "owner@skyyrose.co", "ADMIN_PASSWORD": "StrongP@ss1"},
            password_prompt=lambda: "unused",
        )


def test_owner_config_uses_password_prompt_without_env_secret() -> None:
    config = load_owner_config(
        {"ADMIN_EMAIL": "owner@skyyrose.co", "ADMIN_USERNAME": "founder"},
        password_prompt=lambda: "StrongP@ss1",
    )

    assert config.email == "owner@skyyrose.co"
    assert config.username == "founder"
    assert config.password == "StrongP@ss1"


def test_owner_config_rejects_weak_password() -> None:
    with pytest.raises(SystemExit, match="at least 8 characters"):
        load_owner_config(
            {
                "ADMIN_EMAIL": "owner@skyyrose.co",
                "ADMIN_USERNAME": "founder",
                "ADMIN_PASSWORD": "weak",
            },
            password_prompt=lambda: "unused",
        )
