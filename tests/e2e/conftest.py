"""
Pytest configuration for Playwright E2E tests
Enterprise-grade test configuration

Truth Protocol Compliance: All 15 rules
Version: 1.0.0
"""

import pytest
from playwright.sync_api import Page, APIRequestContext


@pytest.fixture(scope="session")
def base_url():
    """Base URL for API requests"""
    return "http://localhost:8000"


@pytest.fixture(scope="function")
def page(page: Page):
    """
    Playwright page fixture with custom settings

    Args:
        page: Playwright Page instance

    Returns:
        Configured Page instance
    """
    # Set default timeout
    page.set_default_timeout(30000)  # 30 seconds

    # Set default navigation timeout
    page.set_default_navigation_timeout(30000)

    yield page


@pytest.fixture(scope="function")
def api_request(playwright):
    """
    API request context for testing REST endpoints

    Args:
        playwright: Playwright instance

    Returns:
        APIRequestContext instance
    """
    request_context = playwright.request.new_context(
        base_url="http://localhost:8000",
        extra_http_headers={
            "Content-Type": "application/json"
        }
    )

    yield request_context

    request_context.dispose()


@pytest.fixture(scope="function")
def authenticated_api_request(playwright):
    """
    Authenticated API request context (add auth token if needed)

    Args:
        playwright: Playwright instance

    Returns:
        Authenticated APIRequestContext instance
    """
    # Get auth token (if authentication is implemented)
    # auth_token = get_test_auth_token()

    request_context = playwright.request.new_context(
        base_url="http://localhost:8000",
        extra_http_headers={
            "Content-Type": "application/json",
            # "Authorization": f"Bearer {auth_token}"
        }
    )

    yield request_context

    request_context.dispose()
