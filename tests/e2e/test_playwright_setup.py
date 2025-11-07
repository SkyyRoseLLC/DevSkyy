"""
Playwright Setup Verification Test
Tests that Playwright is correctly installed and configured
"""

import pytest
import re
from playwright.sync_api import Page, expect


class TestPlaywrightSetup:
    """Verify Playwright installation"""

    def test_browser_launches(self, page: Page):
        """Should successfully launch browser and navigate"""
        # Navigate to a known working site
        page.goto("https://playwright.dev/")

        # Verify page loaded
        expect(page).to_have_title(re.compile("Playwright"))

    def test_api_request_context_works(self, playwright):
        """Should create API request context"""
        # Create a request context
        request_context = playwright.request.new_context(
            base_url="https://api.github.com"
        )

        # Make a simple API request
        response = request_context.get("/")

        # Verify response
        assert response.ok
        assert response.status == 200

        request_context.dispose()

    def test_page_interaction(self, page: Page):
        """Should interact with page elements"""
        page.goto("https://playwright.dev/")

        # Find and click a link
        get_started_link = page.get_by_role("link", name="Get started")

        if get_started_link.is_visible():
            get_started_link.click()

            # Verify navigation worked
            expect(page).to_have_url(re.compile(".*intro.*"))
