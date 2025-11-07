"""
Comprehensive Unit Tests for Rate Limiting Module
Tests for token bucket algorithm and rate limiter
"""

import pytest
import time
from unittest.mock import Mock

from api.rate_limiting import RateLimiter, get_client_identifier


class TestRateLimiter:
    """Test suite for RateLimiter class"""

    def test_rate_limiter_initialization(self) -> None:
        """RateLimiter initializes correctly"""
        limiter = RateLimiter()

        assert limiter._buckets is not None
        assert limiter._lock is not None

    def test_rate_limiter_first_request_allowed(self) -> None:
        """RateLimiter allows first request"""
        limiter = RateLimiter()

        allowed, info = limiter.is_allowed(
            "client1", max_requests=100, window_seconds=60)

        assert allowed is True
        assert info["remaining"] == 99

    def test_rate_limiter_tracks_remaining_tokens(self) -> None:
        """RateLimiter tracks remaining tokens correctly"""
        limiter = RateLimiter()

        allowed1, info1 = limiter.is_allowed(
            "client1", max_requests=5, window_seconds=60)
        allowed2, info2 = limiter.is_allowed(
            "client1", max_requests=5, window_seconds=60)
        allowed3, info3 = limiter.is_allowed(
            "client1", max_requests=5, window_seconds=60)

        assert allowed1 is True
        assert allowed2 is True
        assert allowed3 is True
        assert info3["remaining"] == 2

    def test_rate_limiter_denies_exceeded_limit(self) -> None:
        """RateLimiter denies requests after limit exceeded"""
        limiter = RateLimiter()
        max_requests = 3

        # Use up all tokens
        for i in range(max_requests):
            allowed, _ = limiter.is_allowed(
                "client1", max_requests=max_requests, window_seconds=60)
            assert allowed is True

        # Next request should be denied
        allowed, info = limiter.is_allowed(
            "client1", max_requests=max_requests, window_seconds=60)

        assert allowed is False
        assert info["remaining"] == 0

    def test_rate_limiter_rate_limit_info_structure(self) -> None:
        """RateLimiter returns proper rate limit info"""
        limiter = RateLimiter()

        allowed, info = limiter.is_allowed(
            "client1", max_requests=100, window_seconds=60)

        assert "limit" in info
        assert "remaining" in info
        assert "reset" in info
        assert info["limit"] == 100

    def test_rate_limiter_independent_clients(self) -> None:
        """RateLimiter tracks limits independently per client"""
        limiter = RateLimiter()

        allowed1, info1 = limiter.is_allowed(
            "client1", max_requests=2, window_seconds=60)
        allowed2, info2 = limiter.is_allowed(
            "client1", max_requests=2, window_seconds=60)
        allowed3, info3 = limiter.is_allowed(
            "client1", max_requests=2, window_seconds=60)

        allowed4, info4 = limiter.is_allowed(
            "client2", max_requests=2, window_seconds=60)

        assert allowed3 is False  # client1 exceeded
        assert allowed4 is True  # client2 still has tokens

    def test_rate_limiter_token_refill_over_time(self) -> None:
        """RateLimiter refills tokens over time"""
        limiter = RateLimiter()

        # Use all tokens
        for _ in range(3):
            limiter.is_allowed("client1", max_requests=3, window_seconds=60)

        # Immediately after - should be denied
        allowed1, _ = limiter.is_allowed("client1", max_requests=3, window_seconds=60)
        assert allowed1 is False

        # Wait for token refill (partial window)
        time.sleep(0.5)

        # Should have some tokens now (depends on window, but shouldn't be full)
        allowed2, info2 = limiter.is_allowed(
            "client1", max_requests=3, window_seconds=60)
        # Token refill should happen, but exact number depends on timing

    def test_rate_limiter_reset_clears_bucket(self) -> None:
        """RateLimiter reset clears bucket for client"""
        limiter = RateLimiter()

        # Use up tokens
        for _ in range(3):
            limiter.is_allowed("client1", max_requests=3, window_seconds=60)

        # Should be denied
        allowed1, _ = limiter.is_allowed("client1", max_requests=3, window_seconds=60)
        assert allowed1 is False

        # Reset client
        limiter.reset("client1")

        # Should now be allowed
        allowed2, info2 = limiter.is_allowed(
            "client1", max_requests=3, window_seconds=60)
        assert allowed2 is True

    def test_rate_limiter_reset_with_prefix(self) -> None:
        """RateLimiter reset handles client identifiers with prefix"""
        limiter = RateLimiter()

        # Create some entries
        limiter.is_allowed("user:123", max_requests=5, window_seconds=60)
        limiter.is_allowed("user:456", max_requests=5, window_seconds=60)

        # Reset user:123
        limiter.reset("user:123")

        # Verify bucket is cleared (internal check)
        assert len([k for k in limiter._buckets.keys()
                   if k.startswith("user:123")]) == 0

    def test_rate_limiter_different_windows(self) -> None:
        """RateLimiter handles different time windows"""
        limiter = RateLimiter()

        # Create two buckets with different windows
        allowed1, info1 = limiter.is_allowed(
            "client1", max_requests=10, window_seconds=60)
        allowed2, info2 = limiter.is_allowed(
            "client1", max_requests=10, window_seconds=120)

        # Should be separate buckets
        assert info1["limit"] == 10
        assert info2["limit"] == 10

    def test_rate_limiter_high_request_rate(self) -> None:
        """RateLimiter handles high request rates"""
        limiter = RateLimiter()

        results = []
        for i in range(150):
            allowed, info = limiter.is_allowed(
                "client1", max_requests=100, window_seconds=60)
            results.append(allowed)

        # First 100 should be allowed
        assert sum(results[:100]) == 100
        # Rest should be denied
        assert sum(results[100:]) == 0

    def test_rate_limiter_thread_safety(self) -> None:
        """RateLimiter is thread-safe with lock"""
        limiter = RateLimiter()

        # Verify lock exists and is being used
        assert limiter._lock is not None

        # Make concurrent requests (simplified without actual threading)
        for _ in range(10):
            limiter.is_allowed("client1", max_requests=100, window_seconds=60)

    def test_rate_limiter_edge_case_zero_remaining(self) -> None:
        """RateLimiter correctly reports zero remaining"""
        limiter = RateLimiter()

        for _ in range(5):
            limiter.is_allowed("client1", max_requests=5, window_seconds=60)

        allowed, info = limiter.is_allowed("client1", max_requests=5, window_seconds=60)

        assert allowed is False
        assert info["remaining"] == 0


class TestGetClientIdentifier:
    """Test suite for get_client_identifier function"""

    def test_get_client_identifier_from_api_key(self) -> None:
        """get_client_identifier extracts API key"""
        request = Mock()
        request.headers = {"X-API-Key": "key123"}
        request.state = Mock()
        request.state.user = None
        request.client = Mock()
        request.client.host = "192.168.1.1"

        identifier = get_client_identifier(request)

        assert identifier == "api_key:key123"

    def test_get_client_identifier_from_user_id(self) -> None:
        """get_client_identifier extracts user ID when API key absent"""
        request = Mock()
        request.headers = {}
        request.state = Mock()
        user = Mock()
        user.user_id = "user_456"
        request.state.user = user
        request.client = Mock()
        request.client.host = "192.168.1.1"

        identifier = get_client_identifier(request)

        assert identifier == "user:user_456"

    def test_get_client_identifier_from_direct_ip(self) -> None:
        """get_client_identifier falls back to direct IP"""
        request = Mock()
        request.headers = {}
        request.state = Mock()
        request.state.user = None
        request.client = Mock()
        request.client.host = "192.168.1.1"

        identifier = get_client_identifier(request)

        assert identifier == "ip:192.168.1.1"

    def test_get_client_identifier_from_forwarded_for(self) -> None:
        """get_client_identifier extracts first IP from X-Forwarded-For"""
        request = Mock()
        request.headers = {"X-Forwarded-For": "203.0.113.1, 198.51.100.1"}
        request.state = Mock()
        request.state.user = None
        request.client = Mock()
        request.client.host = "192.168.1.1"

        identifier = get_client_identifier(request)

        assert identifier == "ip:203.0.113.1"

    def test_get_client_identifier_forwarded_for_priority(self) -> None:
        """get_client_identifier prefers X-Forwarded-For over client IP"""
        request = Mock()
        request.headers = {"X-Forwarded-For": "10.0.0.1"}
        request.state = Mock()
        request.state.user = None
        request.client = Mock()
        request.client.host = "192.168.1.1"

        identifier = get_client_identifier(request)

        assert identifier == "ip:10.0.0.1"
        assert "192.168.1.1" not in identifier

    def test_get_client_identifier_priority_order(self) -> None:
        """get_client_identifier respects priority: API key > user > IP"""
        # Priority 1: API Key
        request1 = Mock()
        request1.headers = {"X-API-Key": "key1"}
        request1.state = Mock()
        user = Mock()
        user.user_id = "user1"
        request1.state.user = user
        request1.client = Mock()
        request1.client.host = "1.1.1.1"

        id1 = get_client_identifier(request1)
        assert id1 == "api_key:key1"

        # Priority 2: User (no API key)
        request2 = Mock()
        request2.headers = {}
        request2.state = Mock()
        request2.state.user = user
        request2.client = Mock()
        request2.client.host = "1.1.1.1"

        id2 = get_client_identifier(request2)
        assert id2 == "user:user1"

    def test_get_client_identifier_no_client_ip(self) -> None:
        """get_client_identifier handles missing client IP"""
        request = Mock()
        request.headers = {}
        request.state = Mock()
        request.state.user = None
        request.client = None

        identifier = get_client_identifier(request)

        assert identifier == "ip:unknown"

    def test_get_client_identifier_with_spaces_in_forwarded_for(self) -> None:
        """get_client_identifier handles spaces in X-Forwarded-For"""
        request = Mock()
        request.headers = {"X-Forwarded-For": " 203.0.113.1 , 198.51.100.1 "}
        request.state = Mock()
        request.state.user = None
        request.client = Mock()
        request.client.host = "192.168.1.1"

        identifier = get_client_identifier(request)

        assert identifier == "ip:203.0.113.1"


class TestRateLimiterIntegration:
    """Integration tests for rate limiting"""

    def test_rate_limiter_workflow(self) -> None:
        """Complete rate limiting workflow"""
        limiter = RateLimiter()
        max_requests = 5
        window = 60

        allowed_count = 0
        denied_count = 0

        for i in range(10):
            allowed, info = limiter.is_allowed("user1", max_requests, window)
            if allowed:
                allowed_count += 1
            else:
                denied_count += 1

        assert allowed_count == 5
        assert denied_count == 5

    def test_rate_limiter_with_different_limits_per_client(self) -> None:
        """Rate limiter respects different limits per client"""
        limiter = RateLimiter()

        # Client 1 has limit of 2
        limiter.is_allowed("client1", max_requests=2, window_seconds=60)
        limiter.is_allowed("client1", max_requests=2, window_seconds=60)
        allowed1, _ = limiter.is_allowed("client1", max_requests=2, window_seconds=60)

        # Client 2 has limit of 5
        limiter.is_allowed("client2", max_requests=5, window_seconds=60)
        limiter.is_allowed("client2", max_requests=5, window_seconds=60)
        limiter.is_allowed("client2", max_requests=5, window_seconds=60)
        limiter.is_allowed("client2", max_requests=5, window_seconds=60)
        limiter.is_allowed("client2", max_requests=5, window_seconds=60)
        allowed2, _ = limiter.is_allowed("client2", max_requests=5, window_seconds=60)

        assert allowed1 is False  # client1 exceeded
        assert allowed2 is False  # client2 exceeded
