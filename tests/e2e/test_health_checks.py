"""
DevSkyy Platform - Health Check E2E Tests (pytest)
Enterprise-grade end-to-end testing with pytest-playwright

Truth Protocol Compliance: All 15 rules
Version: 1.0.0
"""

import pytest
import time


class TestAPIHealthChecks:
    """API health check tests"""

    def test_healthz_endpoint_returns_healthy_status(self, api_request):
        """Should return healthy status from /api/v1/healthz"""
        response = api_request.get("/api/v1/healthz")

        assert response.ok
        assert response.status == 200

        body = response.json()
        assert "status" in body
        assert body["status"] == "healthy"
        assert "timestamp" in body
        assert "components" in body

    def test_prometheus_metrics_endpoint(self, api_request):
        """Should return Prometheus metrics from /metrics"""
        response = api_request.get("/metrics")

        assert response.ok
        assert response.status == 200

        text = response.text()

        # Verify key metrics are present
        assert "devskyy_requests_total" in text
        assert "devskyy_request_duration_seconds" in text
        assert "devskyy_system_cpu_usage" in text

    def test_database_connectivity(self, api_request):
        """Should have database connectivity"""
        response = api_request.get("/api/v1/healthz")
        body = response.json()

        assert "components" in body
        assert "database" in body["components"]
        assert body["components"]["database"] == "healthy"

    def test_redis_connectivity(self, api_request):
        """Should have Redis connectivity"""
        response = api_request.get("/api/v1/healthz")
        body = response.json()

        assert "components" in body
        assert "redis" in body["components"]
        assert body["components"]["redis"] == "healthy"


class TestMLEngineHealthChecks:
    """ML engine health check tests"""

    def test_sentiment_analyzer_health(self, api_request):
        """Should verify sentiment analyzer health"""
        response = api_request.get("/api/v1/ml/health/sentiment")

        assert response.ok

        body = response.json()
        assert "status" in body
        assert body["status"] in ["healthy", "degraded"]

    def test_image_generation_health(self, api_request):
        """Should verify image generation health"""
        response = api_request.get("/api/v1/ml/health/image-generation")

        assert response.ok

        body = response.json()
        assert "status" in body

    def test_text_generation_health(self, api_request):
        """Should verify text generation health"""
        response = api_request.get("/api/v1/ml/health/text-generation")

        assert response.ok

        body = response.json()
        assert "status" in body

    def test_fashion_trend_predictor_health(self, api_request):
        """Should verify fashion trend predictor health"""
        response = api_request.get("/api/v1/ml/health/fashion-trends")

        assert response.ok

        body = response.json()
        assert "status" in body


class TestAgentSystemHealthChecks:
    """Agent system health check tests"""

    def test_executive_orchestrator_health(self, api_request):
        """Should verify executive orchestrator health"""
        response = api_request.get("/api/v1/agents/health/orchestrator")

        assert response.ok

        body = response.json()
        assert "status" in body
        assert body["status"] == "healthy"
        assert "agents" in body

    def test_list_available_agents(self, api_request):
        """Should list available agents"""
        response = api_request.get("/api/v1/agents/list")

        assert response.ok

        body = response.json()
        assert "agents" in body
        assert isinstance(body["agents"], list)
        assert len(body["agents"]) > 0


class TestPerformanceSLOValidation:
    """Performance SLO validation tests"""

    def test_health_check_response_time_slo(self, api_request):
        """Should respond to health check within 200ms (P95 SLO)"""
        start = time.time()
        response = api_request.get("/api/v1/healthz")
        duration_ms = (time.time() - start) * 1000

        assert response.ok
        assert duration_ms < 200, f"Response took {duration_ms:.2f}ms, exceeds 200ms SLO"

    def test_metrics_endpoint_response_time_slo(self, api_request):
        """Should respond to metrics endpoint within 100ms (Prometheus scrape SLO)"""
        start = time.time()
        response = api_request.get("/metrics")
        duration_ms = (time.time() - start) * 1000

        assert response.ok
        assert duration_ms < 100, f"Metrics scrape took {duration_ms:.2f}ms, exceeds 100ms SLO"

    @pytest.mark.parametrize("execution_number", range(10))
    def test_consistent_performance_across_multiple_requests(self, api_request, execution_number):
        """Should maintain consistent performance across multiple requests"""
        start = time.time()
        response = api_request.get("/api/v1/healthz")
        duration_ms = (time.time() - start) * 1000

        assert response.ok
        assert duration_ms < 300, f"Request {execution_number + 1} took {duration_ms:.2f}ms"
