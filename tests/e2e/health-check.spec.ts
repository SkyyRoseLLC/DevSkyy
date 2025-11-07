import { test, expect } from '@playwright/test';

/**
 * DevSkyy Platform - Health Check E2E Tests
 * Enterprise-grade end-to-end testing
 *
 * Truth Protocol Compliance: All 15 rules
 * Version: 1.0.0
 */

test.describe('API Health Checks', () => {
  test('should return healthy status from /api/v1/healthz', async ({ request }) => {
    const response = await request.get('/api/v1/healthz');

    expect(response.ok()).toBeTruthy();
    expect(response.status()).toBe(200);

    const body = await response.json();
    expect(body).toHaveProperty('status', 'healthy');
    expect(body).toHaveProperty('timestamp');
    expect(body).toHaveProperty('components');
  });

  test('should return Prometheus metrics from /metrics', async ({ request }) => {
    const response = await request.get('/metrics');

    expect(response.ok()).toBeTruthy();
    expect(response.status()).toBe(200);

    const text = await response.text();

    // Verify key metrics are present
    expect(text).toContain('devskyy_requests_total');
    expect(text).toContain('devskyy_request_duration_seconds');
    expect(text).toContain('devskyy_system_cpu_usage');
  });

  test('should have database connectivity', async ({ request }) => {
    const response = await request.get('/api/v1/healthz');
    const body = await response.json();

    expect(body.components).toHaveProperty('database');
    expect(body.components.database).toBe('healthy');
  });

  test('should have Redis connectivity', async ({ request }) => {
    const response = await request.get('/api/v1/healthz');
    const body = await response.json();

    expect(body.components).toHaveProperty('redis');
    expect(body.components.redis).toBe('healthy');
  });
});

test.describe('ML Engine Health Checks', () => {
  test('should verify sentiment analyzer health', async ({ request }) => {
    const response = await request.get('/api/v1/ml/health/sentiment');

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('status');
    expect(['healthy', 'degraded']).toContain(body.status);
  });

  test('should verify image generation health', async ({ request }) => {
    const response = await request.get('/api/v1/ml/health/image-generation');

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('status');
  });

  test('should verify text generation health', async ({ request }) => {
    const response = await request.get('/api/v1/ml/health/text-generation');

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('status');
  });

  test('should verify fashion trend predictor health', async ({ request }) => {
    const response = await request.get('/api/v1/ml/health/fashion-trends');

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('status');
  });
});

test.describe('Agent System Health Checks', () => {
  test('should verify executive orchestrator health', async ({ request }) => {
    const response = await request.get('/api/v1/agents/health/orchestrator');

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('status', 'healthy');
    expect(body).toHaveProperty('agents');
  });

  test('should list available agents', async ({ request }) => {
    const response = await request.get('/api/v1/agents/list');

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(Array.isArray(body.agents)).toBeTruthy();
    expect(body.agents.length).toBeGreaterThan(0);
  });
});

test.describe('Performance SLO Validation', () => {
  test('should respond to health check within 200ms', async ({ request }) => {
    const start = Date.now();
    const response = await request.get('/api/v1/healthz');
    const duration = Date.now() - start;

    expect(response.ok()).toBeTruthy();
    expect(duration).toBeLessThan(200); // P95 < 200ms SLO
  });

  test('should respond to metrics endpoint within 100ms', async ({ request }) => {
    const start = Date.now();
    const response = await request.get('/metrics');
    const duration = Date.now() - start;

    expect(response.ok()).toBeTruthy();
    expect(duration).toBeLessThan(100); // Prometheus scrape < 100ms SLO
  });
});
