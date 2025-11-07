import { test, expect } from '@playwright/test';

/**
 * DevSkyy Platform - GDPR Compliance E2E Tests
 * Tests for GDPR Articles 15 (Data Export) and 17 (Data Deletion)
 *
 * Truth Protocol Compliance: All 15 rules
 * Version: 1.0.0
 */

test.describe('GDPR Data Export (Article 15)', () => {
  test('should export user data', async ({ request }) => {
    const response = await request.post('/api/v1/gdpr/data-export', {
      data: {
        user_id: 'test-user-001'
      }
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('export_id');
    expect(body).toHaveProperty('status');
    expect(body).toHaveProperty('data');

    // Verify data structure
    if (body.data) {
      expect(body.data).toHaveProperty('profile');
      expect(body.data).toHaveProperty('export_metadata');
      expect(body.data.export_metadata).toHaveProperty('timestamp');
      expect(body.data.export_metadata).toHaveProperty('format', 'JSON');
    }
  });

  test('should include all required data categories in export', async ({ request }) => {
    const response = await request.post('/api/v1/gdpr/data-export', {
      data: {
        user_id: 'test-user-001'
      }
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();

    if (body.data) {
      // Verify mandatory data categories
      const expectedCategories = [
        'profile',
        'orders',
        'preferences',
        'sessions',
        'analytics',
        'consent_records'
      ];

      expectedCategories.forEach(category => {
        expect(body.data).toHaveProperty(category);
      });
    }
  });

  test('should complete data export within 200ms', async ({ request }) => {
    const start = Date.now();

    await request.post('/api/v1/gdpr/data-export', {
      data: {
        user_id: 'test-user-001'
      }
    });

    const duration = Date.now() - start;
    expect(duration).toBeLessThan(200); // P95 < 200ms SLO
  });
});

test.describe('GDPR Data Deletion (Article 17)', () => {
  test('should delete user data', async ({ request }) => {
    const response = await request.post('/api/v1/gdpr/data-delete', {
      data: {
        user_id: 'test-user-to-delete',
        reason: 'User request'
      }
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('deletion_id');
    expect(body).toHaveProperty('status');
    expect(body).toHaveProperty('items_deleted');
    expect(body.items_deleted).toBeGreaterThanOrEqual(0);
  });

  test('should respect legal retention requirements', async ({ request }) => {
    const response = await request.post('/api/v1/gdpr/data-delete', {
      data: {
        user_id: 'test-user-with-orders',
        reason: 'User request'
      }
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('legal_retention_notice');

    // Orders should be anonymized, not deleted (7-year tax requirement)
    if (body.legal_retention_notice) {
      expect(body.legal_retention_notice).toContain('anonymized');
    }
  });
});

test.describe('GDPR Consent Management', () => {
  test('should update user consent preferences', async ({ request }) => {
    const response = await request.post('/api/v1/gdpr/consent', {
      data: {
        user_id: 'test-user-001',
        consent_type: 'marketing',
        granted: true
      }
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('consent_id');
    expect(body).toHaveProperty('timestamp');
  });

  test('should retrieve user consents', async ({ request }) => {
    const response = await request.get('/api/v1/gdpr/consents/test-user-001');

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('user_id', 'test-user-001');
    expect(body).toHaveProperty('consents');
    expect(Array.isArray(body.consents)).toBeTruthy();
  });
});

test.describe('GDPR Data Retention', () => {
  test('should return retention policies', async ({ request }) => {
    const response = await request.get('/api/v1/gdpr/retention-policies');

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('policies');
    expect(Array.isArray(body.policies)).toBeTruthy();

    // Verify policy structure
    if (body.policies.length > 0) {
      const policy = body.policies[0];
      expect(policy).toHaveProperty('data_type');
      expect(policy).toHaveProperty('retention_days');
      expect(policy).toHaveProperty('legal_basis');
    }
  });

  test('should return user-specific retention status', async ({ request }) => {
    const response = await request.get('/api/v1/gdpr/retention-status/test-user-001');

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('user_id', 'test-user-001');
    expect(body).toHaveProperty('retention_status');
    expect(body).toHaveProperty('overall_summary');
  });
});

test.describe('GDPR Audit Logs', () => {
  test('should retrieve GDPR audit logs', async ({ request }) => {
    const response = await request.get('/api/v1/gdpr/audit-logs', {
      params: {
        limit: '10',
        offset: '0'
      }
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();
    expect(body).toHaveProperty('logs');
    expect(Array.isArray(body.logs)).toBeTruthy();
    expect(body).toHaveProperty('total');
    expect(body).toHaveProperty('limit', 10);
    expect(body).toHaveProperty('offset', 0);
  });

  test('should include required audit fields', async ({ request }) => {
    const response = await request.get('/api/v1/gdpr/audit-logs', {
      params: { limit: '1' }
    });

    expect(response.ok()).toBeTruthy();

    const body = await response.json();

    if (body.logs.length > 0) {
      const log = body.logs[0];
      expect(log).toHaveProperty('timestamp');
      expect(log).toHaveProperty('action');
      expect(log).toHaveProperty('user_id');
      expect(log).toHaveProperty('actor_id');
      expect(log).toHaveProperty('ip_address');
    }
  });
});

test.describe('GDPR Error Handling', () => {
  test('should return 404 for non-existent user', async ({ request }) => {
    const response = await request.post('/api/v1/gdpr/data-export', {
      data: {
        user_id: 'non-existent-user-12345'
      }
    });

    expect(response.status()).toBe(404);
  });

  test('should validate required fields', async ({ request }) => {
    const response = await request.post('/api/v1/gdpr/data-delete', {
      data: {
        // Missing user_id and reason
      }
    });

    expect(response.status()).toBe(422); // Validation error
  });
});
