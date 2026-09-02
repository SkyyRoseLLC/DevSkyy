import { beforeEach, describe, expect, it, vi } from 'vitest';

const mocks = vi.hoisted(() => ({
  fetchWithTimeout: vi.fn(),
  getAuthHeaders: vi.fn(),
}));

vi.mock('@/lib/api/client', () => ({
  fetchWithTimeout: mocks.fetchWithTimeout,
  getAuthHeaders: mocks.getAuthHeaders,
}));

vi.mock('@/lib/api/config', () => ({ API_URL: 'https://api.example.test' }));

import { eliteStudioClient } from '@/lib/elite-studio-client';

describe('Elite Studio client contracts', () => {
  beforeEach(() => {
    mocks.fetchWithTimeout.mockReset();
    mocks.getAuthHeaders.mockReset();
    mocks.getAuthHeaders.mockResolvedValue({ Accept: 'application/json' });
    mocks.fetchWithTimeout.mockResolvedValue(
      new Response(JSON.stringify({ operations: [], total: 0, page: 2, page_size: 7 }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })
    );
  });

  it('uses the v2 page_size parameter when listing operations', async () => {
    await eliteStudioClient.listOperations({ page: 2, limit: 7, status: 'queued' });

    expect(mocks.fetchWithTimeout).toHaveBeenCalledWith(
      'https://api.example.test/api/v2/creative/operations?status=queued&page=2&page_size=7',
      expect.any(Object)
    );
  });
});
