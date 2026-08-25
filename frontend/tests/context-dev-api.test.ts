import { afterEach, describe, expect, it, vi } from 'vitest';

import { API_URL } from '@/lib/api/config';
import { contextDev } from '@/lib/api/endpoints/context-dev';

afterEach(() => {
  vi.unstubAllGlobals();
});

describe('Context.dev API client', () => {
  it('posts a bounded JSON Schema extraction request to the DevSkyy endpoint', async () => {
    const fetchMock = vi.fn().mockResolvedValue(
      new Response(JSON.stringify({ data: { how_it_works: 'Grounded workflow.' } }), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }),
    );
    vi.stubGlobal('fetch', fetchMock);

    const result = await contextDev.extract({
      url: 'https://higgsfield.com',
      schema: {
        type: 'object',
        properties: { how_it_works: { type: 'string' } },
        required: ['how_it_works'],
      },
      follow_subdomains: true,
      max_pages: 25,
      max_depth: 2,
      parse_pdfs: false,
    });

    expect(result).toEqual({ data: { how_it_works: 'Grounded workflow.' } });
    expect(fetchMock).toHaveBeenCalledWith(
      `${API_URL}/api/v1/context-dev/extractions`,
      expect.objectContaining({
        method: 'POST',
        body: JSON.stringify({
          url: 'https://higgsfield.com',
          schema: {
            type: 'object',
            properties: { how_it_works: { type: 'string' } },
            required: ['how_it_works'],
          },
          follow_subdomains: true,
          max_pages: 25,
          max_depth: 2,
          parse_pdfs: false,
        }),
      }),
    );
  });
});
