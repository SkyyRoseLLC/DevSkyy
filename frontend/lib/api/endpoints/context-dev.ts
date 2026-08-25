import { API_URL } from '../config';
import { fetchWithTimeout, getAuthHeaders, handleResponse } from '../client';
import { ContextDevExtractionResponseSchema } from '../schemas';
import type { ContextDevExtractionRequest, ContextDevExtractionResponse } from '../types';

const BASE = '/api/v1/context-dev';
// The backend can crawl up to 25 pages. Keep the browser request alive longer
// than the general 30s API default, while still letting the operator recover.
const EXTRACTION_TIMEOUT_MS = 180_000;

export const contextDev = {
  /** POST /context-dev/extractions — authenticated, fact-checked web extraction. */
  extract: async (request: ContextDevExtractionRequest): Promise<ContextDevExtractionResponse> => {
    const response = await fetchWithTimeout(
      `${API_URL}${BASE}/extractions`,
      {
        method: 'POST',
        headers: await getAuthHeaders(),
        body: JSON.stringify(request),
      },
      EXTRACTION_TIMEOUT_MS,
    );
    return handleResponse(response, ContextDevExtractionResponseSchema);
  },
};
