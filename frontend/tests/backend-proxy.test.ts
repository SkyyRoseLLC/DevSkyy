import { describe, expect, it } from 'vitest';

import { buildBackendUrl, isAllowedBackendPath } from '@/lib/backend-proxy';

describe('dashboard backend proxy policy', () => {
  it('permits the dashboard API namespace and health probes only', () => {
    expect(isAllowedBackendPath('api/v1/agents')).toBe(true);
    expect(isAllowedBackendPath('api/v1/tasks/task-1')).toBe(true);
    expect(isAllowedBackendPath('health')).toBe(true);
    expect(isAllowedBackendPath('ready')).toBe(true);
    expect(isAllowedBackendPath('api/v2/usage')).toBe(true);
    expect(isAllowedBackendPath('api/v2/creative/operations')).toBe(true);
    expect(isAllowedBackendPath('api/v2/creative/operations/op-1')).toBe(true);
    expect(isAllowedBackendPath('api/v2/characters/rosie')).toBe(true);
    expect(isAllowedBackendPath('api/v2/anything-else')).toBe(false);
    expect(isAllowedBackendPath('mcp')).toBe(false);
    expect(isAllowedBackendPath('../api/v1/agents')).toBe(false);
    expect(isAllowedBackendPath('api/v1/../mcp')).toBe(false);
    expect(isAllowedBackendPath('api/v1/%2e%2e/mcp')).toBe(false);
    expect(isAllowedBackendPath('api//v1/agents')).toBe(false);
  });

  it('uses the vetted local backend default outside production', () => {
    const previous = process.env.NEXT_PUBLIC_API_URL;
    delete process.env.NEXT_PUBLIC_API_URL;

    expect(buildBackendUrl('api/v1/agents', '')?.toString()).toBe('http://localhost:8000/api/v1/agents');

    if (previous === undefined) delete process.env.NEXT_PUBLIC_API_URL;
    else process.env.NEXT_PUBLIC_API_URL = previous;
  });

  it('rejects backend URLs with a path prefix instead of silently stripping it', () => {
    const previous = process.env.NEXT_PUBLIC_API_URL;
    process.env.NEXT_PUBLIC_API_URL = 'https://api.example.test/backend';

    expect(buildBackendUrl('api/v1/agents', '')).toBeNull();

    if (previous === undefined) delete process.env.NEXT_PUBLIC_API_URL;
    else process.env.NEXT_PUBLIC_API_URL = previous;
  });
});
