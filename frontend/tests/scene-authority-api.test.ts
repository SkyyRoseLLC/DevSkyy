import { beforeEach, describe, expect, it, vi } from 'vitest';
import { NextRequest } from 'next/server';

const { getServerSessionMock } = vi.hoisted(() => ({
  getServerSessionMock: vi.fn(),
}));

vi.mock('next-auth', () => ({
  getServerSession: getServerSessionMock,
}));

vi.mock('@/lib/auth', () => ({
  authOptions: {},
}));

import { GET as getSceneAuthorityList } from '@/app/api/scene-authority/route';
import { GET as getSceneAuthorityDetail } from '@/app/api/scene-authority/[sceneId]/route';

describe('scene-authority API', () => {
  beforeEach(() => {
    getServerSessionMock.mockReset();
  });

  it('fails closed when the operator has no authenticated session', async () => {
    getServerSessionMock.mockResolvedValue(null);

    const response = await getSceneAuthorityList(new NextRequest('http://localhost/api/scene-authority'), undefined);

    expect(response.status).toBe(401);
    await expect(response.json()).resolves.toEqual({
      success: false,
      error: 'Unauthorized',
    });
  });

  it('returns the sanitized projection to an authenticated operator', async () => {
    getServerSessionMock.mockResolvedValue({ user: { email: 'operator@example.test' } });

    const response = await getSceneAuthorityList(new NextRequest('http://localhost/api/scene-authority'), undefined);
    const body = await response.json();
    const serialized = JSON.stringify(body);

    expect(response.status).toBe(200);
    expect(body.success).toBe(true);
    expect(body.data.schema).toBe('skyyrose.scene-authority-projection/1');
    expect(body.data.scenes.length).toBeGreaterThanOrEqual(5);
    expect(serialized).not.toContain('/Users/');
    expect(serialized).not.toContain('/private/');
    expect(serialized).not.toContain('/var/');
    expect(serialized).not.toContain('/tmp/');
    expect(serialized).not.toContain('file://');
    expect(serialized).not.toContain('Comfy/');
    expect(serialized).not.toContain('tasks/');
  });

  it('returns the Love Hurts identity and independent rejection gates', async () => {
    getServerSessionMock.mockResolvedValue({ user: { email: 'operator@example.test' } });

    const response = await getSceneAuthorityDetail(
      new NextRequest('http://localhost/api/scene-authority/lh-commerce-2'),
      { params: Promise.resolve({ sceneId: 'lh-commerce-2' }) }
    );
    const body = await response.json();

    expect(response.status).toBe(200);
    expect(body.data.sceneId).toBe('LH-COMMERCE-2');
    expect(body.data.identity.id).toBe('LH-MODEL-01');
    expect(body.data.gates.identity).toBe('FOUNDER_APPROVED_IDENTITY_ONLY');
    expect(body.data.gates.productFidelity).toBe('HARD_FAIL');
    expect(body.data.gates.sceneInputEligible).toBe(false);
    expect(body.data.gates.paidExecutionReady).toBe(false);
  });

  it('returns 404 instead of falling back for an unknown scene', async () => {
    getServerSessionMock.mockResolvedValue({ user: { email: 'operator@example.test' } });

    const response = await getSceneAuthorityDetail(
      new NextRequest('http://localhost/api/scene-authority/not-a-scene'),
      { params: Promise.resolve({ sceneId: 'not-a-scene' }) }
    );

    expect(response.status).toBe(404);
    await expect(response.json()).resolves.toEqual({
      success: false,
      error: 'Scene authority not found',
    });
  });
});
