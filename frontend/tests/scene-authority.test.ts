import { execFileSync } from 'node:child_process';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { getSceneAuthorities, getSceneAuthority, getSceneAuthorityProjection } from '@/lib/scene-authority';

describe('scene-authority projection', () => {
  it('is fresh against the repository OODA ledger', () => {
    expect(() =>
      execFileSync(process.execPath, ['scripts/sync-scene-authority.mjs', '--check'], {
        cwd: join(__dirname, '..'),
        stdio: 'pipe',
      })
    ).not.toThrow();
  });

  it('contains only sanitized deploy-safe data', () => {
    const serialized = JSON.stringify(getSceneAuthorityProjection());

    expect(serialized).not.toContain('/Users/');
    expect(serialized).not.toContain('/private/');
    expect(serialized).not.toContain('/var/');
    expect(serialized).not.toContain('/tmp/');
    expect(serialized).not.toContain('file://');
    expect(serialized).not.toContain('Comfy/');
    expect(serialized).not.toContain('tasks/');
    expect(serialized).not.toContain('.env');
  });

  it('keeps the Love Hurts approval boundaries independent', () => {
    const scene = getSceneAuthority('lh-commerce-2');

    expect(scene).toBeDefined();
    expect(scene?.identity).toEqual({
      id: 'LH-MODEL-01',
      referenceAssetCount: 2,
      fullCandidateVerdict: 'REJECT',
    });
    expect(scene?.gates.identity).toBe('FOUNDER_APPROVED_IDENTITY_ONLY');
    expect(scene?.gates.productFidelity).toBe('HARD_FAIL');
    expect(scene?.gates.sceneInputEligible).toBe(false);
    expect(scene?.gates.paidAuthorization).toBe('NOT_AUTHORIZED');
    expect(scene?.gates.paidExecutionReady).toBe(false);
    expect(scene?.gates.runtimeWiring).toBe('NOT_AUTHORIZED');
    expect(scene?.gates.deployment).toBe('NOT_AUTHORIZED');
    expect(scene?.gates.promotionRequirement).toBe('FOUNDER_APPROVED_VISUAL');
  });

  it('returns all scenes and fails closed for unknown IDs', () => {
    expect(getSceneAuthorities().length).toBeGreaterThanOrEqual(5);
    expect(getSceneAuthority('missing-scene')).toBeUndefined();
  });
});
