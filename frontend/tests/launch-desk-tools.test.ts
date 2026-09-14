import { describe, expect, it } from 'vitest';

import {
  assessLaunchReadiness,
  buildOwnerChecklists,
  draftLaunchCopy,
  extractLaunchTasks,
} from '@/app/api/launch-desk/_lib/tools';

const completeBrief = {
  productBrief: 'Launch a developer API with monitoring, support, rollback, and integration guides.',
  audience: 'Platform engineering leads evaluating release automation.',
  launchDate: '2026-09-15',
  constraints: 'Two engineers; security review required before release.',
  availableAssets: 'Demo, API reference, architecture diagram, and beta feedback.',
};

describe('Launch Desk tools', () => {
  it('extracts prioritized cross-functional work', () => {
    const tasks = extractLaunchTasks(completeBrief);
    expect(tasks.some((task) => task.priority === 'P0')).toBe(true);
    expect(tasks.some((task) => task.owner === 'Developer Experience')).toBe(true);
  });

  it('fails readiness closed when launch inputs are absent', () => {
    const checks = assessLaunchReadiness({
      ...completeBrief,
      launchDate: '',
      constraints: '',
      availableAssets: '',
    });
    expect(checks.filter((check) => check.status === 'missing').length).toBeGreaterThanOrEqual(3);
  });

  it('groups every extracted task under an owner', () => {
    const tasks = extractLaunchTasks(completeBrief);
    const checklists = buildOwnerChecklists(tasks);
    const itemCount = checklists.reduce((total, checklist) => total + checklist.items.length, 0);
    expect(itemCount).toBe(tasks.length);
  });

  it('drafts all required channels without inventing metrics', () => {
    const drafts = draftLaunchCopy({
      productName: 'Launch Desk',
      promise: 'turn a rough brief into an owned release plan',
      audience: 'engineering teams',
    });
    expect(drafts.map((draft) => draft.channel)).toEqual([
      'email',
      'product-update',
      'social',
      'internal',
    ]);
    expect(drafts.map((draft) => draft.copy).join(' ')).not.toMatch(/\d+%/);
  });
});
