import { existsSync } from 'node:fs';
import { join } from 'node:path';

import { describe, expect, it } from 'vitest';

import { CONSOLE_NAV_ITEMS } from '@/components/console/nav-config';

describe('console navigation', () => {
  it('points every console item to an implemented admin page', () => {
    for (const item of CONSOLE_NAV_ITEMS) {
      const segments = item.href
        .replace(/^\/admin\/?/, '')
        .split('/')
        .filter(Boolean);
      const page = join(process.cwd(), 'app', 'admin', ...segments, 'page.tsx');
      expect(existsSync(page), `${item.label} points to ${item.href}, which has no page.tsx`).toBe(true);
    }
  });
});
