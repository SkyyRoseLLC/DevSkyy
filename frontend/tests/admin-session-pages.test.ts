import { describe, expect, it } from 'vitest';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';

const authenticatedPages = ['page.tsx', 'customers/page.tsx', 'orders/page.tsx', 'products/page.tsx'];

describe('authenticated console pages', () => {
  it.each(authenticatedPages)('renders %s for the current session request', relativePath => {
    const source = readFileSync(resolve(process.cwd(), 'app/admin', relativePath), 'utf8');
    const pageFunction = source.match(/export default async function [^(]+\(\) \{([\s\S]*?)\n\}/)?.[1];

    expect(source).toContain("import { connection } from 'next/server'");
    expect(pageFunction).toBeDefined();
    expect(pageFunction!.indexOf('await connection()')).toBeLessThan(pageFunction!.indexOf('getAdminSession()'));
  });
});
