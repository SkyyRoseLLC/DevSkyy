#!/usr/bin/env node
/**
 * Captures desktop and mobile visual-QA evidence without OpenWolf's
 * unpatched Puppeteer dependency chain.
 *
 * Usage:
 *   node scripts/designqc-playwright.mjs --url https://example.test
 *   node scripts/designqc-playwright.mjs --url http://localhost:8080 --output .wolf/designqc-captures
 */

import { createRequire } from 'node:module';
import { mkdirSync } from 'node:fs';
import path from 'node:path';

const defaults = {
  output: path.resolve('.wolf/designqc-captures'),
  url: 'http://localhost:8080',
};

const args = process.argv.slice(2);
for (let index = 0; index < args.length; index += 1) {
  if (args[index] === '--url' && args[index + 1]) defaults.url = args[++index];
  if (args[index] === '--output' && args[index + 1]) defaults.output = path.resolve(args[++index]);
}

let chromium;
try {
  const frontendRequire = createRequire(new URL('../frontend/package.json', import.meta.url));
  ({ chromium } = frontendRequire('playwright'));
} catch (error) {
  console.error(`[designqc] Playwright is unavailable: ${error.message}`);
  console.error('[designqc] Install the frontend dependencies and browsers before retrying.');
  process.exit(3);
}

mkdirSync(defaults.output, { recursive: true });
const browser = await chromium.launch();
const failures = [];

for (const viewport of [
  { name: 'desktop', width: 1440, height: 1100 },
  { name: 'mobile', width: 390, height: 844, isMobile: true },
]) {
  const { name, width, height, isMobile = false } = viewport;
  const context = await browser.newContext({
    viewport: { width, height },
    isMobile,
    reducedMotion: 'reduce',
  });
  const page = await context.newPage();
  const pageErrors = [];
  page.on('pageerror', error => pageErrors.push(error.message));

  try {
    await page.goto(defaults.url, { waitUntil: 'networkidle', timeout: 60_000 });
    await page.screenshot({
      fullPage: true,
      path: path.join(defaults.output, `${name}.png`),
    });
    if (pageErrors.length) {
      failures.push(`${name}: ${pageErrors.join(' | ')}`);
    }
  } catch (error) {
    failures.push(`${name}: ${error.message}`);
  } finally {
    await context.close();
  }
}

await browser.close();

if (failures.length) {
  console.error(`[designqc] failed for ${defaults.url}\n${failures.join('\n')}`);
  process.exit(1);
}

console.log(`[designqc] desktop and mobile captures saved to ${defaults.output}`);
