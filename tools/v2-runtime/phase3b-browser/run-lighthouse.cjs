const { spawn } = require('node:child_process');
const path = require('node:path');
const assert = require('node:assert/strict');
const { requireQa, base, out, validateLabel } = require('./runtime.cjs');
const { createOriginProxy } = require('./origin-proxy.cjs');
const { beginRun } = require('./run-evidence.cjs');
if (process.argv[2]) validateLabel(process.argv[2]);
const cases = [
  ['home-mobile', '/', false],
  ['home-desktop', '/', true],
  ['pdp-mobile', '/product/sg-005/', false],
  ['pdp-desktop', '/product/sg-005/', true],
  ['shop-mobile', '/shop/', false],
];
const prefix = 'lighthouse-' + (process.argv[2] ? process.argv[2] + '-' : '');
const run = beginRun(path.join(out, prefix + 'run.json'));
(async () => {
  if (process.env.V2_LH_CASE && !cases.some(([name]) => name === process.env.V2_LH_CASE)) {
    throw new Error('Unknown V2_LH_CASE: ' + process.env.V2_LH_CASE);
  }
  const chrome = requireQa('playwright').chromium.executablePath();
  const cli = requireQa.resolve('lighthouse/cli/index.js');
  const proxy = await createOriginProxy(base);
  const completed = [];
  const proxyTraffic = [];
  try {
    // Await each process: the proxy stays responsive, and cases stay serial.
    for (const [name, route, desktop] of cases) {
      if (process.env.V2_LH_CASE && name !== process.env.V2_LH_CASE) continue;
      const allowedBefore = proxy.allowedRequests;
      const chromeFlags = [
        '--headless',
        '--no-sandbox',
        '--disable-gpu',
        '--disable-quic',
        '--host-resolver-rules="MAP * ~NOTFOUND, EXCLUDE 127.0.0.1"',
        '--proxy-server=' + proxy.url,
        '--proxy-bypass-list=<-loopback>',
      ].join(' ');
      const args = [
        cli,
        base + route,
        '--quiet',
        '--output=json',
        '--output=html',
        '--output-path=' + path.join(out, prefix + name),
        '--only-categories=performance,accessibility,best-practices,seo',
        '--blocked-url-patterns=https://*',
        '--chrome-flags=' + chromeFlags,
        ...(desktop ? ['--preset=desktop'] : []),
        '--screenEmulation.width=' + (desktop ? 1440 : 390),
        '--screenEmulation.height=' + (desktop ? 1000 : 844),
        '--screenEmulation.deviceScaleFactor=1',
        '--screenEmulation.mobile=' + (desktop ? 'false' : 'true'),
      ];
      await new Promise((resolve, reject) => {
        const child = spawn(process.execPath, args, {
          stdio: 'inherit',
          env: { ...process.env, CHROME_PATH: chrome },
          timeout: 180000,
        });
        child.once('error', reject);
        child.once('close', (status, signal) =>
          status === 0 ? resolve() : reject(new Error(name + ' Lighthouse failed: ' + (signal || status)))
        );
      });
      const allowedRequests = proxy.allowedRequests - allowedBefore;
      assert(allowedRequests > 0, name + ' produced no allowed proxy traffic; Chromium proxy use is unverified.');
      proxyTraffic.push({ name, allowedRequests });
      completed.push(name);
      console.log(name + ' complete');
    }
  } finally {
    await proxy.close();
  }
  run.pass({
    completed,
    network: {
      policy: 'HTTP requests restricted to exact local fixture origin',
      origin: base,
      allowedRequests: proxy.allowedRequests,
      cases: proxyTraffic,
      blocked: proxy.blocked,
    },
  });
})().catch(error => {
  run.fail(error);
  console.error(error);
  process.exitCode = 1;
});
