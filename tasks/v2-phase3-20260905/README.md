# Phase 3A local verification evidence

Scope: design-system and global-shell candidate only. See REPORT.md for final source,
results, limitations and the next authorization boundary. This directory contains
review/evidence records, not deployment instructions.

## Isolation

The preview is http://127.0.0.1:18303. WordPress 7.1 and WooCommerce 11.1.0 run
against a disposable MariaDB 11.4 container, bound at 127.0.0.1:18306. The fixture
contains 33 synthetic Woo products and 185 variations mapped to certified catalog
sizes/prices; IDs are local. No live customer records, orders or provider credentials
were copied. Product attachment copies retain existing source bytes. Mail, outbound
HTTP, cron, webhooks and available payment gateways are disabled in the fixture.

Core, plugins, database configuration, bootstrap scripts, dependency installs, logs
and screenshots remain under ignored `.artifacts/v2-phase3-20260905/`. Credentials
are not part of committed evidence. The exact theme source is linked into this local
runtime. This fixture proves runtime behavior under its configuration; it does not
certify live plugins, physical inventory, shipping, authenticated customers or payments.

## Commands

Use the certified toolchain: Node 22.23.2, npm 10.9.8, Python 3.12.12 with pinned
Pillow, and PHP CLI. From the repository, activate the certified Python environment,
then run:

```sh
npm --prefix wordpress-theme/skyyrose-flagship-2 run build
npm --prefix wordpress-theme/skyyrose-flagship-2 run verify
python -m unittest discover -s tools/v2-source-certification -p 'test_*.py'
npm --prefix wordpress-theme/skyyrose-flagship-2 run package:theme
```

The browser scripts require the already provisioned isolated fixture. Their default
module resolver points to `.artifacts/v2-phase3-20260905/qa/package.json`, containing
Playwright 1.58.2 and @axe-core/playwright 4.11.1; these are verification dependencies,
not storefront dependencies. Chromium must be installed for that Playwright version.

```sh
node tools/v2-runtime/test-global-shell-browser.mjs
node tools/v2-runtime/test-shell-recovery-browser.mjs
node tools/v2-runtime/measure-shell-performance.mjs candidate
```

`measure-shell-performance.mjs baseline` is valid only while the isolated preview
serves the unchanged certified Phase 2 theme. The coordinator records and restores
that temporary theme link. The sampler blocks other origins and uses identical
three-second post-load windows with three cold contexts per route and viewport.
Its measurements are synthetic localhost observations, not field Core Web Vitals.

The main browser gate fails on collected Axe, console/page, HTTP, broken-image and
unexpected request failures. Navigation-aborted requests remain recorded separately.
Intentionally intercepted external requests are recorded by their actual interception
identity. The existing About-page YouTube embed is blocked by local isolation and
its remote behavior remains unverified. The recovery test intentionally fulfills one
local native cart-removal AJAX request with HTTP 503 and then verifies Woo's own
nonce-bearing GET recovery; this expected fault is separate from the clean-path gate.

## Evidence convention

JSON receipts and screenshot hashes identify what was observed. Desktop/mobile
screenshots are locally reviewable files; they are not founder approval. The task
ledger is append-only. Independent source and visual decisions retain their own
scope. Runtime source, clean package, visual review, founder approval and deployment
are separate claims.
