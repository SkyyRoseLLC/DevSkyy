# Consolidated staging platform delivery — B01–B12

**Status: NEEDS_MORE_WORK.** The platform probes are complete for the newly deployed candidate. This is not a sitewide or performance PASS.

Candidate ZIP SHA-256: `a4ec431146f31d036d5547137d3274a693ccd3d072be6089126593f642d14f4b` — 536 runtime files.

All requests were gated on root-confirmed filesystem and current-candidate browser parity. Main probes completed at 2026-09-07 01:55:30 UTC; the supplemental pass then finished before root was released to begin B13. There was no concurrent B13/browser workload during these probes.

**750 read-only GET requests:** 639 encoding checks; 35 initial route samples; 35 final route samples; 13 full binary objects; nine ranges; three conditionals; 16 observed Ask Skyy dependency requests.

## B01–B12 results

| Check | Result | Scope and evidence |
|---|---|---|
| B01 | PASS | 64 static URLs were stable across three interleaved Brotli / identity / gzip cycles. Five observed Skyy code URLs were also stable across their three encodings. Dynamic HTML was not required to be byte-identical. |
| B02 | PASS | No transport, decompression, or negotiated encoding failures. Checkout 302 responses are excluded from HTML-document compression assertions. Already compressed media need not be compressed again. |
| B03 | PASS | All sampled 200 HTML/code responses exposed Vary: Accept-Encoding and consistent decoded static representations on the exact observed URLs. |
| B04 | PASS — SAMPLED | Three conditional requests returned 304 with empty bodies. Cache-Control and exposed edge labels are recorded below; these samples do not certify every CDN location. |
| B05 | ENVIRONMENT-BOUND | Current fingerprinted assets match the package, but this run did not repeat a historical old/new key comparison. No universal immutable policy or cache-key isolation claim. |
| B06 | FAIL — STRICT MIME | Four WOFF2 files use application/font-woff2 instead of font/woff2. The GLB uses application/octet-stream instead of model/gltf-binary. All sampled binary signatures and HTML/CSS/JS MIME checks pass. |
| B07 | PASS | Nine range requests across the approved MP4, hero WebM, and new rotating-mark WebM passed exact slice/status/Content-Range checks. |
| B08 | PASS — SAMPLED | 38 untransformed body comparisons across 19 unique URLs match the authorized package. No failed request or package mismatch. Twenty-six optimized bundle URLs were tested for representation consistency, not equated to individual source files. |
| B09 | ENVIRONMENT-BOUND | Anonymous cache policy was observed; authenticated-versus-anonymous isolation needs an existing authorized fixture and was not tested. |
| B10 | ENVIRONMENT-BOUND | No authorized populated-A / empty-B cart fixture was supplied. No cart or customer state was created. |
| B11 | ENVIRONMENT-BOUND | Private, nonce-bound and personalized endpoint isolation was not fabricated. No orders, payments, or state-changing requests were submitted. |
| B12 | FAIL — DIAGNOSTIC LATENCY | Final Cart samples have a 1.847-second median and 4.451-second maximum TTFB. Checkout redirect samples include two responses above 800 ms. Exact PHP/platform/network attribution remains unknown. |

## Actual TTFB through the staging stack

Five final sequential fresh curl processes per route, no cookie jar or cache-busting query/header. Values include DNS, connection, TLS, network, CDN and server time. The initial 35 observations remain separate and are not pooled into these final medians.

| Route | Median ms | Min–max ms | Above 800 ms | HTTP | Exposed edge labels |
|---|---:|---:|---:|---|---|
| home | 78.5 | 68.7–231.5 | 0/5 | 200 | 1.sjc _atomic_bur STALE; 1.sjc _atomic_bur UPDATING |
| shop | 75.4 | 58.9–133.4 | 0/5 | 200 | 3.sjc _atomic_bur HIT |
| signature | 85.6 | 68.9–144.3 | 0/5 | 200 | 3.sjc _atomic_bur HIT |
| pdp | 76.1 | 62.8–105.6 | 0/5 | 200 | 2.sjc _atomic_bur STALE; 2.sjc _atomic_bur UPDATING |
| cart | 1846.6 | 115.2–4450.7 | 3/5 | 200 | 3.sjc _atomic_bur BYPASS |
| checkout | 153.5 | 94.5–1026.4 | 2/5 | 302 | 4.sjc _atomic_bur BYPASS |
| about | 89.9 | 77.2–144.9 | 0/5 | 200 | 4.sjc _atomic_bur HIT |

Checkout returned **302 to Cart** in all five final samples. These are redirect timings, not the loading time of a populated Checkout document. Public pages were much faster in this diagnostic window, but those values do not establish LCP, visual readiness or physical-device performance.

Cart’s BYPASS latency is a real remaining bottleneck. These headers alone cannot distinguish WordPress/PHP work from WordPress.com routing, CDN waits, or network variability. Do not modify theme code on an unsupported attribution. DNS/TCP/TLS and total-transfer medians remain in `analysis.json`.

## Cache behavior and candidate identity

The managed cache transition following deployment is separate from these measurements. Root retained historical Home HTML with the retired phrase and without the optimized mark hook, then observed natural replacement: Last-Modified **01:49:39 UTC**, accepted response **01:50:13 UTC** on September 7. No manual purge command was issued. Standard installer cache housekeeping is documented separately in deployment receipts.

All five final Home timing responses contain `skyyrose-logo-optimized-384w.webm` and omit the retired phrase. Their `STALE` / `UPDATING` labels therefore refer to the current candidate’s cached representation, not evidence that the historical implementation returned.

Observed final HTML Cache-Control:

- **home:** max-age=293, must-revalidate.
- **shop:** (header absent in this sample).
- **signature:** max-age=300, must-revalidate.
- **pdp:** max-age=300, must-revalidate.
- **cart:** no-cache, must-revalidate, max-age=0, private.
- **checkout:** (header absent in this sample); no-cache, must-revalidate, max-age=0, private.
- **about:** (header absent in this sample).

Static code responses expose `max-age=315360000` or `max-age=31536000`, with no `immutable` token in these samples. Long TTL is not a guarantee of correct cache keys or browser/CDN isolation. Three observed ETag validators returned 304/zero body. The exact source comparison protects current untransformed assets; optimized bundles are independently checked only for stable decoded representations.

Browser disk-cache reuse and CDN cache correctness remain distinct. This curl phase uses no browser cache; exposed CDN labels are recorded without inferring HIT from low latency.

## Content types and media integrity

The five distinct strict MIME discrepancies are four fonts and one GLB (the GLB was observed in both the binary and intent-dependency passes):

- `assets/models/skyy-mascot.glb?ver=2.4.4-c571e26fb126`: `application/octet-stream`; target `model/gltf-binary`.
- `assets/sot/fonts/anton-latin.woff2`: `application/font-woff2`; target `font/woff2`.
- `assets/sot/fonts/archivo-latin.woff2`: `application/font-woff2`; target `font/woff2`.
- `assets/sot/fonts/cinzel-latin.woff2`: `application/font-woff2`; target `font/woff2`.
- `assets/sot/fonts/hanken-grotesk-latin.woff2`: `application/font-woff2`; target `font/woff2`.

All **14 sampled binary-signature observations pass**, including the repeated GLB. Valid signatures and successful browser decoding do not erase the strict MIME discrepancy. HTML/CSS/JavaScript response types were correct. PNG/JPEG/AVIF/WASM classes were not sampled here and are not certified by this report.

## Video ranges — including the rotating identity

| Object | Full bytes | Content-Type | Range results |
|---|---:|---|---|
| Approved scene MP4 | 1,209,097 | video/mp4 | PASS |
| Approved hero WebM | 676,790 | video/webm | PASS |
| Optimized rotating mark WebM | 617,195 | video/webm | PASS |

Each object returned exact 1,024-byte slices at offsets 0 and 1,024 with HTTP 206 and correct Content-Range, plus an out-of-bounds HTTP 416 with the correct total. All three exposed a one-year max-age. These tests establish range support; browser playback, reverse scene navigation, seeking and re-entry belong to B13. A browser request beginning at byte zero can still transfer the whole object, so no blanket bandwidth-saving claim is made.

## Ask Skyy deferred dependencies

The fresh browser receipt supplied six actual URLs: five runtime JavaScript modules and one model. Five code URLs were fetched under all three encodings; the GLB once with identity encoding. No decoder WASM or texture URL was manufactured.

The model transferred **6,058,568 body bytes**, equal to the decoded file size and expected source hash; the response exposed a CDN HIT. This does not prove browser session reuse or fast activation. The model remains unchanged; its activation cost is measured separately by B13.

## Ownership and unresolved boundaries

- MIME headers, compression, managed cache policy and CDN representation delivery are platform responsibilities. No platform configuration or theme workaround was applied.
- Cart and Checkout redirect delays are measured; their exact causal split remains UNKNOWN.
- Header/footer animation identity, About content, hero/scenes, native Quick View, cards and Ask Skyy behavior remain separately reported browser/feature gates.
- Home startup CLS, LCP, video readiness, scene proximity and Ask Skyy activation require B13 measurements. This report does not clear previous performance failures.
- Authenticated and populated-cart isolation, physical devices, full accessibility and payment certification remain unverified.
- The probes performed no content edits, orders/payments, manual purges, product/media-authority changes, or configuration changes.

## Evidence

Machine-readable results: `.artifacts/v2-consolidated-staging-20260906/platform/analysis.json`. Detailed records: `encoding.json`, `binary.json`, `ranges.json`, `conditional.json`, `routes.json`, `timings-final.json`, `skyy-observed.json`, `completed.json`, `extras-complete.json`, and `run-binding.json`.

Raw response headers and HTML are restricted beneath `/Users/theceo/.codex/private-evidence/v2-consolidated-staging-20260906/platform`; cookie values are redacted in public JSON.

Executed probe hashes:

- `run.py`: `13c42dbd83131f95b8c860211467dde62c77978644d896dfabc91b97054cb880`.
- `extra-probes.py`: `23114cb3f897ddbee3ba1e36e8af4f0a1095f64a68e08b9c6cc1563b2ee85ca8`.
