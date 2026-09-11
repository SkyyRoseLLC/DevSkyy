# Track B — delivery / platform readiness

**READY_TO_VERIFY_ON_STAGING** — checklist and evidence contract ready; no staging verification or configuration change has been performed. This is readiness of the verification procedure, not a platform PASS. Owner: hosting/platform engineer; theme engineer supplies source identity and actual asset URLs. Baseline: accepted V2-READINESS-20260906. Track A results must remain separately reported.

## Bind the target before execution

1. Record explicitly authorized staging origin, candidate commit/source-manifest hash, theme version, WordPress/Woo/plugin versions, time in UTC, test location/network, browser/curl version and negotiated protocol. Match deployed asset bytes to the candidate; if staging is another candidate, label all results with that identity. Do not deploy merely to make it match.
2. Preserve current privacy, authentication, noindex and robots policy. Do not make staging public, purge caches, change CDN settings or use random cache-busting URLs to manufacture a cold edge result. A private/authenticated site may not expose anonymous public-edge behavior: label that portion ENVIRONMENT-BOUND.
3. Discover actual URLs from the staging HTML and browser waterfall. The local inventory attached to this track is a selection aid, not proof of remote paths or CDN rewrites. Select Home, Shop, Signature collection, one approved-gallery PDP, Cart, Checkout; include one representative asset of every class below.
4. Use authorized synthetic test sessions only. Provisioning a cart/customer/order is outside this read-only checklist and requires separate authority. With no existing synthetic cart sessions, record personalization tests BLOCKED rather than silently creating sessions. Do not submit payment/order requests. Avoid add-to-cart query URLs, logout actions and state-changing AJAX calls during header collection.
5. Keep cookies, authorization headers, nonces and raw personalized bodies in restricted local evidence. Publish only redacted headers, body hashes and non-sensitive assertions. Never paste cookie jars or full private response bodies into a report.

## Exact verification matrix

Every row starts NOT_RUN. Record PASS, FAIL or ENVIRONMENT-BOUND with observed values and artifact paths; missing access/data is not PASS.

| ID | Check / sample | Expected acceptance | Owner when failed |
|---|---|---|---|
| B01 | GET HTML, theme CSS/JS and native Woo CSS/JS separately with `Accept-Encoding: br`, `gzip`, `identity`; repeat each 3 times | Negotiation produces supported coding; explicit identity must not receive an unsupported coding. Decode responses successfully. Static decoded hash matches identity/source bytes. Do not compare dynamic HTML hashes as if nonces were static. Record actual compressed byte count, not decoded browser resource size. | Platform/server/CDN |
| B02 | Brotli and gzip separate response headers plus decoded payload | WordPress.com documents Brotli for HTML/CSS/JS with gzip fallback; any observed exception gets a specific route/MIME/status/access explanation. Already-compressed WOFF2/images/video need not be double-compressed. A missing Brotli-capable client is ENVIRONMENT-BOUND. | WordPress.com platform |
| B03 | `Vary` and CDN representation keys | Negotiated content must not leak incompatible variants. Record `Vary: Accept-Encoding` where representation varies, or documented CDN normalization/key behavior plus interleaved br→identity→gzip proof. Test actual image Accept/WebP/AVIF variants if negotiated; do not demand Vary on invariant files. | CDN/server |
| B04 | `Cache-Control`, `Expires`, `Age`, `ETag`, `Last-Modified`, exposed cache status on HTML and every static class | Record policy per resource; no blanket immutable HTML. Do not infer HIT from speed or MISS from absent proprietary headers. Conditional GET should honor applicable validators; note changed representation/weak validator limits. | Platform/CDN |
| B05 | Existing content/versioned static URLs and existing second-version URL, if available | Long-lived `immutable` only if complete cache key changes with bytes. Verify current version returns current source hash and older version does not alias new bytes incorrectly. If no historical version/key documentation exists, immutability/key verification is ENVIRONMENT-BOUND; do not upload a test file or change query parameters and call that proof. | Theme for version key; CDN for key honoring |
| B06 | GET content type and actual file signature for HTML/CSS/JS/WOFF2/JPEG/WebP/AVIF/MP4/WebM/GLB/WASM | Valid MIME compatible with response bytes: text/html, text/css, valid JS MIME, font/woff2, matching image type, video/mp4 or video/webm, model/gltf-binary, application/wasm. No login/error HTML disguised as an asset. Cross-origin font/model/decoder requests must also load under the actual CORS policy. | Server/CDN/platform |
| B07 | MP4/WebM GET `Range: bytes=0-1023` and a nonzero range with identity encoding | 206, correct `Content-Range`, exact requested bytes and matching slices of full object. Record 416 behavior for an out-of-bounds range. `Accept-Ranges` alone is insufficient. A 200 full response is legal HTTP but fails this film-delivery target. Verify actual playback/seek/resume and reverse scene navigation later in browser. | Server/CDN |
| B08 | Actual final URLs/redirects for custom theme, Woo/Core, font, film, model, poster/card | HTTPS, correct origin/CDN attribution, no redirect to auth/error content, no unnecessary chain. Preserve canonical hashes where transform is not authorized. Image CDN transformations require correct aspect/crop and approved product identity. Do not assume Site Accelerator handles custom films/models. | CDN/platform; theme if emitted URL is wrong |
| B09 | Anonymous public-eligible route repeated GET; separate empty synthetic session and logged-in synthetic session | Shared cache must not expose another session's contents. Record cache state for each identity separately. Preserve private/noindex policy; unavailable anonymous-public behavior is ENVIRONMENT-BOUND. | Platform/CDN |
| B10 | Existing Cart A versus empty B, and Checkout A versus B, request same URLs in A→B→A order | Session A line item/variation/quantity/totals stay A-only; B remains empty. No public shared caching of personalized pages or native cart/session AJAX endpoints. Inspect relevant Woo session/cart cookies and authentication bypass behavior with values redacted. Absence of an exposed HIT header is not isolation proof. | Platform/CDN/cache integration |
| B11 | Account/login-sensitive pages, cart fragments, `wc-ajax`, Store API cart/checkout routes actually used | Capture only read-only endpoints already used by the installed architecture; exclude personalized responses from shared public caching. Do not issue mutation endpoints. Verify no mixing across sessions and no stale nonce/variation state. Plugin/Blocks/classic architecture determines exact route list. | Platform/cache plugin/Woo integration |
| B12 | Actual TTFB across six routes, 5 sequential new-client samples per identity/cache class | Record DNS, TCP, TLS, request-to-first-byte, total, redirects, HTTP version, encoding and edge state. Report median and range; initial network-inclusive TTFB is distinct from origin/server-only time. DNS/TLS reuse and authenticated cache bypass must be labeled. Compare no sample with a different network/profile as if causal. Proposed diagnostic threshold: median navigation TTFB ≤800ms; above it triggers attribution, not automatic blame of theme. | Platform first; application only with profiling evidence |

## Resource sample contract

Include HTML for all six routes; active global theme CSS, route CSS, theme JS, a native Woo stylesheet/script, a Core script, primary heading/body fonts and fallback font, Home poster/mobile film, Collection arrival poster/scene film, Shop portal/card image, approved PDP gallery image, and intent-loaded Skyy runtime/GLB/WASM. A file not requested on the route is recorded as absent rather than fetched preemptively for a performance run. Standalone header probes and browser navigation profiles are separate experiments.

For each request retain: check ID, route, resource class, original and final URL (without credentials), exact request method and non-sensitive headers, session label, UTC, candidate/source hash, client/network, protocol, status, redirect chain, content type/encoding, cache policy/state, Vary, validators, range metadata, downloaded bytes, decoded bytes/hash for static assets, timing, interpretation, owner and next action.

## Prepared command recipe — NOT EXECUTED against staging

Bind `ASSET_URL` to a verified, non-mutating HTTPS resource and `EVIDENCE_DIR` to restricted local storage. Do not put credentials in a URL or shell history. Review redirects before following them; these examples deliberately do not follow automatically. `curl --version` must show the relevant compression support. Use the three exact modes separately; `--compressed` decodes supported responses into the output while size_download measures downloaded body bytes. Do not add `--raw`, which would bypass this decoding contract.

```sh
curl --version
curl --silent --show-error --compressed --max-time 30 \
  -H 'Accept-Encoding: br' --dump-header "$EVIDENCE_DIR/br.headers" \
  --output "$EVIDENCE_DIR/br.decoded" \
  --write-out '%{json}\n' "$ASSET_URL" > "$EVIDENCE_DIR/br.metrics.json"
curl --silent --show-error --compressed --max-time 30 \
  -H 'Accept-Encoding: gzip' --dump-header "$EVIDENCE_DIR/gzip.headers" \
  --output "$EVIDENCE_DIR/gzip.decoded" \
  --write-out '%{json}\n' "$ASSET_URL" > "$EVIDENCE_DIR/gzip.metrics.json"
curl --silent --show-error --max-time 30 \
  -H 'Accept-Encoding: identity' --dump-header "$EVIDENCE_DIR/identity.headers" \
  --output "$EVIDENCE_DIR/identity.body" \
  --write-out '%{json}\n' "$ASSET_URL" > "$EVIDENCE_DIR/identity.metrics.json"
shasum -a 256 "$EVIDENCE_DIR/br.decoded" "$EVIDENCE_DIR/gzip.decoded" "$EVIDENCE_DIR/identity.body"
```

Do not treat successful curl exit as status/MIME/encoding correctness; inspect the response and metrics. For the verified film URL, repeat identity GET with `-H 'Range: bytes=0-1023'` and `-H 'Range: bytes=1024-2047'`, distinct output files, and compare with the corresponding full-body slices. Save raw private evidence locally and produce a redacted result separately. Five repeat timing samples use separate curl processes without cache purges; name them first-observed/warm only when response evidence supports the distinction. Browser TTFB/performance run remains a separate artifact.

## Release of a verification result

A second reviewer checks resource/source identity, session isolation, encoding equivalence, range bytes and timing classification. Any failure receives one of THEME / SERVER / CDN / WORDPRESS.COM / UNKNOWN with observed evidence. Theme changes require their own visual and native-commerce gates. No failed platform check authorizes a PHP compressor, new caching plugin, cache purge or staging alteration. The next phase may execute this checklist only against an explicitly authorized staging target.

Current outcome: procedure **READY_TO_VERIFY_ON_STAGING**; all remote checks **NOT_RUN**. Stage origin/access/session inputs are execution prerequisites, not invented observations.

## Current authoritative references

- WordPress.com documents managed Brotli for code responses and gzip fallback: [platform storage/compression](https://developer.wordpress.com/docs/platform-features/storage/).
- Site Accelerator scope and private-image limitations: [official CDN guide](https://wordpress.com/support/site-accelerator-cdn/). Verify each observed asset hostname; custom film delivery is not inferred from an image CDN.
- WooCommerce caching exclusions and session considerations: [official WooCommerce caching guidance](https://developer.woocommerce.com/docs/best-practices/performance/configuring-caching-plugins).
- HTTP validation semantics: [Cache-Control](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cache-Control), [Vary](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Vary), [Range](https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Range). The stricter film-range and personalization gates above are project acceptance criteria.

References checked during this phase. Documentation describes expected behavior; it is not live staging evidence.
