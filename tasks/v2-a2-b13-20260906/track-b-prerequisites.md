# Track B prerequisites for B13

**GO for read-only public-route B13 diagnostics; NO-GO for platform or accepted-candidate certification.** Stop HTTP probing here. Root must release the exclusive browser window before browser work. No config change, staging write, cache purge, cookie/session provisioning, order or payment was performed.

Target: `https://staging-7e48-skyyrose.wpcomstaging.com`. Results belong to **STAGING-OBSERVED-f2893d998356**, not the accepted local source. Root's read-only SSH identity reports 322 observed files from 451 requested: 295 match, 27 differ and 129 are absent versus accepted local inventory. This is an observed subset, not a complete remote-tree/commit digest. See `.artifacts/v2-a2-b13-20260906/deployed-identity.json`.

Home, Shop, Signature, Black Rose, Love Hurts and `/product/sg-005/` each return public HTML200 with `noindex, nofollow, noarchive`. Their read-only browser diagnostics are eligible. Keep the actual browser coding and delivered bundle hash in B13 evidence because encoded variants currently differ. Missing deployed features must remain unexercised, not be borrowed from local results. The earlier SG-005 discovery HTML sample contained no product-gallery image markup. B13 subsequently captured a visible Product Archive gallery in the browser; the earlier absence is response-specific and must not be generalized. Neither observation independently grants approved-gallery media authority. Anonymous Checkout302 to Cart is an empty-session redirect, not a tested Checkout page.

## B01–B12 outcomes

PASS is strictly scoped to observed requests below. ENVIRONMENT_BOUND means prerequisites or relevant samples were unavailable; it is not PASS.

| Check | Status | Observed result / boundary |
|---|---|---|
| B01 | **FAIL** | 117 encoding GETs across seven actual code URLs and six HTML endpoints, three br→identity→gzip repeats each. Two combined resources return different decoded code under br versus identity/gzip. Other five sampled code resources match. Combined CSS/JS includes theme/Woo constituents; no un-emitted standalone URLs were invented. |
| B02 | **PASS** | All sampled200 HTML/code responses negotiate br, gzip and unencoded identity correctly and decode successfully. Empty Checkout302 has no encoding. This does not cure B01 byte drift. |
| B03 | **FAIL** | `Vary: Accept-Encoding` is present, but Signature CSS/PDP JS representations are inconsistent. Three interleaved repeats reproduce the differences. Suspected cache/combined-asset delivery layer; exact cause/owner remains UNKNOWN pending platform investigation. |
| B04 | **PASS** | Policies, Vary, ETags and exposed `_atomic_bur` states recorded. Three same-representation conditional GETs return304/empty body. Cart/Checkout private/no-cache observed. This is not session-isolation proof. |
| B05 | **ENVIRONMENT_BOUND** | No historical version URL/key behavior documentation supplied. Long max-age observed; complete cache-key/immutability correctness unverified. No test upload or query manipulation. |
| B06 | **FAIL** | Font responses use `application/font-woff2` and GLB uses `application/octet-stream`, differing from specified `font/woff2`/`model/gltf-binary` targets. File signatures and all13 applicable source hashes are valid. Actual browser compatibility is a separate check; no claim these MIME exceptions alone prevent rendering. JPEG/AVIF and approved-gallery sampling unavailable. |
| B07 | **PASS** | Signature heroWebM/MP4 and first scene720pMP4 pass exact start/nonzero206 slices and out-of-range416. Browser playback/seek/reverse navigation remains B13. |
| B08 | **PASS** | Sampled assets resolve directly to HTTPS200 on authorized origin; all13 applicable files match SSH observed source. No cross-origin CORS/image-transform certification. Checkout redirect explicitly recorded. |
| B09 | **ENVIRONMENT_BOUND** | Anonymous requests completed without cookie jars. No existing authorized logged-in synthetic session supplied; identity comparison unavailable. |
| B10 | **ENVIRONMENT_BOUND** | Existing populatedA/emptyB sessions unavailable. No session/cart created; A→B→A personalized isolation unverified. |
| B11 | **ENVIRONMENT_BOUND** | No authorized session-bearing account/cart-fragment/StoreAPI fixture supplied. No mutation or private/customer endpoint probes. |
| B12 | **ENVIRONMENT_BOUND** | Five sequential fresh-curl samples for each of six original checklist routes recorded. Checkout samples measure302 only. Cart median1524.5ms exceeds proposed800ms diagnostic threshold; this is not attributed to theme without profiling. Missing identities prevent full B12 completion. |

## Material representation drift

Same exact URL, no cookie jar, same client, three interleaved repeats:

| Combined resource | Brotli decoded | Identity/gzip decoded | Evidence |
|---|---:|---:|---|
| `/_jb_static/??c9711b8a62` (Signature CSS) | 450930 bytes | 452031 bytes | Different decoded SHA-256 and ETag each representation; stable across repeats. |
| `/_jb_static/??955c29654e` (PDP JS) | 29026 bytes | 30679 bytes | Different decoded SHA-256 and ETag each representation; stable across repeats. |

At zero-based decoded byte **200994** in Signature CSS, Brotli continues `--sr2-header:76px`; identity/gzip inserts `--sr2-layer-guide:80;--sr2-layer-header:100;--sr2-layer-skip:999;` first. At byte **2618** in PDP JS, Brotli starts the older menu toggle; identity/gzip adds the `main, body > footer, #skyyrose-mascot, #skyyrose-mascot-recall` selection, saved inert state and dialog closure/focus-management code. These are substantive source differences, not compression-only differences. This evidence does not identify which origin/edge build is authoritative. No purge or remediation attempted.

Exact snippets, hashes, request UTCs, ETags, coding and byte counts are in `platform/bundle-variant-drift.json`, `platform/encoding.json` and individual redacted request JSON. Raw headers and anonymous HTML bodies were moved outside the artifact webroot to `/Users/theceo/.codex/private-evidence/v2-a2-b13-20260906/platform/`.

## Range proof

All three films returned matching full objects first. Requests `bytes=0-1023` and `bytes=1024-2047` then returned206,1024 bytes and correct Content-Range; both bodies equal exact full-object slices. Out-of-range GET returned416 with `bytes */TOTAL`.

| Film discovered from Signature HTML | Full object bytes |
|---|---:|
| `signature-golden-gate-monuments-motion-v1.webm` | 472889 |
| `signature-golden-gate-monuments-motion-v1.mp4` | 855303 |
| `sig-commerce-1-motion-k1-720p.mp4` | 1222252 |

Range acceptance covers these representatives, not all films or playback behavior. Media signatures match WebM/MP4. WOFF2/WebP/GLB/WASM signatures also pass, with thirteen sampled binary/runtime SHA-256 values matching the deployed observed inventory.

## TTFB observations

Curl8.20.0 with Brotli1.2.0/gzip support, macOS host internet connection, HTTP/2 and exposed SJC edge labels. Each sample uses a new curl process with default config disabled and no cookie jar. DNS/TCP/TLS cumulative timings and total are retained separately. TTFB below is network-inclusive `time_starttransfer`; it is not origin-only time. No cache purge or manufactured cold request was used.

| Route | HTTP | Median TTFB | Range |
|---|---|---:|---:|
| Home | 200 | 59.4ms | 55.9–105.7ms |
| Shop | 200 | 71.4ms | 62.6–114.1ms |
| Signature | 200 | 67.2ms | 61.7–124.6ms |
| SG-005 PDP | 200 | 65.0ms | 55.4–131.5ms |
| Cart | 200 | 1524.5ms | 104.3–1741.1ms |
| Checkout, redirect only | 302→Cart | 145.1ms | 84.1–1296.0ms |

Initial public responses expose STALE, while Cart/Checkout first samples expose BYPASS and private/no-cache. All individual cache labels are retained; do not interpret latency alone as HIT/MISS. Cart variability and code-variant drift are platform investigation inputs, not authorization to alter cache settings.

## Evidence and handling

- `track-b-prerequisites.json`: per-check decisions, timing statistics, representation summaries, source comparison and range results.
- `.artifacts/v2-a2-b13-20260906/platform/`: reproducible curl probe scripts,117 negotiation results,30 timing results,12 binary objects,9 range responses,3 conditional responses,2 additional public route GETs and4 initial discovery GETs.
- Raw headers and anonymous HTML bodies are outside the artifact webroot in `/Users/theceo/.codex/private-evidence/v2-a2-b13-20260906/platform/`, directory0700/files0600. JSON path references were updated and JSON headers redact Set-Cookie values. Only static CSS/JS/binary bodies remain in platform artifacts. No credentialed session or customer data was requested. Anonymous nonces remain private and must not be pasted into public reports.

One attempted full-text comparison became unexpectedly CPU-heavy and was immediately interrupted; root was notified to invalidate any overlapping A2 performance sample. The retained drift proof uses only linear byte-prefix comparison. No browser process was launched.
