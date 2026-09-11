# Staging platform evidence — current authorized package

**EVIDENCE_COMPLETE_WITH_FAILURES_AND_ENVIRONMENT_BOUNDS**. Fresh observations show stable code representations and correct film ranges. Strict font/GLB MIME targets and the Checkout redirect latency target remain unmet. This is not a blanket platform PASS.

Package SHA-256: `e47588205c55e6303588a207d7ee766ad175b82f606d167198f4ec2c25b1d80d`. Root confirmed filesystem parity (527 files) and normal/fallback browser parity before release. Root performed706 requests; report author performed zero remote requests.

Counts:603 encoding,30 preliminary timing (excluded from B12),30 final uncontended timing,12 binary,6 range,3 conditional,16 observed Skyy requests across6 URLs, and6 historical/current key-separation requests. All final timing samples followed browser fallback closure.

| Check | Status | Result |
|---|---|---|
| B01 | PASS | 603 requests: 6 HTML endpoints plus 61 code URLs × 9. All 61 static decoded bodies stable across 3 interleaved br/identity/gzip cycles. Dynamic HTML hashes not compared. Skyy extras five code URLs stable across three encodings. |
| B02 | PASS | All curl decodes successful. 200 responses negotiate br/gzip correctly; identity absent coding. Six compressed-negotiation Checkout302 responses have no coding and are excluded from document compression assertions. Wire and decoded sizes recorded separately. |
| B03 | PASS | All 594 encoding200 responses expose Vary: Accept-Encoding; no static representation divergence. This proves sampled exact URLs only. |
| B04 | PASS | Three observed same-representation ETag conditionals return304 with empty bodies. Cache headers and explicit HIT/MISS/STALE/UPDATING/BYPASS labels retained; no latency-derived hit inference. |
| B05 | PASS_SCOPED | Sampled existing historical/current key separation PASS: three encodings each return distinct correct old5498-byte/current13847-byte bodies. Both observed URL keys pre-existed the probes; no fabricated version query. Global immutable-cache policy remains ENVIRONMENT_BOUND: max-age=315360000 is present, immutable token absent. |
| B06 | FAIL | Binary signatures and direct package hashes match, but strict MIME fails for four WOFF2 URLs (application/font-woff2 versus font/woff2) and one GLB URL (application/octet-stream versus model/gltf-binary, repeated in Skyy). HTML/CSS/JS/WebP/MP4/WebM MIME sampled correctly. JPEG/AVIF/WASM unobserved and not certified. |
| B07 | PASS | MP4 and WebM: four206 responses each1024 bytes with exact Content-Range and independently checked source slices; two416 responses carry exact bytes */TOTAL. Browser playback/seek is independent evidence. |
| B08 | PASS | All sampled emitted/observed URLs HTTPS; successful200 assets, only expected anonymous Checkout302 to Cart, no redirect following. All 37 direct package-comparable request bodies match across 18 unique URLs.24 optimized bundle URLs stable but not asserted byte-identical to source files. |
| B09 | ENVIRONMENT_BOUND | Anonymous Cart/Checkout emit private/no-cache and BYPASS; no authorized authenticated fixture establishes isolation. |
| B10 | ENVIRONMENT_BOUND | No existing approved populatedA/emptyB fixture; no cart was created or mutated. |
| B11 | ENVIRONMENT_BOUND | No approved synthetic session/nonce fixture for personalized endpoints. Public Skyy asset fetches do not establish session isolation. |
| B12 | FAIL | Authoritative30 uncontended final samples: five document-route medians under800ms; Checkout302 redirect median exceeds800ms. Cart median passes but2/5 samples exceed800ms. This is diagnostic end-to-end response latency, not an origin attribution or Checkout-document metric. |

## Final timing — five fresh processes per route

Network-inclusive milliseconds; DNS/TCP/TLS cumulative milestones and total medians/ranges are in JSON. No redirects were followed. Checkout measures its302 response to Cart, not a Checkout document.

| Route | Status | Median TTFB | Range | Samples >800ms | Explicit edge labels |
|---|---|---:|---|---:|---|
| home | [200] | 111.3 | 69.9–128.0 | 0/5 | 1.sjc _atomic_bur HIT |
| shop | [200] | 73.4 | 62.9–126.6 | 0/5 | 3.sjc _atomic_bur STALE, 3.sjc _atomic_bur UPDATING |
| signature | [200] | 72.8 | 67.9–116.1 | 0/5 | 3.sjc _atomic_bur HIT, 3.sjc _atomic_bur STALE |
| pdp | [200] | 75.8 | 59.2–143.7 | 0/5 | 2.sjc _atomic_bur HIT |
| cart | [200] | 137.8 | 100.1–1647.2 | 2/5 | 3.sjc _atomic_bur BYPASS |
| checkout | [302] | 1267.8 | 87.9–1587.0 | 4/5 | 4.sjc _atomic_bur BYPASS |

Home/Shop/PDP expose max-age=300, must-revalidate. Signature samples do not expose Cache-Control. Cart/Checkout expose no-cache, must-revalidate, max-age=0, private and BYPASS. These anonymous observations do not prove authenticated/session separation.

## Remaining scoped findings

- Four WOFF2 URLs serve application/font-woff2; target font/woff2. GLB serves application/octet-stream; target model/gltf-binary. Their magic signatures and package bytes are correct. Runtime consumption does not convert strict MIME failure into PASS.
- Checkout redirect median exceeds800ms; Cart has two slow samples despite a passing median. Browser, field and origin-only measurements require separate evidence. Headers alone do not establish the responsible platform layer.
- Sampled historical/current key separation passes; global immutable policy and authenticated/populated-session isolation remain ENVIRONMENT_BOUND. No sessions were fabricated.

## Byte and range evidence

61 static code URLs are equal across all nine negotiated requests. Five additional observed Skyy code URLs are equal across three encodings. All37 package-comparable response bodies (18 unique URLs) match expected hashes.24 optimized bundles are stable representations, not direct source-byte assertions.

MP4 total1209097 bytes and WebM total676790 bytes: each first/nonzero range returns206 and exact1024-byte source slice; out-of-range returns416 and correct total. No MIME or range verdict is inferred from playback alone.

Expected direct-source hashes: [platform/expected-source-hashes.json](/Users/theceo/.codex/worktrees/7116/DevSkyy/.artifacts/v2-staging-release-20260906/platform/expected-source-hashes.json), bound to release manifest SHA-256 `57c7b48529bd5822d19b9400825b1283c43fd0671edae3e68b9335a20025a291`.

[Full report and observations](/Users/theceo/.codex/worktrees/7116/DevSkyy/tasks/v2-staging-release-20260906/platform-report.json) include original acceptance criteria, static representation hashes, wire/decoded sizes, source comparisons, range validation, timing breakdowns and evidence hashes. B13/browser certification and unresolved founder media gates remain independent.

## Supplemental historical key proof

Six GETs used two already observed URLs: old `99abdd1ab80c` yields the correct baseline5498-byte body across br/identity/gzip; current `d753ac42574e` yields the authorized13847-byte body across all three. Both return `max-age=315360000`, without an `immutable` token. This establishes sampled key separation, not global immutable policy. All six supplemental probes are excluded from the original37 current-package body comparisons; expected historical differences are not release mismatches. B13/browser CLS investigation remains separate and unresolved by these HTTP checks.
