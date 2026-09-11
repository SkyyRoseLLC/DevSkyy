# Consolidated V2 staging release — deployed and verified

The consolidated WordPress V2 release is installed at https://staging-7e48-skyyrose.wpcomstaging.com. This is staging only; production was not changed or approved.

ZIP: `skyyrose-flagship-2.zip`

SHA-256: `a4ec431146f31d036d5547137d3274a693ccd3d072be6089126593f642d14f4b`

| Independent result | Status | Evidence |
|---|---|---|
| Filesystem parity | PASS | Fresh 527-file baseline matched; all 536 installed files matched immediately and again after verification. Existing protected media bytes remain unchanged. |
| Browser parity | PASS — bounded | 18 current-candidate route/viewport cases, 34 interaction checks, two full Search GET cases, five About widths and eight rotating-mark/fallback cases. No thrown page JavaScript errors or failed resources in the functional route suite. |
| Feature parity | PASS — bounded | Five animated heroes, nine approved scenes, responsive founder-approved cards, native WooCommerce Quick View, intent-triggered Ask Skyy and chat UI, About, Shop, representative PDP, Bag, empty Cart and Search verified. |
| Platform delivery | NEEDS_MORE_WORK | 750 read-only requests; compression, sampled representations/hashes and video ranges pass. Strict font/GLB MIME targets fail; anonymous Cart response times vary. Authenticated/personalized cache isolation remains environment-bound. |
| Performance | NEEDS_MORE_WORK | 15 cinematic sessions. Home desktop CLS0.845; one normal mobile Home run shows poster-to-video LCP replacement0.468→4.220s. Ask Skyy takes roughly15–16s from intent to first draw on the moderate profile. |

Checkout navigations correctly redirect the empty cart to Cart; populated checkout, payment sandbox, physical devices and field-INP certification remain unverified. No order, payment or chat question was submitted.

## Included in this release

- Accepted cinematic Home and collection system, all nine approved Scroll World scenes, cards, commerce handoffs and native Quick View.
- New native About surface with the existing daughter photograph, collection graphic marks, supplied Kids character, permitted Oakland reference crops and the intent-loaded Blox interview. Exact native page content matches the packaged payload.
- Mandatory animated header and footer identity using the reviewed 617,195-byte transparent WebM derivative, with preserved animated WebP and intentional static fallbacks. Five widths, transparent animation, stable component geometry, deferred footer delivery, reduced motion, Save-Data and controlled WebM failure verified.
- Retired-tagline removal from theme output and the WordPress site description. Fresh Home delivery was verified after normal cache revalidation; cached pre-release HTML is retained as diagnostic evidence rather than counted as current parity.

The release is the WordPress theme/runtime and its narrow About/site-description migration. Local Python, Next.js and other non-WordPress source changes were not deployed as unrelated services.

## Deployment and recovery

The exact package was independently reviewed. Attempt one reached536-file parity but WordPress rejected About against a preexisting legacy template assignment. Whole-theme recovery and exact prior content/description/template identity were verified. An independently reviewed migration correction preserved that metadata and allowed attempt two to complete using the unchanged ZIP. All original logs remain available.

The immediately prior527-file backup SHA is `292ee73ab2bb494b2fe56c20ee8e871d539f658e81dadce20cac7da7912181ab`. The original447-file backup SHA `1b8164b36374aadea85aa4ec7c4d3f8a18e9f8e9f34042f706bd21c090a45ac3` is also preserved and rechecked.

WordPress's installer reported its normal automatic cache housekeeping. No separate manual purge or platform/CDN configuration command was run. About page content and the retired site-description phrase were the only intentional content/option migration scope; this is not a claim of a completely unchanged database.

## Remaining bottlenecks for founder review

1. **Home startup geometry:** critical CSS lacks the header/hero geometry needed before first paint. The desktop hero shifts by the76px header offset. The rotating video begins after this shift and should remain protected.
2. **Home scheduling and delivery:** classic blocking scripts precede hero initialization, and one film request then waits~2.26s for headers despite a HIT label. Dependency/optimizer order and network/CDN delay require separate controlled diagnosis.
3. **Ask Skyy asset:** deferral works and reload browser-cache reuse is observed with zero model transfer, but the unchanged6.06MB model takes~12.25s to fetch plus~1.68s to decode on the moderate primary run. Preserve fidelity in the separate Blender phase.
4. **Platform readiness:** strict MIME corrections, variable uncached Cart TTFB, and authenticated/session cache isolation remain open. No platform changes were made.
5. **About platform UI:** WordPress.com Likes/sharing adds a contrasting strip near the footer. This is recorded for founder/platform review without changing settings.

Shop's slow normal sample was dominated by a~2.99s secure-connection interval; the card image was discovered promptly. Mobile product derivatives were correctly sized. Do not turn that connection delay into an unsupported theme-image fix.

## B13 results

| Delivery surface | Result |
|---|---|
| Home hero delivery | FAIL |
| Collection scene delivery | PASS |
| Product card delivery | PASS |
| Quick View delivery | PASS |
| Ask Skyy deferred delivery | PASS |
| Mobile cinematic delivery | FAIL |

PASS is limited to the recorded Chromium profiles. Performance failure does not negate verified feature preservation; it prevents an unqualified readiness or production claim.

## Evidence

- `deployment-receipt.md`, `deployment-and-rollback.md`, `authorization.json`, `independent-review.md`
- `feature-parity.md`, `about-logo-verification.md`
- `platform-delivery.md`
- `b13-delivery-performance.md`, `b13-causal-notes.md`, `b13-results.json`
- Machine evidence and screenshots: `.artifacts/v2-consolidated-staging-20260906/`

No production deployment, model optimization, product/media-authority changes or Town Line work occurred. Work stops here for founder review.
