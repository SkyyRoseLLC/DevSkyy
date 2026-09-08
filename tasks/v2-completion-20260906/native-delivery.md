# Native media and archive delivery checkpoint

2026-09-06. Local candidate only; no deployment or release acceptance.

The hero now exposes its actual decoded native video poster before playback, rather than keeping that video layer transparent through the playing fade. The existing responsive picture remains underneath. Source selection, video bytes, decode, reduced motion, Save-Data, pause, visibility and error boundaries remain intact. Eight controller regressions pass. Independent source review found no defect; four route/viewport transition comparisons found no new flash or crop jump. Their raw AFTER navigation timings were slower, including before controller attachment; timing nonregression remains unverified by those captures.

A separate native archive hint follows the already-resolved main WooCommerce query, does not consume its cursor or set up a Woo loop, and preloads only the first visible product's exact existing frame. Unknown first-product presentation, category/mixed displays and extension-owned visibility/loop behavior omit the hint. Standalone regressions cover ordering, no mutation, deduplication and 15 exclusion cases. Nine actual Shop, sort, pagination, category and empty-result responses match the hint to the rendered first frame or omit both.

Home PHP formatting preserves byte-identical rendered HTML across omitted, partial and custom escaped label fixtures. This closes the source issue behind the PR readability comment after integration.

## Measurements

Same isolated Nginx static/gzip and PHP Xdebug-off profile, unchanged Lighthouse settings, serial runs with no competing local jobs:

| Case | Before this checkpoint | Single candidate run | Acceptance |
|---|---:|---:|---|
| Home mobile LCP | 3.801 s | 2.027 s | Repeat matrix still required |
| Shop mobile LCP | 3.401 s | 3.247 s | Fails 2.500 s target |

Home candidate score93, CLS0.008324; Shop score92, CLS0.000215. The native-poster run's TTFB1291ms is retained in evidence, not excluded. No broad production performance claim is made. Raw local PHP serving and previous proxy profiles are distinct from this profile and remain recorded separately.

Evidence: `.artifacts/v2-completion-20260906/hero-handoff/`, `archive-preload-parity.json`, `lighthouse/lighthouse-profile-native-poster-*`, `lighthouse/lighthouse-profile-archive-preload-*`, and `verify-native-delivery.log`. Full pinned build and verify passed, including 86 Node tests and all PHP/native/source/token/media contracts. Delivery profile tooling has its separate source review and wire-parity receipts.

All 402 protected media/font/model/decoder files remain byte-identical to the recorded baseline. Artifact-only font/frame studies are not included or wired by this checkpoint. Final repeated performance, final integrated package and acceptance remain open.
