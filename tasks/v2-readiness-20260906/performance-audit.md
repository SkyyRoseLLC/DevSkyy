# V2 readiness: font, CSS, JavaScript and transport audit

**Read-only source audit; engineering targets, not newly measured gains.** The source anchor is `.artifacts/v2-readiness-20260906/baseline-theme/skyyrose-flagship-2`, frozen from the accepted finalization candidate. The inherited Git HEAD is `aabd2bffdc1e322862acd05a5640c14cf00f3acf`; the snapshot contains subsequent uncommitted work, so HEAD alone is not its identity. Input hashes, every authored font declaration, top-level minified bundle size/hash, and route observations are recorded in `audit/critical-path-inventory.json`. No source, staging, cache or server settings were changed by this audit.

## Prior measured critical path

These are the prior finalization candidate’s same-profile Lighthouse simulated mobile values, used as this pass’s starting evidence. They are not results of readiness fixes. Gzip is a separate same-source local transport diagnostic.

| Route | Before LCP ms | Gzip diagnostic ms | CSS KiB | JS KiB | Font KiB |
|---|---:|---:|---:|---:|---:|
| home | 5874 | 3997 | 147.0 | 211.9 | 207.2 |
| shop | 5874 | 4761 | 244.8 | 193.7 | 207.2 |
| collection | 5268 | 3542 | 134.6 | 204.2 | 207.2 |
| pdp | 5508 | 2598 | 260.9 | 287.9 | 207.2 |
| cart | 5743 | 2443 | 289.4 | 259.4 | 134.2 |
| checkout | 5118 | 2420 | 269.5 | 258.9 | 140.6 |

All canonical route LCP gates remained failing. Shop/Cart/Checkout CSS, Checkout JS and Home/Checkout initial transfer also failed. Gzip passing Cart/Checkout does not rewrite canonical results. INP and route animation render-work remain unverified.

Actually throttled observer evidence separates the mechanisms: Home preloaded high-priority 28,800-byte poster completed at 1657.5 ms and painted as LCP at 3860 ms; PDP preloaded high-priority 29,266-byte product derivative completed at 1477.9 ms and painted at 4280 ms. These post-transfer gaps support CSS/layout/scheduling investigation, not another blind image compression pass. Shop’s high-priority 124,228-byte frame had fetch start 816.9 ms, request start 3402.6 ms and response end 5792.6 ms before LCP 5804 ms: request scheduling/competition and payload are stronger targets there. Local HTTP/1.1 queuing is not evidence of identical production behavior.

Home video began around 7959.7 ms with no later VIDEO candidate in that actual-throttled observer. Lantern simulation selected VIDEO. This method-dependent difference remains unexplained; do not invent a eligibility/decode cause or claim video transfer solely caused the observed image LCP.

## Font inventory and use

All registered faces are normal style WOFF2 and are declared globally by `assets/css/design-tokens.css:7–11`; no italic face is registered. CSS can synthesize italic/bold where asked. Registration alone does not fetch an unused face. The JSON records every source font-family/font-shorthand/weight declaration, including inactive legacy source; declaration presence is not computed above-fold proof.

| Family | Declared weights | File bytes | Display | Role / above-fold evidence |
|---|---|---:|---|---|
| Archivo | 100 900 | 90,096 | optional | Display headings; above-fold H1 on Cart/Checkout. Native PDP/story headings and menu use it; whether visible at an exact viewport depends on composition. |
| "Hanken Grotesk" | 100 900 | 34,664 | swap | Body and global header controls (500 at .875rem); above-fold shell on all routes. |
| Anton | 400 | 12,004 | swap | Utility/button/kicker face, normal 400; some authored controls request heavier weights, potentially synthesized. Used above-fold utility controls. |
| Cinzel | 400 900 | 25,904 | swap | Collection title (400), portal inscriptions (700), price/details; visible collection/card art-dependent use. Not fetched on prior Cart/Checkout. |
| Inter | 100 900 | 48,432 | swap | Secondary body fallback after Hanken, no direct authored component family use. Fetched despite fallback role; actual glyph trigger not established. |

The five-face body totals sum to 211,100 file bytes before response overhead; route traces show approximately 212,135 transferred bytes on Home/Shop/Collection/PDP. Cart’s three face set is Archivo/Hanken/Anton; Checkout adds the native WooCommerce icon font (about 6.3 KiB body, 6.6 KiB transferred). Inter is only named in the body fallback chain in authored styles. Removing its downloaded fallback may avoid up to 48,432 body bytes, but requires checking symbols, language coverage and missing-font behavior; the trace alone does not prove those bytes are unnecessary.

No font preloads appear in the inherited six-route observations or the baseline performance preload owner. Retain image preloads. Do not preload five fonts: in actually throttled evidence font-ready is around 7067/6675/7398 ms on Home/Shop/PDP, after each image LCP. Earlier delivery of a font could help text LCP but may compete with images. Test one needed above-fold font on one route with matching crossorigin/type, and check CLS and glyph behavior before adoption. Font subsetting should preserve names, actual ranges, language/punctuation and license records; no subset was created in this audit.

Preserved unregistered authoring fonts: Grand Hotel 18,928 bytes, Pinyon Script 34,692, SkyyRose Love Hurts graffiti 11,352, and SkyyRose Black Rose script 9,520. They are not requested through current design-tokens and must not be counted as initial-network savings if removed. Their authoring purpose remains protected. System monospace is used for index labels and does not create a font request.

## CSS and JavaScript route ownership

`functions.php:343+` is the enqueue owner. Shared CSS: design-tokens, theme, controls, global-shell, visual-recovery. Home adds collection-world/home-page; collection adds collection-world; PDP product-page; Shop/taxonomy shop-page. Remaining content routes, including Cart/Checkout, add legacy-world-components and content-page. Hero/scene film controllers and their CSS are scoped to Home/collection/legacy scene routes; scene-handoff is collection-scoped.

Search-preview and premium-commerce CSS/JS load outside Checkout. Mascot CSS/bootstrap load outside Checkout, with later runtime/model intent loading. Theme JS is global and includes overlays/menu, progressive card preview, bag/variation feedback and older hero-related branches. Native cart fragments/add-to-cart remain dependencies outside Checkout. The JSON contains exact minified source bundle sizes and inherited route transfer/resource entries, so disk weight and wire weight remain distinct.

Native Woo CSS is retained on Shop/PDP/Cart/Checkout. `inc/performance.php:32–88` removes block/public extras only on governed routes and dequeues native Woo general/layout/smallscreen only on Home/editorial collection with opt-back-in. Do not generalize that exception to transactional pages: their native forms, notices, gallery and extension hooks require ownership proof.

**Quiet-route extraction target:** Cart/Checkout currently load legacy-world-components + content-page, alongside theme and native Woo. The initial recommendation to extract a transactional subset is superseded by the root implementer’s narrower source and computed-style audit: no Woo/cart selectors were found in legacy-world-components/content-page, and omission produced zero visible-DOM computed-style differences on Cart/Checkout at 390 and 1440 pixels. These observations support a narrow quiet-route omission trial, rather than asserting that those two files contain required native cart styling. Native Woo/theme/controls/shell styling remains separate. The ablation occurred before the fixture-font isolation correction below, so final acceptance requires corrected-fixture verification. Theme.css still includes legacy component sections and shared commerce contracts; remove only rules with tested route ownership.

**Quick View target and constraint:** `functions.php:418–423` loads Quick View purchase CSS/JS and native `wc-add-to-cart-variation` on every non-Cart/Checkout/Account Woo-enabled route. This is broader than guaranteed card surfaces, so ordinary content pages without cards are candidates for a tighter gate. PDP is **not automatically redundant**: `inc/quick-view-commerce.php:18` explicitly supports variable related cards on a simple PDP; `product-hero.php` preserves native after-summary hooks and Woo related/upsell output. A variable PDP already needs the native variation dependency. Shared handles are not duplicate network loads. Before excluding PDP QV, prove there are no related/upsell/custom cards or provide an intent-loading path for them. Retain extension dependencies and footer variation templates.

Native-only defer was already rejected and reverted in finalization: its isolated measurements did not establish reliable improvement. The Quick View runtime ownership graph also checks late dependencies/inline-after code. Do not blanket-defer jQuery, wp-util, native variation, checkout or extension scripts. Reduce known route payload first, then test a bounded ordering change with commerce regression. Search and concierge access on content routes are intentional; absence of a product grid does not make those modules unused.

## Production and staging transport responsibilities

WordPress.com documents automatic Brotli compression for HTML/CSS/JS, with gzip fallback. This is hosting responsibility; adding a PHP theme output compressor or cache plugin is not the proposed repair. It does not compress image content under that promise. [Official platform storage/compression documentation](https://developer.wordpress.com/docs/platform-features/storage/).

Site Accelerator is enabled by default; its static-file service explicitly covers Core, Jetpack and WooCommerce assets, while custom theme delivery must be verified from actual URLs/headers. Image service operates on public images and supports WebP; it does not serve video/audio. Therefore do not presume the nine films, hero WebM or mascot GLB gain transport benefits from enabling its image toggle. [Official Site Accelerator documentation](https://wordpress.com/support/site-accelerator-cdn/).

WordPress.com distinguishes global edge caching from Memcached object caching. Public visibility is required for global edge caching; object cache is automatically provided. Cache purges are troubleshooting actions that can temporarily slow delivery, not recurring performance optimization. No settings were inspected or changed on the user’s host here. [Official cache documentation](https://wordpress.com/support/clear-your-sites-cache/).

Before an authorized staging verification, record exact source/hash/version, privacy/noindex state and test identity. Read-only GET response evidence should capture negotiated Content-Encoding under br/gzip/identity, Vary, Cache-Control, Age/cache-status if exposed, Content-Type, content length or actual transfer size, and protocol for HTML, custom theme CSS/JS, a native Woo asset, a font, hero image and video. Compare anonymous/public-eligible responses separately from logged-in, cart and checkout sessions. Do not make private staging public to obtain cache benefits; record its limitation. Never cache personalized cart/checkout HTML globally. No cache flush or staging write is needed merely to collect headers.

The local gzip proxy already demonstrates a material text transport sensitivity, not production equivalence. If WordPress.com text is already Brotli-compressed, production may avoid much of the local uncompressed cost; only measured headers and identical route tests establish that. Remaining discovery/critical CSS/JS execution/font contention still belongs to theme engineering even on a compressed host. Static asset hashes/versioning belong to the theme build; encoding/edge configuration belongs to the host.

## Recommended bounded experiments for the root implementer

1. Split verified Cart/Checkout critical styles from legacy/content and preserve all native form states; compare same-source-profile screenshots and budgets.
2. Test the Inter fallback request cause and a system fallback alternative with glyph audit; retain primary typography and authoring fonts.
3. Inspect Shop first frame request priority/discovery under the same profile; a matching first-card-only preload may be tested if the product/card frame resolves deterministically. Do not preload every portal.
4. Gate QV assets on actual supported card surfaces; retain PDP related/upsell behavior and native dependency graph.
5. Keep responsive hero/product preloads, source integrity, no duplicate media requests, reduced-motion/Save-Data and all paid visuals. Measure each change; prior metric improvements are not a substitute for current regression evidence.

This audit intentionally launches no browser or performance job so the root can run exclusive tests. No new LCP result, computed-font attribution, physical-device result or staging transport status is claimed.


## Follow-up correction: measurement isolation and root ablation evidence

The root identified a shared-fixture theme.json leak after this inventory. `theme_root` alone did not constrain Core's `wp_get_theme()` raw-root path; another worktree supplied Core font declarations, including a missing `archivo-normal-width.woff2`. All four authorized routers were corrected as documented in `fixture-isolation.md`. The prior ablations and font-use observations were collected with the contaminated font fixture and are provisional until repeated with corrected source isolation. Their CSS-specific observations are not discarded as fabrication, but they cannot certify the final isolated candidate or explain font costs.

Root's provisional evidence: removing all three native Woo styles on Shop changes pagination/Quick View styling and is rejected; preserving general Woo styles while testing only layout/smallscreen omission is the narrower experiment. Cart/Checkout omission of legacy/content produced zero visible-DOM computed-style differences at 390/1440. Settled leaf-node CDP font counts found no Inter glyph use, while Hanken/Archivo/Anton/Cinzel and system glyph fallbacks were used. That scoped observation does not prove Inter is unused in every loading, language or error state, especially before fixture correction.

Accordingly, the broad transactional CSS extraction recommendation and any inferred unused-PDP-QV recommendation are superseded. Root owns measured narrow changes and the final corrected-fixture report. This audit owns no theme source changes or new performance claim.
