# SkyyRose Home Critical Rendering Repair Report

Branch `codex/v2-home-critical-rendering-20260907`, commit `5f807df3b`, baseline `233e9990d`
(byte mirror of the tree staging served on 2026-09-07 01:45 UTC). Evidence tags:
`[live]` staging probe · `[repro]` executed locally · `[repo]` read from source · `[test]` a
gate that ran and could fail · `[inferred]` reasoned, not observed.

## Status

**Deployed to staging 2026-09-07 14:15Z on founder approval; verified live.** Home reached
the acceptance bar in the local mirror at every width `[repro]` and on staging `[live]`
(CLS 0.000 at 320/390/414/768/1440, derived-output parity PASS, policies and commerce
27/28 in Chromium and WebKit with the single failure being pre-existing third-party
payment-script console noise). Production skyyrose.co untouched. One deploy incident
(11 runtime data files dropped by the V1 deploy allowlist) was found and repaired within
minutes; see "Staging After Deploy".

## Critical CSS

- Served block on staging is Jetpack Boost row 10476 (`cornerstone_d41d8cd9`), generated
  2026-09-01 against the previous header; it carries none of the current house header,
  hero stage or concierge geometry `[live]`. Boost's concatenation defeats WP Core's
  inline-small-styles policy, so nothing structural reached `<head>` `[live]`.
- Fix: theme-owned structural contract `assets/css/critical/home.min.css` (15,813 bytes,
  budget 16,384), built deterministically from the enqueued source sheets by
  `scripts/build-critical-css.mjs` per `home.contract.json`, printed at `wp_head` priority 1
  on the front page by `inc/critical-rendering.php`. Covers header, brand-mark container,
  Ask Skyy launcher, hero stage, first-view typography, primary/secondary controls,
  concierge stage (mascot rules the loader needs before full CSS), worlds strip, and the
  `@font-face` set. Fails closed on an absent or malformed file. Independent of Boost.
- Computed-style parity, critical-only vs settled, at 320/390/414/768/1440: no first-view
  geometry differences remain `[repro]` (`scratchpad/measure/style-diff-after2.json`).

## Jetpack Boost

- Identity `[live]`: `jb_store_css` post 10476 for Home; posts 10668/10669 refreshed
  2026-09-06 16:41 for /pre-order and the posts page; Home regeneration recorded
  "Load failed" with no provider errors. Invalidation only fires on `after_switch_theme`
  `[repo]`; same-theme file deploys never refresh the block. `wp jetpack-boost` has no
  regenerate command `[live]`; regeneration runs in the admin browser.
- Isolated regeneration with Boost's own generator (`@automattic/jetpack-critical-css-gen`
  1.0.20, Boost's viewports 414×896/1200×800/1920×1080, keyframes/animation filters,
  sandboxed iframe with scripts off) succeeds locally on both BEFORE and AFTER builds in
  Chromium and WebKit (33,991 / 34,178 bytes, ~1 s, no errors) `[repro]`. "Load failed"
  did not reproduce; its cause on WP.com remains `[inferred]`.
- Coverage finding `[repro]`: because Boost renders without scripts, its "above the fold"
  set never includes `.sr2-archive-scene__intro`, the primary control or the concierge
  stage, even when fresh. A Boost regeneration alone could not have fixed Home; the theme
  contract is load-bearing. Boost's block is now informational for Home.

## First Paint

Chromium, local mirror, 3 samples per width (median, min–max) `[repro]`:

| Width | Build | CLS | FCP ms | Controller init ms | Film request ms | First film frame ms |
|---|---|---|---|---|---|---|
| 320 | BEFORE | 1.000 | 352 | 397 | 415 | 470 |
| 320 | AFTER | 0.000 | 340 | 286 | 384 | 435 |
| 390 | BEFORE | 1.000 | 336 | 383 | 401 | 464 |
| 390 | AFTER | 0.000 | 332 | 288 | 386 | 431 |
| 414 | BEFORE | 1.000 (0–1.0) | 388 | 393 | 412 | 451 |
| 414 | AFTER | 0.000 | 328 | 276 | 372 | 440 |
| 768 | BEFORE | 0.885 | 344 | 389 | 408 | 454 |
| 768 | AFTER | 0.000 | 328 | 274 | 371 | 430 |
| 1440 | BEFORE | 0.845 (0–0.85) | 368 | 396 | 415 | 474 |
| 1440 | AFTER | 0.000 | 332 | 275 | 373 | 439 |

Staging BEFORE `[live]`, 2 samples per width: CLS 1.004 / 1.016 / 0.514 / 0.516 / 0.925;
FCP 2.2–2.8 s; controller init 2.2–2.9 s; first film frame 2.3–3.8 s. The local mirror
reproduces the geometry collapse (hero −360…−403 px, CTA −755 px) `[repro]`.
First paint was not made later: no stylesheet was made render-blocking; the contract is
inline and the full sheets stay asynchronous.

## CLS

BEFORE ≈ 1.0 (hero and CTA collapsing hundreds of pixels at the first-paint→settled
correction). AFTER 0.000 at every width and sample after three fixes: the structural
contract, the concierge-stage rules (critical-only render was 419 px vs 240/288 px
settled), and preloading the four first-view faces (the intro grew one line and the
worlds strip re-wrapped when Hanken Grotesk / Cinzel swapped in). Layout-shift entries in
the AFTER runs: none.

## Hero Startup

- Blocking chain BEFORE `[live]`: jQuery in `<head>` (platform-added
  `data-jetpack-boost="ignore"`, not theme-added `[live]`), then at the end of body:
  jquery-migrate → Boost bundle (WC + theme) → underscore → wp-util →
  add-to-cart-variation → … → the hero controller inside a later bundle. Theme `defer`
  strategies are dropped by Boost's concatenation `[repo]`. Registered dependencies of the
  controller: none; delivered position: last.
- AFTER: the unchanged controller is printed inline immediately after the hero markup with
  `data-jetpack-boost="ignore"` (Boost's rewriter leaves it in place); its footer copy is
  dropped on Home only; collection routes keep the enqueued controller. Delivered order:
  bootstrap before every classic body script `[repro]`. Controller init 275–290 ms locally
  vs 383–397 ms BEFORE; the remaining ~100 ms between init and the film request is
  browser request scheduling while classic scripts are in flight, not application waiting.
- WooCommerce, Quick View dependency order, and Ask Skyy dependencies untouched `[repo]`.

## Film Delivery

Staging, 6 cold samples (fresh context) per width, Chromium `[live]`
(`scratchpad/measure/film-delivery-before-staging.json`):

| Width | Request start after nav (median, range) | TTFB | Transfer end | First frame after request |
|---|---|---|---|---|
| 390 | 2033 ms (1801–2296) | 28 ms (24–54) | 84 ms (56–128) | ≈ 60–150 ms |
| 1440 | 2140 ms (1781–2391) | 28 ms (24–31) | 89 ms (64–155) | ≈ 60–130 ms |

All 12 samples: HTTP 206, 676,790 bytes, TLS 1.3, `x-ac: 3.sjc _atomic_bur HIT`,
`cache-control: max-age=31536000`. A cold-edge MISS was not observed and is not
characterised. Delivery is platform-owned and fast. **Correction to the first draft of
this section:** of the 1.8–2.4 s between navigation and the film request, the uncached
document render (TTFB of a cache-busted request, 2.0–2.6 s median per width) is the
bulk; the in-page schedule from document arrival to the film request was 140–840 ms
BEFORE (see "Staging After Deploy"), and that in-page part is what the repair changes.
Real visitors normally receive the page from the edge cache (`max-age=300`), so the
document TTFB they see is far lower than the cache-busted figures here.

## Fonts

First-view faces `[repo]`: Archivo (title, `font-display: optional`), Hanken Grotesk
(intro, `swap`), Anton (controls, `swap`), Cinzel (worlds strip, `swap`). No new font.
Front page now preloads the four files with hrefs byte-equal to the `@font-face` URLs
(one download each, gated by the derived-output verifier). Chromium: fonts complete at
≈295 ms, first paint ≈388 ms, no swap `[repro]`. WebKit still paints ~10 ms before the
faces land locally and then reflows (title width, intro one line) `[repro]`; that
residual is a race the preload narrows but cannot close in every engine. A
metric-matched local fallback (`size-adjust`/`ascent-override`) would close it without a
new font but changes the type fallback stack — founder decision, not done here.

## Rotating Mark

`.sr2-brand-media` container (80×48) is in the contract; header height 64/76 px at first
paint equals settled; brand-mark Δy 0 at every width `[repro]`. Animation and markup
unchanged.

## Ask Skyy

Untouched. Launcher present in the header, no 3D request on load (0 GLB/WASM/three
requests) `[repro]`; concierge-stage rules added to the contract only for geometry.

## Commerce

Product Card → Quick View → size → native variation (id 181) → Add to Bag (POST 200,
`wc-ajax=get_refreshed_fragments`, bag 0→1) · Search (dialog, 9 results) · Bag (dialog
opens) · Menu (12 links): all pass in Chromium and WebKit, no console/page errors
`[repro]` (`tools/v2-runtime/verify-home-policies.mjs`, 28/28 both engines).

## Reduced Motion / Save Data

Reduced motion: film not revealed, not fetched, poster visible, control hidden.
Save-Data: same. Hidden document pauses, visible resumes. Pause/Play toggles. No-JS:
header, hero, CTA and title laid out from the inline contract, poster shown, no film
source, no console errors `[repro]`.

## WebKit

BEFORE and AFTER behave identically in a plain Playwright WebKit context (headless and
headed): the engine denies the unattended muted `play()` and the controller keeps the
poster with a "Play motion" control; clicking it starts the film `[repro]`. Under the
measurement harness both builds autoplay (AFTER first frame 519–842 ms vs BEFORE
684–1015 ms). Real Safari on staging needs an eyes-on check after deploy.

## Visual Review

BEFORE/AFTER first-paint and settled captures at 320/390/414/768/1440 and startup
recordings (390/768/1440) are under `scratchpad/measure/{before-baseline,before-staging,after-local}/`.
Independent red-team verdict: see the appended section.

## Release Pipeline

Documented in the theme README ("Home critical rendering and release parity"):
SOURCE ARTIFACT (`npm run build && npm run verify`, `check:critical`) → FILESYSTEM PARITY
(deploy verification) → GENERATED CRITICAL CSS PARITY (`verify-home-derived-output.mjs`:
contract bytes, bootstrap position/ignore, font preloads, served-file sha256) → MANAGED
BUNDLE PARITY → BROWSER PARITY (`measure-home-critical.mjs`, `verify-home-policies.mjs`).
Boost regeneration is an optional post-deploy step from the Boost admin; Home no longer
depends on it.

Gates run this session `[test]`: `check:critical` OK; `test-critical-rendering.php` OK;
derived-output parity PASS (local); `npm run verify` green on every step except
`check:commerce-scenes`, whose runtime-PHP baseline was already red at the deployed
baseline for 13 files this repair did not touch (reconciled only front-page.php,
functions.php, + the new module), and `test-pdp-gallery.php`, which passes when
`V2_WP_FIXTURE` points at the local WP 7.1 / WC 11.1.0 root. phpcs: new module clean;
edited files unchanged against the pristine tree (77/181 and 6/9 identical).

Also fixed: `scripts/deploy-theme.sh` hardcoded the archive root `skyyrose-flagship`;
a V2 deploy would have moved the live directory away and then failed. Now uses the
source basename and checks it exists before the swap (syntax-checked, not exercised).

## Independent Visual Verdict (fashion-visual-qa-red-team, Chromium captures)

**PASS at 390, 768 and 1440. No design change; AFTER first paint is already the final
structure.** Settled AFTER vs BEFORE renders are pixel-identical below the header at every
width; at 768 and 1440 at least one AFTER sample equals a BEFORE sample byte for byte. The
only settled-pixel difference is the rotating mark's rotation phase (same asset, size and
position), a side effect of the earlier controller init. AFTER paints the CTAs in their
final rose fill / white outline and the concierge label and buttons at first paint, where
BEFORE painted gold-outlined interim buttons and an empty stage at 390. DOM geometry for
header, mark, hero, title, intro, both CTAs, concierge, worlds strip and Ask Skyy launcher
is equal in both builds at every width. Two pre-existing behaviours are shared by both
builds and untouched: the title re-fades at controller-ready, and the poster is dimmer than
the first film frame. Evidence: `scratchpad/measure/diffs/`, `frames-<label>-<width>/`.
Limit noted by the reviewer: the BEFORE unstyled paint is evidenced by the DOM geometry
snapshot and layout-shift entries, not by a PNG (the first-paint capture landed after the
shift).

## Staging After Deploy `[live]`

**Deploy.** Founder-approved. Source was an exact staged set: staging's 536 files (531
byte-identical to the worktree, 5 changed: front-page.php, functions.php, README,
CHANGELOG, translation catalog) plus the 3 new files (`inc/critical-rendering.php`,
`assets/css/critical/home.min.css`, `home.contract.json`). Hot-swap succeeded, caches
flushed, version stamp verified, backup kept at
`skyyrose-flagship-2.old.1788790527-87183`. The plain Home URL went
`x-ac STALE → UPDATING → HIT` with the new head within about a minute (`max-age=300`).

**Incident (bug-325).** The deploy script's V1 `data/` allowlist stripped 11 V2 runtime
files from the archive (approved card fronts, scroll-world scenes, collection hero and
scene motion, opening product media, presentation registry, scene blueprints, hero
commerce scenes, founder-selected placeholders, font provenance, editor about page), so
the swap deleted them from the live theme; Home and collection routes rendered without
their film for a few minutes. Restored from the staged source, cache flushed, filesystem
parity re-verified 539/539 with zero hash mismatches. Two further V1-only preflights
(`SKYYROSE_VERSION` regex, V1 asset floor) were patched the same session; the script is
now V2-aware (commit `b3cc98845`). Lesson recorded: diff the archive's file set against
the live manifest before the swap.

**Parity.** `verify-home-derived-output.mjs` against staging: contract 16,168 bytes in
head byte-equal to the build; inline bootstrap after the hero with the Boost ignore
attribute, no footer copy; four font preloads with `@font-face`-equal hrefs; all 12
served theme CSS/JS files sha256-equal to the build; bootstrap index 19 before the first
classic body script at 35. Boost's stale block is still printed (31,177 bytes,
informational).

**Measurement** (Chromium, 3 samples per width, cache-busted so every document is an
uncached render; medians relative to document TTFB so platform latency is separated
from in-page scheduling):

| Width | Doc TTFB before → after | FCP after doc | Controller init after doc | Film request after doc | First film frame after doc | CLS |
|---|---|---|---|---|---|---|
| 320 | 2340 → 1763 | 114 → 187 | 130 → 159 | 142 → 182 | 257 → 377 | 1.008 → 0.000 |
| 390 | 2645 → 1884 | 173 → 133 | 171 → 96 | 179 → 131 | 477 → 231 | 1.033 → 0.000 |
| 414 | 2027 → 1892 | 177 → 196 | 184 → 159 | 212 → 193 | 299 → 285 | 1.003 → 0.000 |
| 768 | 2030 → 1675 | 758 → 138 | 828 → 101 | 841 → 132 | 1731 → 222 | 1.017 → 0.000 |
| 1440 | 2402 → 1795 | 258 → 227 | 281 → 196 | 305 → 219 | 1297 → 358 | 1.004 → 0.000 |

CLS is 0.000 in all 15 samples; hero, CTA and brand-mark geometry are identical at first
paint and settled. The controller now initialises before or at first paint in every
sample; the BEFORE outliers (768: 828 ms, 1440: 281 ms with first frame at 1.3–1.7 s)
are gone. Document TTFB differences are server-side variance between runs, not an
effect of the theme. The absolute BEFORE figures quoted earlier in this report
(controller 2.2–2.9 s) were dominated by that uncached TTFB; the in-page table above
is the honest comparison.

**Policies and commerce** (`verify-home-policies.mjs`): 27/28 in Chromium and 27/28 in
WebKit. Every functional check passes: poster-first continuity, reveal after a frame,
pause/play, hidden-tab pause/resume, reduced motion, Save-Data, no-JS coherence from the
inline contract, rotating mark, Ask Skyy launcher with no 3D request, Quick View → size →
native variation → Add to Bag (bag 0→1), Search, Bag, Menu. WebKit again denied the
unattended autoplay and the Play control started the film. The single failing check is
the "no console errors" assertion during the commerce journey: Chromium logs
"Permissions policy violation: payment is not allowed" from Stripe's express-checkout
script, WebKit logs Stripe keepalive/`r.stripe.com` access-control noise. The
`permissions-policy: … payment=()` header was already on staging before the deploy and
`inc/security.php` is untouched by this repair `[live]` `[repo]`. Pre-existing and
outside this scope, but worth a decision: with `payment=()` the Stripe express checkout
(Apple Pay / Google Pay buttons) cannot use the Payment Request API on staging.

**Film delivery after deploy** (6 cold samples per width, fresh context each, `[live]`):

| Width | Request start after nav (median, range) | TTFB | Transfer end | loadstart → first frame |
|---|---|---|---|---|
| 390 | 1887 ms (1636–2119) | 37 ms (26–91) | 80 ms (61–130) | 98 ms (75–116) |
| 1440 | 1926 ms (1741–2092) | 31 ms (26–34) | 67 ms (57–177) | 78 ms (66–175) |

All 12 samples HTTP 206, 676,790 bytes, TLS 1.3, edge HIT. Delivery is unchanged from
BEFORE (TTFB 28 ms, transfer ≈80 ms) as expected for a platform-owned variable; the
request start still tracks the uncached document TTFB. First frame after loadstart
improved from 150/126 ms to 98/78 ms because the controller is no longer competing with
the classic script chain when the film arrives. A cold-edge MISS was not observed.

**Still open.** Real Safari eyes-on on staging (Playwright WebKit denies the unattended
autoplay on both builds and shows the poster with a working Play control). Optional:
regenerate Jetpack Boost's critical CSS from the Boost admin so its informational block
matches the current header; Home no longer depends on it.

## Post-merge addendum `[live]` — 2026-09-08

- **PR #920 merged** into `codex/v2-completion-pr-20260906` (merge commit `4ad080c9a`, head `7a1572a05`, 5/5 checks). The PR agent resolved 11 conflicts against the base branch, which had moved 50 commits past the staging pin.
- **Regression found and fixed (bug-326).** Printing the controller inline directly after the hero meant it ran before the Scroll World rail below was parsed, so the rail's prev/next controls stayed hidden on Home. Confirmed live on staging (`.sr2-recovery-controls` hidden), absent on the pre-repair mirror. Fix: rails bind once the document is parsed (`initRails` on `DOMContentLoaded` when still loading, immediately otherwise), commit `7c7ae5171`; `verify-home-policies.mjs` now asserts the rail controls.
- **Staging hotfix** (founder-approved): the pinned controller plus that 5-line fix, minified with the theme's terser settings, replaced `assets/js/visual-recovery.js` and `.min.js` in the live staging theme (sha256 `9e545f20…` / `e35ee720…`); the previous files are kept on the host at `/tmp/sr2-hotfix-backup/`. Validated first on an exact replica of staging's 539 files (rail controls visible, parity PASS, 28/28 policies), then live: rail controls visible, plain Home re-primed at the edge with the fixed inline bootstrap, derived-output parity PASS against the hotfix set, policies 27/28 (the one failure is the pre-existing Stripe `payment=()` console noise).
- Staging therefore runs the pinned tree + the repair + this two-file hotfix; the merged completion branch carries the base's newer controller with the same fix. A later deploy of the completion branch supersedes the hotfix.
- Also logged: bug-327 (V2 `.min` outputs hidden by the repo ignore rule), bug-328 (fail-closed gate inside a process substitution), bug-329 (stop-gate hook false positives on the deploy script name).
