# Home critical rendering + cinematic startup repair — 2026-09-07

Branch `codex/v2-home-critical-rendering-20260907` (worktree
`/Users/theceo/.claude/worktrees/home-critical-20260907`). Baseline commit
`233e9990d` = byte mirror of the tree staging served at 2026-09-07 01:45 UTC.

- [x] Pin deployed candidate as git baseline (7116 worktree → staging parity:
      536/536 deployed files match)
- [x] Diagnose Boost critical CSS identity (post 10476 `cornerstone_d41d8cd9`,
      2026-09-01, pre house-header)
- [x] Isolated local fixture: `127.0.0.1:18330`, DB `homecrit`, theme symlink →
      this worktree
- [x] Mirror Boost 4.7.0 + stale critical CSS locally (BEFORE reproduction)
- [x] Measurement harness (CLS/FCP/LCP/first-paint+settled shots/film timings)
      at 320/390/414/768/1440
- [x] BEFORE run (local Boost mirror) + staging BEFORE (live)
- [x] Critical structural contract: `assets/css/critical/home.min.css` built
      from source, contract-verified, inlined on Home
- [x] Hero bootstrap: inline the unchanged controller after the hero, ignored by
      Boost's defer, footer copy dropped on Home
- [x] Rotating mark container geometry in the contract
- [x] Blocking-chain map (registered deps vs delivered order)
- [x] AFTER run local (Chromium 5 widths ×3, WebKit 390/1440 ×2)
- [x] no-JS; reduced-motion; Save-Data; visibility; pause/play; commerce
      regression (Quick View → variation → Add to Bag; search; bag; menu) —
      `tools/v2-runtime/verify-home-policies.mjs`
- [x] Boost regeneration test in isolation (real generator) + coverage
      comparison
- [x] Fonts residual-shift audit
- [x] STOP-AND-SHOW staging deploy → derived-output verification → staging
      AFTER + film delivery samples (≥5 cold)
- [x] Release checklist update (source → filesystem → critical CSS → bundle →
      browser parity)
- [x] Independent red-team visual verdict (fashion-visual-qa-red-team)
- [x] Final report

- [x] Font preloads for the four first-view faces (front page only), parity
      gate in `verify-home-derived-output.mjs`
- [x] `scripts/deploy-theme.sh`: archive root = basename of THEME_DIR (was
      hardcoded `skyyrose-flagship`; a V2 deploy would have stranded the live
      theme directory) — fixed, syntax-checked, not yet exercised
- [x] Register new theme files in tools/v2-source-certification registries;
      `npm run verify` green; phpcs on changed PHP
- [x] Commit implementation on the branch
- [x] Policies + commerce regression: 28/28 Chromium, 28/28 WebKit (engine
      autoplay denial path exercised; BEFORE denies identically in plain WebKit)
- [x] `npm run verify` with the pinned toolchain: every gate green except the
      pre-existing `check:commerce-scenes` runtime-PHP baseline (13 unrelated
      files already drifted at the deployed baseline) and `test-pdp-gallery.php`
      needing `V2_WP_FIXTURE` (passes against the local WP 7.1 / WC 11.1.0 root)
- [x] Committed 5f807df3b on `codex/v2-home-critical-rendering-20260907`
- [x] Cold film-delivery samples on staging BEFORE (done: 12 samples, request 1.8–2.4 s after nav, TTFB ≈28 ms, transfer ≈85 ms, all edge HIT)
- [x] Staging deploy 2026-09-07 14:15Z (founder y): exact staged source 539
      files (staging manifest + 3 new), hot-swap OK, `verify-home-derived-output`
      PASS live, plain Home edge cache re-primed with the new head within ~1 min
- [x] bug-325: V1 data/ allowlist stripped 11 V2 runtime files in the swap;
      restored within minutes, 539/539 parity; deploy script made V2-aware
      (b3cc98845)
- [x] Staging AFTER: measurement ×5 widths, policies Chromium+WebKit, cold film
      delivery ×12 — done: CLS 0 ×15, policies 27/28 both engines (pre-existing Stripe console noise), film TTFB ≈30 ms / first frame ≈80–100 ms after loadstart
- [ ] Real Safari eyes-on (open; Playwright WebKit denies unattended autoplay
      on both builds)
