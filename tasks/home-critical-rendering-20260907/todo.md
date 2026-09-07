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
- [ ] no-JS; reduced-motion; Save-Data; visibility; pause/play; commerce
      regression (Quick View → variation → Add to Bag; search; bag; menu) —
      `tools/v2-runtime/verify-home-policies.mjs`
- [x] Boost regeneration test in isolation (real generator) + coverage
      comparison
- [x] Fonts residual-shift audit
- [ ] STOP-AND-SHOW staging deploy → derived-output verification → staging
      AFTER + film delivery samples (≥5 cold)
- [ ] Release checklist update (source → filesystem → critical CSS → bundle →
      browser parity)
- [ ] Independent red-team visual verdict (fashion-visual-qa-red-team)
- [ ] Final report

- [x] Font preloads for the four first-view faces (front page only), parity
      gate in `verify-home-derived-output.mjs`
- [x] `scripts/deploy-theme.sh`: archive root = basename of THEME_DIR (was
      hardcoded `skyyrose-flagship`; a V2 deploy would have stranded the live
      theme directory) — fixed, syntax-checked, not yet exercised
- [ ] Register new theme files in tools/v2-source-certification registries;
      `npm run verify` green; phpcs on changed PHP
- [ ] Commit implementation on the branch
