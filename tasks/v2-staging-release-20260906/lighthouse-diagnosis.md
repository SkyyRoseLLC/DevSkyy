# Home initial layout stability — observed FAIL, diagnosis only

**The accepted staging Home has a measured initial layout shift. Initial-load stability must not be described as clean based on the earlier settled feature-parity checks or mobile B13's recent-input-filtered value.** This review read existing local artifacts and accepted source only. No browser, source edit, platform change, regeneration or purge was performed.

## Direct measurements

| Evidence | Recorded result |
|---|---|
| Lighthouse Home | Performance 49; reported and observed CLS **1.0**. |
| Lighthouse LCP | **5434.052ms simulated** LCP; **2751ms observed** LCP. `throttlingMethod` is `simulate`. CLS is separately recorded as an observed layout metric, not inferred from the simulated LCP. |
| Lighthouse largest shifted node | `main#primary > div.sr2-archive__inner > section#sr2-archive-arrival > div.sr2-archive-scene__image`; score1.0, reported bounds top64, width412, height837. |
| B13 normal mobile | At316.8ms, a value1.0 event moves the hero from `(0,0,390,844)` to `(0,64,390,780)`. Initial raw shift sum1.048855; all initial events have `hadRecentInput=true`, giving a filtered raw sum0. |
| B13 moderate mobile | Initial raw shift sum1.062665; non-recent-input raw sum0.031001. The first event is again value1.0 with recent input. |
| B13 desktop | At306.7ms, value0.840972 moves the hero from `(0,0,1440,900)` to `(0,76,1440,824)`, with **`hadRecentInput=false`**. Initial raw and unexcluded sums are both0.846951. This independently establishes a substantial unexcluded initial shift. |

B13's first recorded explicit character action occurs at12807ms on normal mobile and18703.4ms on desktop, well after these shifts. Why the early mobile events have the recent-input flag is not established by the available artifacts; do not attribute them to the later character action. Raw sums above are identified as event sums, not silently substituted for a differently windowed metric.

## Concrete geometry and delivery evidence

The theme owns the required settled geometry:

- `assets/css/home-page.css:2` positions the scene and sets its minimum height to `100svh - var(--sr2-header)`.
- `home-page.css:3` positions the hero image wrapper absolutely with `inset:0`.
- `home-page.css:30` adds `.sr2-archive { padding-top:var(--sr2-header) }`.
- `assets/css/theme.css:236` sets the mobile header variable to64px; the desktop header is76px.
- `assets/css/global-shell.css:5` defines the current header grid and height.

Those64/76px offsets match the observed changes. Header brand/end elements also acquire their current dimensions and positions in the same initial events. This supports late application of current layout rules as the mechanism to investigate; the node attribution alone does not prove that one padding declaration accounts for the entire score.

The **optimizer owns critical coverage and delivery**. Recovered Home HTML contains:

- Inline style ID **`jetpack-boost-critical-css`**,31177 bytes, SHA-256 `9ec38548b78344db9d72aeddf1f0f6234348d14f5e33412f5d6ba048c0f6a07b`.
- No current `.sr2-archive`, `.sr2-archive-scene`, `.sr2-archive-scene__image`, `.sr2-house-header` or `.sr2-house-header__end` rules in that payload.
- Combined stylesheet ID **`all-css-6f6f5118a3fb9fb371f459160964e162`**, URL `/_jb_static/??27276687c6`, delivered with `data-media="all"`, `media="not all"` and an onload handler that activates it. A separate stylesheet reference is the no-script fallback.

The observed decoded combined CSS contains the current scene rule at offset156656, archive padding at159568, and header rule at182822. B13 records this stylesheet finishing at229.456ms, before the316.8ms mobile shift; the exact application/recalculation timestamp is not recorded here.

Root's `execution/critical-css-old-current.json` additionally shows the old stale Home and recovered accepted Home carry **the identical31177-byte critical CSS hash**, despite HTML recovery. That proves unchanged and inadequate critical coverage across those captured documents. It does not establish whether the platform reused a stale cache entry, regenerated identical output, or applied another configuration/coverage policy.

## Disposition and next action

**Strong likely contributor:** current theme geometry is absent from the managed critical artifact and becomes available through the deferred full stylesheet. This matches the observed header-offset correction. Responsibility is partitioned: the theme defines necessary geometry; WordPress.com/Jetpack generates and delivers the critical subset and deferral policy. This is not yet a controlled causal demonstration or a blanket theme defect.

The inspected visual-recovery code changes video readiness/opacity and animates the inner picture; it does not directly assign the outer image wrapper's64/76px offset. This makes a direct video-readiness offset explanation unsupported by the code inspected. Additional font/text reflows are present, but they do not erase the large initial geometry event.

Investigate managed critical-generation coverage and delivery/cache lineage for the current Home and header first. Obtain controlled staging evidence before deciding whether any theme change is justified. **Do not insert platform workarounds into the theme, change configuration, regenerate critical CSS or purge caches as part of this read-only diagnosis.**

Settled feature/visual parity remains a separate bounded result. It cannot clear initial CLS, simulated mobile LCP, physical-device or field-performance gates. Exact evidence hashes, event records and ownership distinctions are in `lighthouse-diagnosis.json`.
