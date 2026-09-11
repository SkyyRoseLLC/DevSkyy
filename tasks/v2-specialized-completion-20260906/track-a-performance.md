# Track A — frontend performance experiment

**Status: NEEDS_MORE_WORK. Decision: reject broad theme-CSS inlining for integration.** The accepted readiness source is unchanged. All four tested mobile routes still fail the unchanged 2500 ms LCP gate. This bounded experiment identifies an application-owned delivery cost but does not earn visual/release certification or justify replacing the accepted stylesheet architecture.

## Ownership and frozen input

Accepted source digest: `4a23629d6b17c1d0393000cfbbe982eedf47712837dd772a66c993ad7e46bbe7`. Every file in the readiness source-integrity manifest was independently rehashed after profiling: zero differences. The readiness evidence remains immutable. Cart and Checkout were excluded from the experiment's route predicate and were not tested or modified. No model, artwork, content, native Woo dependency, staging or hosting setting was changed.

The inspected readiness traces identify Home's hero VIDEO, Shop's portal-frame IMG, Collection's arrival IMG and PDP's native-gallery IMG as distinct LCP candidates. Existing source preloads Home/Collection/PDP responsive imagery; Shop's portal image is eager/high priority. The candidate preserves all of those behaviors. Current Home has 13 external theme stylesheets, including theme.min.css at 54,404 transferred bytes. This supports testing theme-owned stylesheet delivery without assuming every blocking millisecond belongs to CSS or removing native dependencies.

The scratch router at `.artifacts/v2-specialized-completion-20260906/track-a/inline-theme-router.php` uses port 18427 with the same corrected theme-root isolation as accepted port 18416. Its `style_loader_tag` filter replaces only `skyyrose2-*` stylesheet links on the four selected routes with identical CSS contents in the same cascade order and media condition. Relative CSS URLs are rewritten to their prior theme directory. Native WooCommerce/Core/plugin styles remain external and ordered as before. This is a deliberately broad diagnostic upper bound, not production implementation. It was authorized as an isolated local experiment; no theme source integration was requested.

## Comparable measurements

Raw reports, HTML reports, trace JSON and DevTools logs are under `track-a/baseline` and `track-a/inline`. `track-a/metrics.json` binds selected report hashes, full configuration, measured quantities and observed LCP insight. Lighthouse 13.4.1 uses the existing helper: simulated mobile 390×844, DPR1, 150ms RTT, 1638.4Kbps throughput, CPU×4, fresh browser context. The local origins use equivalent plain transport; this comparison is not gzip/Brotli/CDN proof. Each cell is one sample, not a median or a causal effect size.

| Route | Baseline LCP ms | Inline LCP ms | Delta ms | Baseline FCP ms | Inline FCP ms | LCP gate |
|---|---:|---:|---:|---:|---:|---|
| Home | 5436 | 5118 | -318 | 4225 | 3460 | FAIL |
| Shop | 5945 | 5414 | -532 | 3914 | 3609 | FAIL |
| Collection | 4211 | 4057 | -154 | 3308 | 3157 | FAIL |
| PDP | 5427 | 5114 | -313 | 4066 | 3764 | FAIL |

| Route | Baseline HTML bytes | Inline HTML bytes | Baseline CSS bytes | Inline external CSS bytes | Baseline total bytes | Inline total bytes |
|---|---:|---:|---:|---:|---:|---:|
| Home | 97,753 | 244,030 | 150,598 | 0 | 1,581,262 | 1,576,941 |
| Shop | 146,241 | 272,348 | 221,635 | 92,285 | 1,042,168 | 1,038,925 |
| Collection | 133,883 | 267,442 | 137,881 | 0 | 1,631,420 | 1,627,098 |
| PDP | 115,853 | 246,722 | 267,295 | 132,969 | 1,183,341 | 1,174,314 |

Wire totals are the Lighthouse audit window, not a separately measured load+10-second budget window. Inline CSS is counted in HTML; zero external CSS must not be represented as zero stylesheet cost. All eight selected reports retain CLS below 0.1. Their exact values, request counts and TBT are in metrics.json. No field INP, physical mobile GPU or peak-memory conclusion is made.

Home improves sampled first paint more than LCP. Shop shows the largest sampled LCP reduction, while Collection changes only154ms. Small total-wire reductions cannot offset the structural cost of moving126–146KB into each HTML response and losing independently cacheable stylesheet delivery. Repeated-navigation/cache cost is an architectural consequence, not a measured warm-cache regression in this run. An application cache or server compression change must not be invented to hide it. Home remains a VIDEO candidate; we did not change its activation or assert a new video-eligibility cause. Observed LCP subparts remain distinct from Lantern simulated LCP and are retained in metrics.json; they must not be added to the table's simulated values.

## Exclusions and reproducibility

The first helper invocation used a package without Playwright and failed before a browser started. The corrected dependency is `/Users/theceo/.codex/worktrees/19db/DevSkyy/.artifacts/v2-phase3-20260905/qa/package.json`, with Node `/Users/theceo/.hermes/node/bin/node`. Its failure receipts are retained rather than counted as a route result.

The initial screenshot helper stalled on image decoding after the first Home DOM capture. It was stopped and replaced with bounded readiness waits. The first baseline-secondary PDP sample overlapped that process and is explicitly excluded; `lighthouse-baseline-pdp-clean-pdp-mobile.report.json` is the selected serialized rerun. This is a measurement-integrity correction, not a performance gain. No theme changes occurred between runs. Original reports remain in place.

The same helper can reproduce each selected run with `V2_BASE_URL`, `V2_ARTIFACT_DIR`, `V2_QA_PACKAGE` and `V2_LH_CASE` set as recorded in the artifact labels. The scratch router serves existing accepted files; its lifetime is local to the experiment and it is not a deployment artifact.

## Next application-owned intervention

Do not integrate this all-inline experiment. Next isolate the genuinely above-fold portion of theme.min.css and route shell CSS using mobile/desktop rule coverage across Home, Shop and Collection, plus menu/search/Quick View and no-JavaScript states. Test a small, explicit critical-style budget while retaining cacheable remaining styles and their dependency order; do not preload or inline every stylesheet. This is a proposed next experiment, not measured improvement or authorization to trim unproven rules. Shop's retained native general stylesheet remains necessary for pagination/Quick View; PDP native variation/gallery behavior remains authoritative.

A narrower candidate must produce repeatable same-profile improvement, unchanged native interactions and independent visual EQUIVALENT/BETTER review before root considers integration. No current optimization is recommended for source promotion. Hosting transport remains Track B's environment-bound responsibility, and media/model/content remain protected under their separate owners.

## Bounded visual evidence and closeout

The corrected capture helper completed once and closed its browser. Twenty-four PNGs cover Home, Shop and Collection at390/1440, both baseline and inline, immediately after DOMContentLoaded and after bounded image/font readiness plus1200ms. Reduced motion is explicit. All six settled pairs are **byte-identical**, and all six compared visible DOM geometry/font/color/background records are equal. All12 page cases report zero page errors, fonts loaded and no horizontal overflow. `visual-comparison.json` contains the exact comparison outcomes; `capture-results.json` retains DOM records. This proves equality in those sampled states, not every dynamic dialog or unvisited scroll position.

Five of six earlier DOMContentLoaded pairs are also byte-identical. Collection390 differs in the early text raster presentation around “Cross the water. Find the origin.” and converges to the identical settled capture. The early difference was viewed and is retained; no exact readiness cause is asserted. DOMContentLoaded screenshots are not a continuous filmstrip and cannot establish absence of every flash of unstyled content. The viewed Home mobile pair and Shop/Collection desktop frames preserve composition and native controls. Root separately inspected the Home390 settled pair; final independent visual certification of a promotable candidate remains outside this experiment because no candidate is being promoted.

Track A is complete as a bounded diagnostic and remains **NEEDS_MORE_WORK** against the user's performance objective. The broad inline variant is rejected for integration; raw evidence is retained. Browser profiling and capture have stopped. Accepted source hashes remain unchanged, and the temporary tool expansion ends with this handoff.
