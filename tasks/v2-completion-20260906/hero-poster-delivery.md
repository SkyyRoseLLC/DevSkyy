# Responsive hero poster delivery and native failure recovery

The three complete measurements of PR `191c2fa61` recorded Home mobile LCP at 2.191, 2.216 and 2.978 seconds. All three product-page and Shop measurements passed their 2.5-second budget. The Home failure remains recorded; the first two passes did not establish repeatability.

The third Home trace identifies the native video poster as the later paint, at 779 ms, before the film request begins at 790 ms. The approved responsive image was already available. Publishing the same poster inside the image decode continuation created a second independently painted surface. The correction publishes the browser-selected `currentSrc` at attachment or image load when `naturalWidth` is available. Decode settlement still gates film loading and playback. It never substitutes the desktop `img.src`, republishes an unchanged poster, changes the approved film, delays a benchmark, hides content, or changes Lighthouse settings.

Browser fault testing also exposed native source exhaustion: when WebM and MP4 both return 404, Chrome can emit errors on the two `<source>` elements while leaving the video error null and `play()` pending. The controller now tracks activated candidates, preserves MP4 fallback after a WebM failure, deduplicates source errors, and performs terminal cleanup only after all candidates fail. Cleanup pauses the failed video, clears pending state and removes readiness classes. Late playback completion cannot revive it. The loaded original image remains underneath the existing 700 ms fade.

Root commit `1c2d5d42a` contains only the controller source, its rebuilt minified output and controller regressions. Final source SHA-256 is `4d4281b86964740ddb6ae73a635528b37f2a90274d91f7b347ea7d588b37ea93`. Independent TypeScript review approved the final source; 14 controller tests, scoped ESLint, syntax and locked project type checking passed. Canonical full build/verification passed at the early-poster checkpoint; the final JavaScript-only exhaustion delta then passed asset rebuilding/freshness and all 99 Node regressions. Unchanged font, frame, PHP and media suites were not needlessly repeated locally for that JavaScript-only delta.

The normal-motion before/after recordings cover Home at 390 and 1440 pixels, Signature at 390 and Love Hurts at 390. An independent reviewer examined all four Home filmstrips and twelve collection stills. Hero identity, crop, text, CTAs and approved film sources remain intact, with no new visible blank/black hero interval after first page paint in the sampled evidence. Both versions contain initial white page frames. Sampling at 10 fps cannot exclude sub-100 ms flashes, and the review does not claim pixel-identical motion or a controlled timing improvement.

Final actual Metal browser fault evidence passes eight cases:

| Case | Verified result |
| --- | --- |
| Home 390 and 1440 | Selected native poster matches responsive image; one poster URL/request; film advances; pause/resume works |
| Reduced motion / Save-Data / no JavaScript | Static image remains available; zero film requests |
| Both film formats return 404 | Native source exhaustion pauses playback, clears readiness, and completes the existing fade to the loaded image |
| Hero image returns 404 | Native film still recovers and plays |
| WebM returns 404 | Browser selects MP4, advances playback, and retains pause/resume |

The final receipts record Apple M5 Metal rendering, served controller/poster hashes matching source, and no unexpected page, console, HTTP or request failures. Explicit injected 404s are recorded separately. Earlier probe attempts remain: default Chromium failed the native-renderer preflight before opening a page; an over-restrictive probe blocked WooCommerce's read-only fragments POST; and a 200 ms opacity assertion sampled the valid existing 700 ms fade before completion. The final probe permits only the exact native fragments endpoint beyond GET/HEAD and awaits actual transition completion without changing application timing.

Evidence lives in `.artifacts/v2-completion-20260906/native-poster-handoff/`: `before/`, `after/`, `fallbacks-final/receipt.json` and `webm-fallback/receipt.json`. The first corrected-root Home trial measured 2.145 seconds, but final acceptance requires repeated measurements of the integrated PR under the matched delivery profile.
