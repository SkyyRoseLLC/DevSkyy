# Staging release decision — V2 product-card framing and animated hero

**Task:** `STG-V2-f4b13d13-product-card-and-animated-hero`
**Exact scope:** `wordpress-theme/skyyrose-flagship-2` at commit `f4b13d13ccff6858a39059a5f53425cb2cf7d3a9`; target `staging` only; surfaces: V2 home hero and collection product portals. Production is explicitly out of scope.

**Outcome:** **blocked — no upload or deployment performed.**

The requested deployment must not substitute a static hero for the requested animated hero. The current commit does contain the V2 portal/card scope (`template-parts/commerce/product-card.php`, `data/opening-product-media.json`, and the verified preview-media additions), but it does not contain an active V2 home-page binding for the animation controller:

- `assets/js/theme.js` contains an animated-hero controller that selects `[data-hero-video]`.
- `front-page.php` does not render a `[data-hero-video]` element or matching `source[data-src]` for that controller.
- The repository's `scripts/deploy-theme.sh` identifies itself as a **production** deploy script. It is not a staging procedure and must not be repurposed for this task.

Deploying this input would make the visual acceptance request untestable and could falsely report the poster/static hero as animation evidence. I therefore recorded the required first-seen issue below and stopped before any external side effect.

## Required controlled task record before staging

Create a `staging_deploy` task before the staging upload, with this immutable scope and evidence contract:

```json
{
  "theme": "skyyrose-flagship-2",
  "ref": "f4b13d13ccff6858a39059a5f53425cb2cf7d3a9",
  "target": "staging",
  "surfaces": ["v2-home-hero", "v2-collection-product-portals"],
  "viewports": ["desktop-1440x1024", "mobile-390x844"],
  "motion": ["no-preference", "reduced"],
  "production": false
}
```

The task cannot be completed without all of the following evidence, recorded as file paths and hashes or structured summaries—never credentials or image bytes:

| Evidence | Acceptance requirement |
| --- | --- |
| `source_hash` | `f4b13d13ccff6858a39059a5f53425cb2cf7d3a9` plus SHA-256 values for V2 `front-page.php`, `product-card.php`, `assets/css/theme.min.css`, and `assets/js/theme.min.js`. |
| `hero_runtime_binding` | Source proof that the V2 home markup renders the element consumed by the runtime controller, including its poster and deferred video sources. |
| `deployment_receipt` | Staging-specific deployment identifier, exact uploaded theme hash, target host, UTC timestamp, and cache-purge receipt. No production host or production credentials. |
| `desktop_capture` | Fresh browser capture at `1440×1024` of the V2 hero and portal card, after a cache-busting navigation. |
| `mobile_capture` | Fresh browser capture at `390×844`, touch/coarse-pointer emulation, after a cache-busting navigation. |
| `motion_capture` | Desktop capture/state report showing the animated hero actually reaches `loadeddata`/playing and is not merely showing the poster. |
| `reduced_motion_capture` | `prefers-reduced-motion: reduce` capture proving no autoplay/loop motion and an intentional static poster/fallback. |
| `fallback_capture` | Fresh capture after intentionally blocking the video response, proving the poster remains legible and the page remains usable. |
| `network_report` | Per-viewport request log with URL path, resource type, status, transfer result, and all failures (`requestfailed`, HTTP `>=400`, page errors, console errors). |
| `card_framing_qa` | Desktop/mobile review proving the stone/statue frame is present, product media is fully within the intended crop, and hover/focus reel disclosure is readable; mobile/touch must remain static and usable. |

## First-seen issues to append to the ledger

| Stable fingerprint | Category | First observation | Release effect |
| --- | --- | --- | --- |
| `v2-animated-hero-runtime-unbound-v1` | `release_scope` | The V2 controller queries `[data-hero-video]`, but the V2 front-page markup has no such element/source pair. | Blocks the requested animated-hero release and any claim of hero motion/fallback QA. |
| `v2-staging-procedure-unresolved-v1` | `deployment_control` | The available named theme deploy script is production-targeted; no verified staging-only upload entrypoint was established for this exact V2 input. | Blocks upload until the configured staging procedure and target are identified. |

If either appears again, append an `issue_seen_again` event using the same fingerprint; do not create a new vague issue, retry the upload, or mark the task complete.

## Authorized next release sequence

1. Bind the animated-hero markup to the existing V2 controller and rebuild the committed minified assets. Record the new commit/hash rather than silently changing this task's source identity.
2. Run the V2 candidate source gate and a PHP/JavaScript build check against that exact commit. Attach their reports to the task.
3. Start the controlled staging task and attach the source and runtime-binding evidence **before** using the verified staging-only deployment procedure.
4. Deploy to staging only; record the receipt and cache purge. If any deployment or cache discrepancy occurs, open an issue before another upload.
5. In a real browser, collect the two fresh viewport captures, reduced-motion capture, forced-video-failure fallback capture, and complete network report. Treat any 4xx/5xx, failed resource, page error, console error, crop regression, or missing motion state as a ledger issue.
6. A named reviewer may complete the task only after every declared evidence item is present and both issues are resolved with the corrective commit/receipt. This does not authorize production.

## Evidence status now

| Evidence | Status |
| --- | --- |
| Source commit identified | Present: `f4b13d13ccff6858a39059a5f53425cb2cf7d3a9` |
| Source/minified asset hashes | Partially inspected; CSS `b1dbc012066d0c687743abe34e6cd21578c6890929abeb48649c47b69c06efb4`, JS `0164258263fdc45fa5b65039ddfce5bfca31337e638ae9ed455d6be79aa23fed` |
| V2 candidate source gate | Pass: `wordpress-theme/skyyrose-flagship-2/scripts/verify-v2-candidate.sh` |
| Active hero runtime binding | Missing; issue open |
| Staging deployment receipt / URL / active deployed hash | Missing; no deployment attempted |
| Desktop, 390px, reduced-motion, and fallback captures | Missing; not fabricable from local source inspection |
| Network failures report | Missing; requires the post-deploy browser run |

**Release decision:** staging upload is not authorized for this commit until the two open issues are resolved and the evidence contract can be fulfilled. Production remains untouched.
