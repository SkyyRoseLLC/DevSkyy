# Staging-only V2 deployment evidence contract

I will deploy the current commit to the configured **staging** environment only, using this repository's existing staging deployment procedure. I will not invoke a production deployment command, change a production target, or treat a local build as deployment proof.

Before upload, I will record the exact candidate commit and prove that it contains both requested V2 runtime scopes:

- product-card framing source plus the built stylesheet/runtime artifact it loads;
- animated-hero markup, controller/runtime binding, and its media/fallback asset path.

The deployment report will identify the exact files included, the resulting staging URL, and the local/source and remote active SHA-256 hashes for each deployed theme asset. A deployment is incomplete if either the hero controller binding or the product-card framing implementation is absent from the candidate.

After cache propagation, I will run a fresh browser verification against the staging URL at both desktop and a 390 px viewport. The verification will include:

| Required check | Desktop | 390 px mobile |
| --- | --- | --- |
| Verified product imagery loads from the expected URLs | Yes | Yes |
| Stone product-card frame and crop composition are intact | Yes | Yes |
| Pointer hover / keyboard focus reel and variant disclosure work | Yes | Keyboard focus; touch-safe equivalent checked |
| Reduced-motion / no-autoplay fallback is visible and usable | Yes | Yes |
| Animated hero starts when supported | Yes | Yes |
| Hero poster or static fallback appears when motion is disabled or unavailable | Yes | Yes |
| Network console contains no failed image, video, script, stylesheet, or font request in scope | Yes | Yes |

The final report will include browser captures for both viewport sizes, the viewport dimensions, timestamps, active deployed hashes, image and animation load results, and a list of every failed resource with URL, HTTP status, initiating asset, and whether it blocks the release. If an expected asset does not load, a crop differs from the requested framing, or the fallback cannot be demonstrated, I will report that as a staging blocker and will not represent the deploy as verified.

I will preserve unrelated work in the current checkout. No production URL, production credentials, or production deployment action is within this request.
