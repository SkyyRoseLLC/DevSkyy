# Gate 2.4 — PDP commerce media

The opening manifest remains unchanged: 16 stale, 9 missing-front, 5 rejected, 3 approved. Woo attachments and gallery IDs are a separate commerce source; approved card crops do not grant editorial approval.

The resolver prefers valid approved editorial attachments, otherwise valid Woo primary/gallery attachments. Explicit authenticity rejection blocks both paths, including schema and variation-image payloads. Missing media uses a compact status. The complete before-summary action fires even without images; only the native image callback is temporarily removed and restored when media is blocked.

Verification: PHP resolver/schema/variation regression test PASS; full V2 build and verify PASS. Staging sentinel callbacks fired once for approved BR-006, stale SG-005, missing-front BR-002 and rejected BR-003; native image hook and scoped filters restored. Browser confirmed responsive, loaded Woo images for the first three, no gallery for rejected BR-003. Browser viewport at this point is 525px; 390px validation remains Gate 2.8 work. See media-browser.json, media-hook-probe.json and media-staging-receipts.json. No assets, catalog authority or approval states changed.
