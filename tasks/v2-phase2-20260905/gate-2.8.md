# Gate 2.8 — Mobile ownership and reachable commerce

Reproduced chat and navigation simultaneously open at 390px; chat occupied x214–374/y509–756. Navigation did not isolate background focus. Also found search page excerpts exposing builder CSS, long-query text overflow, and missing fixed-header clearance.

Repair: menu makes main/footer/guide inert while preserving original inert state, focuses first link, wraps Tab within the header and restores Escape focus. Native dialog beforetoggle closes the menu and other dialogs; native modal trapping/Escape remain. Dialogs own scroll lock, bounded viewport height and safe-area clearance. Search returns focus to the visible menu trigger instead of its hidden opener. Quick view and size guide retain native close/focus lifecycle. No cart overlay exists; cart remains a page. Photo gallery remains Woo's own viewer.

Skyy's mobile controls stay in normal flow before the footer, with no proactive walk-on or translating animation. Chat cannot cover headlines or commerce CTAs; menu/dialog states hide the guide. Delayed minimize focus returns only if the user has not moved focus elsewhere. Desktop mascot art/behavior remains apart from modal coordination.

PASS: 10 Node regressions (including menu isolation/wrap, dialog exclusivity and delayed focus), PHP regressions, build/full verify. Browser at actual 390px: menu and search focus/Escape/scroll lock, size guide, quick view, guide open/minimize, guide hidden/inert during navigation. Native SG-005 S purchase once -> cart correct name, size S, quantity1, subtotal25, width/scroll390 -> removed/empty. Browser-control transient navigation timeouts were resolved by reading state, without repeating purchase. Long-query wrapping and header clearance repaired. Evidence mobile-overlay-browser.json and screenshots under .artifacts/v2-phase2-20260905.

The historical mascot.min.js preimage matched recovery commit bb0e6eecf0af54b3f0c14e5eadb7b138530ce09b. The guard stopped before overwriting; provenance was reconciled, prior successful writes read back, and each remaining scoped write hash-verified. See mobile-staging-receipts.json.
