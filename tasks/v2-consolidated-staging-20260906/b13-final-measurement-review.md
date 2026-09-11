# Final bounded B13 measurement review

PASS. Independent read-only verification; no network requests or source changes.

Recomputed totals from the first snapshot's raw layout-shift entries in all 15 session files. Every machine-summary `allEventSum` equals the sum of every recorded event; every `rawSum` equals the sum after `hadRecentInput` exclusion. The maximum session-window CLS remains distinct. A synthetic test including a recent-input shift also passed: all events 1.0, filtered sum 0.1, CLS 0.1.

Examples: normal Home allEventSum 1.0317950460027614 versus filtered sum/CLS 0.0002019142219834028; desktop Home all three totals 0.84501953125. The report explicitly retains excluded initial shifts and does not equate diagnostic all-event sums with CLS. The top-level machine descriptor correctly defines all three fields.

Verified exactly four `skyy-explicit-character-intent` sessions: normal/moderate Home and normal/moderate Home re-entry. The desktop Home control has no such action. The report's four-intent-session and untouched-desktop wording matches these machine records.

This approval covers the described measurement amendments and consistency checks, not site performance approval. Home delivery failure, mobile uncertainty and physical-device/field limitations remain stated in the B13 report.
