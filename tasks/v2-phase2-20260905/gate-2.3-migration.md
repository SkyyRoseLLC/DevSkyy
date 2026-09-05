# Gate 2.3 — Product-model migration evidence

Migration portion completed; browser certification remains in progress.

All 33 canonical SKUs were reconciled. 31 garments now use variable products,
with 185 native child variation IDs. One-size accessories lh-005 and sg-007 remain
simple. Noncanonical drafts br-012-legacy and sg-004 were not changed. LH-006's
incorrect One Size options were replaced by canonical S, M, L, XL, 2XL, 3XL.

Every parent ID/SKU/slug/URL, purchase price, media attachment/gallery/category
relationship and stock/backorder setting was checked against the exported
preimage. All retain unmanaged stock, null quantity, in-stock, no backorders;
children likewise receive no invented quantity. Existing parent availability is
preserved, not newly asserted physical per-size stock. Native Woo moves regular
and sale prices to children; its variable product sync clears those parent meta
copies and maintains the aggregate purchase price. Every child price matches
the prior parent. The source CSV and historical frozen sync projection were not
changed or used to overwrite live values.

The full product export includes post/meta/taxonomy state and counts of existing
order-item references. All affected parent reference counts were zero. Backup
hashes and durable paths are recorded; migration snapshots are also committed.
Each product's original snapshot and newly created IDs are retained in staging
option skyyrose_v2_phase2_variations_{ID}, autoload disabled. Transactions require
InnoDB core/product tables; preimage hashes reject concurrent drift. SG-005 was
migrated first, then explicitly passed a no-op second apply and rollback dry-run.
The remaining products were processed sequentially using reviewed preimage hashes.

Active product webhooks target www.devskyy.app. The migration suppresses their
scheduling with woocommerce_webhook_should_deliver (verified in installed Woo
source to run BEFORE delivery scheduling), and blocks immediate outbound HTTP
within the migration process. It does not disable webhooks persistently. General
third-party asynchronous plugin side effects have not been comprehensively
certified; no claim is made about arbitrary external systems. No orders/payments
were created by these tests.

Rollback mode verifies journal/plan/current hash and refuses if any parent/child
order references exist. It deletes only the recorded children belonging to that
parent, restores native simple type, original attributes/defaults, prices and
stock settings, and removes that journal. Rollback dry-run exports this exact
representation without writing. Other parent fields remain untouched. Do not
run rollback after intervening catalog edits without reconciliation.

Native WC_Cart tests accepted all 185 variations and verified product ID,
variation ID, canonical size, quantity 1 and original price, then emptied the
isolated test cart. HTTP negative tests left carts empty for missing size, invalid
variation ID and invalid size. Browser SG-005 S/M/L tests confirmed selected-size
cart names/prices, one item after a rapid double-click, cart refresh persistence,
and mobile 390px. Crewneck S also passed; a fast subsequent selection encountered
a hidden-ID readiness discrepancy still under investigation. Browser coverage
for the remaining garment categories is not yet complete. This is NOT a final
Gate 2.3 or Phase 2 certification.
