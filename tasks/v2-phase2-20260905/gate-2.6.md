# Gate 2.6 — Payment and acknowledgement truth

Reproduced unsafe failed-payment assurance and missing failure gateway hooks with a failing real-template regression. Staging Woo native thankyou template calls gateway and generic thankyou hooks outside its failed/nonfailed branch. woocommerce_order_details_table is already registered at priority 10; the V2 manual order-details include duplicated it.

Repair: seven status-specific messages (failed, pending, on-hold, processing, completed, cancelled, refunded), conservative unknown/no-order fallback, native retry URL for failed/pending, all three native lifecycle hooks once for every known order. Removed the manual duplicate details include. No claim about charges or refund arrival timing.

PASS: local seven-state regression, full build/verify, staging render with seven unsaved WC_Order objects (ID 0 throughout), sentinel hook counts once each and no unsafe assurance. No persisted order, customer data, provider redirect, payment attempt or charge was created. Gateway redirects and real settlement remain untested.
