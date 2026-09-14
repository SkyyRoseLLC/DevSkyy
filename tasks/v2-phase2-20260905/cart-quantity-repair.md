# Native cart quantity control repair

Discovered during Gate 2.3 native variation cart tests: the quantity label was
visible, but its input was absent from both the DOM and accessibility tree.
A staging PHP reproduction confirmed Woo's native output contains an input and
wp_kses_post() removes it. The custom template incorrectly treated trusted native
form markup as post content. It now calls woocommerce_quantity_input() directly;
WooCommerce owns output escaping, input naming and quantity constraints.

Validation: PHP syntax, production build and full V2 verify passed. An initial
verify correctly flagged stale POT line references; build regenerated them and
the next full verify passed. Only POT source-line references changed, no strings
or visual styles. Staging PHP replaced its exact certified preimage atomically;
receipt and local preimage retained. The POT upload guard rejected staging's
historical differing POT preimage; readback confirmed its unchanged hash
0dfa80de920f47273a9ec847ebe467f5184a6cd90760da8e5800221ded3a1e22.
POT is a translation extraction artifact, not executed runtime; no overwrite
was attempted after that rejection. The rebuilt canonical POT is committed locally.

Browser: SG-005 / size M at $25, native Product quantity appeared as 1. Editing
to 2 and clicking Update bag yielded native Cart updated notice, bag count 2,
quantity 2 and $50 subtotal/total. Raw before/after snapshots are retained in
.artifacts/v2-phase2-20260905/. No payment/order was submitted.

Rollback: restore exact prior cart.php bytes from git parent or retained preimage,
only after current hash matches the staging receipt. No database migration is
required for this template repair.
