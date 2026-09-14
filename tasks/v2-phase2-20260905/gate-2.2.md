# Gate 2.2 — Native WooCommerce account

Status: repaired and verified anonymously; authenticated flows not exercised.

Root cause: WooCommerce's configured account page contained starter placeholder
paragraphs. The classic theme already calls the_content(), so the native shortcode
is the correct content path. No authentication or theme rendering code changed.

`tools/v2-runtime/account-migration.php` resolves wc_get_page_id('myaccount'),
requires the exact staging host and V2 theme, defaults to dry-run, and checks the
reviewed content hash before writing. Dry-run exported original content, slug,
template and resolved ID. Apply changed only post_content to
`[woocommerce_my_account]`; native WordPress revision/modified metadata may update.
The second apply reported changed=false. No environment-specific ID enters theme
logic. Account registration remains disabled as configured; guest checkout remains
enabled. No user accounts, credentials, order data or endpoint options changed.

Rollback: invoke the same migration with mode=rollback, expected_sha256 equal to
the apply target_sha256, and rollback_content equal to dry-run before_content.
The content guard rejects intervening edits. Resolve the account page again;
confirm its ID matches the exported record before applying a rollback.

Verified: logged-out native form, Woo login nonce, one deliberately invalid
nonexistent username rejected with native focused error alert, lost-password form,
orders/edit-account/edit-address/payment-methods routing to native login while
anonymous. Form and login button remain within 390px viewport; document width
390 at mobile and 1440 at desktop. Screenshots retained under
`.artifacts/v2-phase2-20260905/account-{before,after-390,after-1440}.png`.
No password-reset email was sent. No authorized customer credentials were
available; dashboard contents, actual order details, address/payment-method edits,
account edits and authenticated logout are NOT VERIFIED. The native Woo rendering
path owns these operations; no custom replacement was introduced.

PHP syntax passed. Theme source and build artifacts are unchanged by this gate.
