# Gate 2.5 — Pre-order truth

Fresh staging plugin/gateway audit is in preorder-audit.json. No dedicated pre-order engine was identified. Stripe is enabled with testmode=no and capture=yes. WooPayments is enabled; no sandbox authorization exists. Product flags (_skyyrose_is_preorder=yes, _is_preorder=1, edition size=250) do not establish reservations, delayed capture or physical per-size allocation. Products retain native instock/backorders=no semantics. Payment was not submitted and no gateway configuration changed.

Removed unsupported reservation/security-of-allocation and fulfillment-date assurances from home, PDP, pre-order templates and service FAQ copy. The PDP explicitly states that this label does not reserve stock or defer payment; shipping estimates require Client Services. Registry flags and catalog selection remain presentation only. Scene links use neutral view/choose-options wording.

Full production build and V2 verification PASS. This change does not implement a transactional pre-order engine or certify a shipping schedule.

Staging browser: /pre-order/ includes the explicit standard-checkout boundary and no former secured-place/reservation/date promise; SG-005 PDP renders the same truthful note. Exact six-file writes and hashes are in preorder-staging-receipts.json.
