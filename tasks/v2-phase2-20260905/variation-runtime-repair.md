# CSP-safe native variation rendering and readiness

During real variable-product tests, a fast selection sometimes retained hidden
variation_id=0 despite the theme announcing Selection confirmed. A later trace
exposed the cause: WooCommerce's variation template helper delegated to
wp.template/Underscore, whose new Function compilation violates the existing CSP.
The actual DOM template was the native three-field Woo template (not a custom
CommerceKit template). Installed Woo 11.1.0 add-to-cart-variation.js lines 1238–1275
claims a CSP-safe fast path, but its hard-template regex returns true even for
`{{{ data.variation.price_html }}}`. This was reproduced separately in Node.

The theme now fills WordPress's existing memoized template cache for exactly
variation-template and unavailable-variation-template after wp-util loads. It
interpolates only the native description/price/availability HTML fields, preserves
escaped-field behavior and localized unavailable markup, and rejects unsupported
expressions. It does not replace wp.template or modify vendor files, product
resolution, variation IDs, prices, inventory, forms or server validation. CSP
headers remain unchanged; no unsafe-eval permission was added.

The status adapter now waits for show_variation plus a matching native hidden ID.
Zero-ID/unready submissions and repeated in-flight submissions are prevented;
pageshow restores the button after browser back/forward restoration. No selected
ID is invented or written by the adapter. The original found_variation event can
precede complete rendering, so it no longer means a confirmed purchase selection.

Six Node regression tests pass, including execution with string/wasm code
generation disabled. The three readiness tests failed before the repair. Tests
are now part of npm run verify. Production build and full V2 verify pass.
Source/minified theme JS were deployed with hash guards; staging source matched
all three historical baselines, while its minified file matched recovered
bb0e6eecf0af54b3f0c14e5eadb7b138530ce09b (the certification rebuild had normalized
it locally). Both identified preimages were retained before replacement.
The PHP inline-template addition likewise replaced its exact current preimage.
Cache was flushed on staging; no optimizer setting or production file changed.

Post-repair browser tests passed hoodie/jacket/shorts/bomber/sweatpants/set/kids
and SG-005 size selections, additions, correct native IDs where read, quantity 1,
price/subtotal, and removal. New error logs after the CSP fix are empty. Earlier
CSP failures remain in the evidence rather than being erased. Browser-tool
navigation/context timeouts were recovered by inspecting completed cart state,
without repeating purchases. The first apparent timing-only explanation was
superseded by the CSP exception trace; readiness guards alone were not sufficient.

Commerce browser evidence contains actual observed layout widths. Several nominal
390px-emulated runs expanded to 525 CSS px; these are functional size/cart passes,
NOT mobile-layout certification. Gate 2.8 must address/verify that separately.
The final source certification and full responsive matrix remain outstanding.
