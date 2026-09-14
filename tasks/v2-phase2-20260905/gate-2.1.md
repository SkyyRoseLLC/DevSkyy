# Gate 2.1 — jQuery execution order

Status: PASS for the tested staging routes. Phase 2 overall remains in progress.

Baseline: `ddb3dce6db82d69010f8d15fee4fbb6b99661066`, V2 2.4.4.

## Reproduction and cause

Homepage repeatedly emitted deferred core followed by blocking Migrate and
Jetpack Boost `_jb_static/??8405d66706`; both consumers raised jQuery ReferenceError.
Core registrations belong to WordPress `wp_default_scripts`: jquery is an alias
for core plus Migrate; Migrate itself has no direct core dependency. Woo registers
wc-jquery-blockui with jquery dependency; add-to-cart/woocommerce also depend on it.
The actual scripts object is Jetpack Boost `Concatenate_JS`. Earlier recovered
optimizer source evidence is in the baseline reconciliation report; fresh registry
and final emitted tags are retained alongside this record. Page Optimize's blocking
concat behavior and Boost's exclusion/movement interact with the theme's isolated
core defer request. This is not evidence that the theme is the only optimizer layer.
LiteSpeed's inspected JS defer option is 0.

## Minimal controlled repair

Removed only the homepage jquery-core strategy callback in inc/performance.php.
Theme-owned defer handling is unchanged. No optimizer plugin/settings changed.
The regression test first failed with the callback present, then passed after removal;
it tests front/non-front actions and confirms owned scripts retain defer.

Staging preimage exactly matched the certified SHA-256. Replacement used strict
staging hostname/theme guards, before/after hashes, a same-directory temporary
file and atomic rename. Local exact preimage is retained at
`.artifacts/v2-phase2-20260905/performance.before.php`. The write receipt records
both hashes. Restore only that preimage after checking the current hash equals
the recorded repaired hash; invalidate OPcache and flush staging page/object cache.
No product/database/theme-version changes were part of this repair.

Fresh query requests immediately emitted blocking core before Migrate and the
same Boost bundle, without the exceptions. Canonical homepage initially retained
stale HTML and reproduced both exceptions. Staging wp_cache_flush returned true;
subsequent canonical homepage/reload emitted the repaired order. This cache step
is necessary to reach ordinary visitors, not a plugin-configuration change.

## Verification

Production build and full V2 verify passed (PHP, source hashes, 33 product fronts,
16/9/5/3 opening states, scene casts/motion, registry/projection, translations,
asset parity, marketplace, performance, SEO). No compiled asset changes resulted.
Only the performance.php runtime hash was advanced; checker wording now describes
hash-bound approved runtime rather than requiring the original known defect.
Historical commits and all other PHP/content/media locks remain intact.

Browser evidence includes fresh query, canonical and reload navigations. After
cache flush, 14 canonical/reload visits across Home, Shop, Signature, SG-005 PDP,
Cart, populated Checkout shell and My Account produced zero new console errors
and exactly one identified jquery-core payload, blocking before consumers.
Menu opened; navigation Search opened and focused its searchbox, and closed.
Native SG-005 Add to cart once produced count 1, quantity 1 and $25 subtotal in
cart and checkout. The smoke-test item was removed afterward. No order/payment
was submitted. The product remains simple pending Gate 2.3, so this is not size
selection certification. Account content remains broken pending Gate 2.2.

Raw browser log retention includes pre-purge failures intentionally; read mode
`canonical-after-purge` and `reload-after-purge` for final acceptance results.
These are route smoke checks, not the later complete responsive/network/a11y gate.

The read-only strategy control explicitly restored the original defer request
inside a single CLI process after loading the current runtime. WordPress's
`get_eligible_loading_strategy` calculated blocking core in both cases, while
Boost's observed HTML had honored the requested defer before the repair. Thus
WordPress's dependency-safe calculation alone did not protect the optimized
emission path. Removing the owned request fixes that disagreement at its source.
The probe retains Woo's defer eligibility, and does not change persisted settings.
