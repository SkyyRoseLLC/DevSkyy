---
name: fashion-visual-commerce-qa
description: Independently tests SkyyRose visual fidelity, responsive states, browser interactions, imagery provenance, and complete WooCommerce customer journeys.
---

# Fashion Visual and Commerce QA

Use only on an identified integrated candidate. Keep the reviewer independent
from the author and builder.

## Owners and inputs

Route `fashion-visual-qa-red-team`, `visual-commerce-qa`, and
`accessibility-performance-reviewer`. Read candidate manifest, creative,
page, commerce, image SOT, and build metadata.

## Procedure

1. Capture 390/768/1440 plus loading, empty, error, success, unavailable,
   reduced-motion, keyboard, and media-failure states.
2. Run eyes-on anti-generic, logo-off recognition, crop/focal, type, contrast,
   overflow, touch, focus, and screenshot-diff checks.
   For collection-scene imagery, independently score product fidelity, optical
   integration, anatomy/pose, collection story, and commerce readiness against
   the hash-bound `product-fidelity-native-review.v1` receipt. Floating feet,
   mismatched camera/light/floor/reflections, pasted edges, wrong product details,
   or repeated scene storytelling are hard failures.
3. Exercise search, filters, variable selection, add-to-bag, cart, coupons,
   shipping/tax, guest/auth checkout, payment recovery, order, and account.
4. Tie every assertion to candidate ID, route, viewport, state, artifact hash,
   source/SKU, and reviewer identity.
5. Reject stale, mixed-candidate, filename-only, skipped, or narrative-only
   evidence. Return `PASS`, `REJECT`, or `BLOCKED`.

## Verification

Browser evidence must be fresh and eyes-on. A failed preview, unavailable
staging site, broken asset, console/network error, or missing route is a gate
failure—not a pass with a note.

## Boundaries

QA is read-only and never changes code, catalog, live WooCommerce, or deployment.

## End-to-end execution

Run `fashion-e2e-task-execution` with this skill. Record the candidate hash,
route, viewport, state, SKU/provenance, eyes-on captures, journey traces, and
reviewer. Preserve every defect as a first-seen issue; do not let a later pass
erase an earlier candidate-bound failure.
