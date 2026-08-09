---
name: fashion-visual-commerce-qa
description:
  Independently tests SkyyRose visual fidelity, responsive states, browser
  interactions, imagery provenance, and complete WooCommerce customer journeys.
---

# Fashion Visual and Commerce QA

Use only on an identified integrated candidate. Keep the reviewer independent
from the author and builder.

## Owners and inputs

Route `fashion-visual-qa-red-team`, `visual-commerce-qa`, and
`accessibility-performance-reviewer`. Read the candidate manifest, approved
creative contract, page plan, commerce contract, SOT image registry, and fresh
build metadata.

## Procedure

1. Capture the route matrix at 390/768/1440 and critical loading, empty, error,
   success, unavailable, reduced-motion, keyboard, and media-failure states.
2. Run eyes-on anti-generic, logo-off recognition, crop/focal, typography,
   contrast, overflow, touch, focus, and screenshot-diff checks.
3. Exercise product discovery, search, filters, variable selection, add to bag,
   cart, coupons, shipping/tax, guest/auth checkout, payment recovery, order
   confirmation, and account journeys where applicable.
4. Tie every visual/product assertion to candidate ID, route, viewport, state,
   screenshot/trace hash, source/SKU, and reviewer identity.
5. Reject stale, mixed-candidate, filename-only, skipped, or narrative-only
   evidence. Return `PASS`, `REJECT`, or `BLOCKED` with exact findings.

## Verification

Browser evidence must be fresh and eyes-on. A failed preview, unavailable
staging site, broken asset, console/network error, or missing route is a gate
failure—not a pass with a note.

## Boundaries

QA is read-only. It never changes code, catalog data, live WooCommerce, or
deployment state.
