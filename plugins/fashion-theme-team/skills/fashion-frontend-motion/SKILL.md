---
name: fashion-frontend-motion
description:
  Implements SkyyRose PHP/CSS/JS storefront surfaces, responsive composition,
  immersive interactions, WebGL/360 fallbacks, and generated asset parity.
---

# Fashion Frontend and Motion

Use after design-system and commerce contracts are approved and implementation
ownership is assigned.

## Owners and inputs

Route `fashion-frontend-engineer`, `fashion-motion-responsive-engineer`,
`brand-experience-architect`, and `woocommerce-theme-engineer`. Read the
approved component/token contract, V2 motion contract, page plan, asset SOT, and
theme build instructions.

## Procedure

1. Implement the assigned route with canonical tokens/components and stable
   page/section/feature IDs. Keep the garment and CTA reachable in every state.
2. Cover 390/768/1440 layouts, long copy, loading, empty, error, unavailable,
   slow network, decode failure, context loss, GPU timeout, offline, and route
   abort states.
3. Use poster-first, capability-gated, lazy WebGL/360 enhancement. At 390px,
   never initialize WebGL before an explicit opt-in; reduced motion removes RAF,
   autoplay, parallax, and auto-rotate while preserving DOM order and
   reachability.
4. Enforce CSS/JS/font/image budgets, keyboard/touch parity, cancellation, focus
   restoration, and overflow checks. Rebuild committed `.min` assets.
5. Return changed files, build output, responsive captures, interaction trace,
   fallback trace, and schema-valid HTML/JSON handoff.

## Verification

Run theme build, min-drift, browser console/network checks, responsive captures,
reduced-motion checks, and performance budgets. A visual claim without eyes-on
evidence remains `BLOCKED`; frontend authors never self-approve.

## Boundaries

Do not deploy, upload media, add unapproved dependencies, or override brand/SOT
truth to make a render look complete.
