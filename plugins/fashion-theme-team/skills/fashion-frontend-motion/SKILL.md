---
name: fashion-frontend-motion
description: Implements SkyyRose PHP/CSS/JS surfaces, responsive composition, immersive interactions, WebGL/360 fallbacks, and generated asset parity.
---

# Fashion Frontend and Motion

Use after design-system and commerce contracts are approved.

## Owners and inputs

Route `fashion-frontend-engineer`, `fashion-motion-responsive-engineer`, brand
experience, and WooCommerce. Read approved tokens/components, V2 motion,
page plan, asset SOT, and theme build instructions.

## Procedure

1. Implement assigned routes with canonical tokens/components and stable page,
   section, and feature IDs. Keep garment and CTA reachable in every state.
2. Cover 390/768/1440, long copy, loading, empty, error, unavailable, slow
   network, decode failure, context loss, GPU timeout, offline, and abort.
3. Use poster-first, capability-gated, lazy WebGL/360 enhancement. At 390px do
   not initialize WebGL before explicit opt-in; reduced motion removes RAF,
   autoplay, parallax, and auto-rotate while preserving DOM reachability.
4. Enforce CSS/JS/font/image budgets, keyboard/touch parity, cancellation,
   focus restoration, and overflow checks. Rebuild committed `.min` assets.
5. Return changed files, build output, captures, traces, and HTML/JSON handoff.

## Verification

Run theme build, min-drift, browser console/network, responsive, reduced-motion,
fallback, and performance checks. Visual claims without eyes-on evidence are
`BLOCKED`.

## Boundaries

No deployment, upload, unapproved dependency, or brand/SOT override.
