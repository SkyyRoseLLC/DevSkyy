---
name: fashion-design-system
description:
  Builds and governs SkyyRose tokens, typography, primitives, component APIs,
  states, responsive rules, motion contracts, fixtures, and adoption evidence.
---

# Fashion Design System

Use before broad storefront implementation or when a component, token, state, or
route pattern is missing.

## Owners and inputs

Route the inner pod through `fashion-design-system-engineer`,
`fashion-token-foundations-engineer`, `fashion-component-commerce-engineer`,
`fashion-motion-responsive-engineer`, `fashion-accessibility-content-engineer`,
and `fashion-designops-governance-engineer`. Read the design-system contract,
existing token sources, component census, V2 motion/responsive contract, and
official platform guidance when APIs are involved.

## Procedure

1. Census existing tokens, selectors, primitives, components, variants, and
   generated outputs before creating anything.
2. Build the primitive → semantic → component token graph with aliases, modes,
   contrast, forced-color, RTL, deprecation, and transform rules.
3. Define anatomy, slots, variants, invalid combinations, all interaction and
   commerce states, responsive behavior, reduced-motion equivalence, and
   accessibility hooks.
4. Generate deterministic fixtures for content extremes, viewports, themes,
   states, and collection overrides. Record adoption and migration impact.
5. Return a versioned conformance contract plus candidate-bound evidence.

## Verification

Reject raw literals, parallel token systems, ungoverned breakpoints, and
permanent exceptions. Run schema, contrast, fixture, generated-parity, and
adoption checks. The independent visual red team—not the author—owns the pixel
verdict.

## Boundaries

This skill governs the reusable system. It does not self-approve visuals,
deploy, or silently rewrite existing canonical sources.
