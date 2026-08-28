---
name: fashion-a11y-performance
description: Reviews and hardens WCAG, inclusive content, keyboard/focus, reduced motion, localization, fallbacks, Core Web Vitals, and storefront performance.
---

# Fashion Accessibility and Performance

Use during design-system acceptance, implementation hardening, and independent
candidate review.

## Owners and inputs

Route `fashion-accessibility-content-engineer` for implementation guidance and
`accessibility-performance-reviewer` for independent verification. Read route,
state, motion, performance, and current W3C/WAI contracts.

## Procedure

1. Check landmarks, headings, names, labels, errors, contrast, non-color cues,
   zoom/reflow, forced colors, touch targets, localization, RTL, and screen
   reader output.
2. Exercise keyboard order, focus trap/restore, Escape, skip links, forms,
   validation, loading/empty/error/success/unavailable, and payment recovery.
3. Verify reduced-motion equivalence: no RAF, autoplay audio, parallax,
   auto-rotate, or forbidden WebGL initialization.
4. Measure LCP, CLS, INP, FPS, critical CSS, route JS, fonts, media, network
   errors, and fallback timing against the candidate budget.
5. Return route/state/viewport, command, artifact, severity, evidence scope, and
   disposition for every finding.

## Verification

Use authoritative WCAG/WAI references plus executable browser evidence. Static
lint is not a substitute; missing or skipped evidence is `UNVERIFIED`.

## Boundaries

The reviewer is read-only and cannot repair or approve its own candidate.
