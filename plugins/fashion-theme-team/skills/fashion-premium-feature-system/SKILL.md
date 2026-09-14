---
name: fashion-premium-feature-system
description: Executes and governs every premium storefront feature in the SkyyRose feature catalog, from landing-page immersion and merchandising controls through WooCommerce states, accessibility, security, performance, and release proof.
---

# Fashion Premium Feature System

Use this skill when a request names a premium theme feature, asks for feature
parity with a reference theme, or requires the Fashion Theme Team to build a
production-ready feature set. The complete 44-item inventory is the machine
contract at `brain/features/premium-feature-catalog.json`; this skill defines
how every item becomes an implementation, review, and release artifact.

## Enterprise operating rule

Every feature is a versioned capability, not a visual promise. Before editing,
load its catalog record, canonical page/section IDs, SOT dependencies, state
matrix, security boundary, performance budget, and evidence requirements. A
feature is not complete until implementation, fallback, accessibility, browser
proof, commerce proof where applicable, and release evidence all point to the
same candidate hash. Missing or stale proof is `UNVERIFIED` or `BLOCKED`.

## Capability routing

| Capability family                                                                 | Primary skill/owners                                                                              | Required handoff                                                                        |
| --------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Theme settings, typography, backgrounds, borders, glow, layout                    | `fashion-design-system`, `fashion-brand-experience`; token and typography owners                  | token map, settings schema, preview, contrast and responsive evidence                   |
| Landing page, password gate, countdown, collector, waitlist, preload, media       | `fashion-frontend-motion`, `fashion-commerce-engineering`; frontend, growth, accessibility owners | route/state contract, consent/security review, slow-network and reduced-motion captures |
| Catalog, quick add, stock, sold-out, pre-order, product navigation, filters, cart | `fashion-commerce-engineering`; catalog, WooCommerce, merchandising, fit/returns owners           | SOT mapping, server-truth action contract, purchase journey evidence                    |
| 3D/GLB, background video, GIF/image, second background, music, cursor, animation  | `fashion-frontend-motion`; motion/responsive and brand owners                                     | capability gate, poster/fallback, input/focus contract, performance and GPU evidence    |
| Lookbook and high-conversion composition                                          | `fashion-brand-experience`, `fashion-strategy-brain`; brand, merchandising, analytics owners      | imagery provenance, CTA/experiment contract, eyes-on visual and measurement evidence    |
| Release, marketplace packaging, customization safety                              | `fashion-handoff-contracts`, `fashion-a11y-performance`, `fashion-release-evidence`               | schema-valid HTML/JSON/evidence packet and PASS/FAIL/BLOCKED gate                       |

## Build procedure

1. Select feature IDs from the catalog; never infer coverage from a screenshot.
2. Map each feature to a page, section, component, CTA, state, and owner. Add
   loading, empty, error, unavailable, success, keyboard, reduced-motion, and
   no-JavaScript behavior where relevant.
3. Define the customization surface with typed defaults, validation, migration,
   reset, preview, capability detection, and safe server-side persistence.
4. Implement with canonical SkyyRose tokens and SOT-backed products/media. Use
   progressive enhancement for GLB, video, GIF, music, cursor, and animation;
   poster/DOM fallbacks must remain useful and purchasable.
5. Protect customer and admin boundaries: sanitize settings, escape output,
   verify nonces/capabilities, use prepared queries, constrain remote URLs,
   validate uploads, obtain consent for email/waitlist, and never trust client
   price, stock, preorder, shipping, or countdown state.
6. Budget every interaction for LCP/CLS/INP, route JavaScript, media bytes,
   font loading, GPU/CPU work, battery, and 390/768/1440 behavior. Respect
   `prefers-reduced-motion`, keyboard/touch parity, RTL, zoom, forced colors,
   and localization.
7. Test real and degraded journeys: first visit, returning visitor, slow/offline
   network, blocked media, unsupported WebGL, context loss, expired countdown,
   invalid email, sold-out/preorder, guest/auth checkout, and recovery.
8. Return `preview.html`, `contract.json`, and `evidence.json` with stable IDs,
   source/provenance references, candidate hash, commands, screenshots/traces,
   reviewer, and unresolved gaps. A prose feature list is not a handoff.
9. Run independent visual, accessibility, commerce, security, and performance
   review. Builders cannot approve their own feature. Recompute candidate and
   invalidate evidence after every mutation.

## Feature-specific hard gates

- **3D, media, music, cursor, glow, and animation:** explicit opt-in where
  needed, poster/fallback, no autoplay audio, reduced-motion equivalence,
  cancellation, focus safety, and measured budget.
- **Email collector, newsletter, waitlist, social, and external countdown:**
  consent and privacy copy, validation, rate limiting, CSRF protection, provider
  failure state, same-origin/CSP-safe URL handling, and no fabricated status.
- **Quick add, stock, preorder, sold-out hiding, filters, cart drawer, and
  previous/next:** server-authoritative WooCommerce data, variation/quantity
  correctness, keyboard announcements, unavailable/retry states, and purchase
  journey proof.
- **Custom settings, fonts, backgrounds, borders, gradients, glow, mobile
  overrides, and landing-page modes:** schema validation, safe defaults,
  contrast/readability, reset/migration, sanitization, cache invalidation, and
  no unbounded CSS/remote asset injection.
- **High conversion rate, easy to use, minimal, eye-catching:** classify as
  design intent or `EXPERIMENT`; never claim uplift without candidate-bound
  instrumentation, metric, guardrail, sample, and rollback evidence.

## Verification result

The skill may report only `PASS`, `FAIL`, or `BLOCKED` for a candidate. `PASS`
means ready for founder review, never permission to deploy. The catalog remains
`governed-not-implemented` until the actual theme candidate supplies evidence
for the selected records.

## End-to-end execution

Run `fashion-e2e-task-execution` with this skill for every selected feature ID.
Bind the state matrix, fallback, security, accessibility, performance, commerce,
and release proof to one candidate; record the first observed missing state or
broken journey before a further implementation or release attempt.
