---
name: fashion-strategy-brain
description:
  Turns fashion-commerce strategy, research, page architecture, V2 planning, and
  the Fashion Theme Brain into evidence-bound HTML/JSON work packets. Use before
  design or implementation.
---

# Fashion Strategy Brain

Use this skill when the request concerns what the storefront should sell, say,
show, measure, or contain before code is changed.

## Route

Load `fashion_strategy`, `page_architecture`, `merchandising_conversion`,
`fit_imagery_returns`, and `growth_measurement` only when needed from
`skills/fashion-theme-team/brain/taxonomy.json`. Read the dated source registry,
SkyyRose canon, catalog SOT, and the V2 page plan. Classify every assertion as
`OBSERVED`, `APPROVED`, `RECOMMENDED`, `UNKNOWN`, `INFERENCE`, or `EXPERIMENT`.

## Procedure

1. Inventory audience, assortment, service promise, launch model, routes, and
   available evidence. Missing authority stays `UNKNOWN`.
2. Build the route/job matrix: customer job, page purpose, section order, CTA,
   product truth, states, responsive transformation, and measurement question.
3. For V2, preserve the 28-page plan, imagery provenance, page IDs, CTA state
   IDs, and responsive/motion contracts. Deviations require a reason and owner.
4. Separate durable principles from current research and hypotheses. Never turn
   a trend or competitor pattern into a brand rule.
5. Hand off `preview.html`, schema-valid `contract.json`, and `evidence.json`
   with stable page/section/feature IDs and source IDs.

## Verification

Run `jq -e .` on every JSON artifact, validate against the relevant schema,
check source freshness, and confirm the HTML IDs and JSON IDs are identical.
Unverified claims, stale research, missing imagery authority, and invented
commercial facts remain `BLOCKED`.

## Boundaries

This skill plans and routes. It does not implement theme code, alter catalog
truth, approve creative, declare conversion uplift, or deploy.
