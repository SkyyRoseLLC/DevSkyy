---
name: fashion-strategy-brain
description: Turns fashion-commerce strategy, research, page architecture, V2 planning, and the Fashion Theme Brain into evidence-bound HTML/JSON work packets.
---

# Fashion Strategy Brain

Use before design or implementation when the storefront must decide what to
sell, say, show, measure, or contain.

## Route

Load only the needed routes from `brain/taxonomy.json`: `fashion_strategy`,
`page_architecture`, `merchandising_conversion`, `fit_imagery_returns`, and
`growth_measurement`. Read the dated source registry, SkyyRose canon, catalog
SOT, and V2 page plan. Classify assertions as `OBSERVED`, `APPROVED`,
`RECOMMENDED`, `UNKNOWN`, `INFERENCE`, or `EXPERIMENT`.

## Procedure

1. Inventory audience, assortment, service promise, launch model, routes, and
   authority. Missing evidence remains `UNKNOWN`.
2. Build the route/job matrix: page purpose, section order, CTA, product truth,
   states, responsive transformation, and measurement question.
3. Preserve V2 page IDs, imagery provenance, CTA state IDs, and responsive/motion
   contracts. Deviations require a reason and owner.
4. Separate durable principles from current research and hypotheses. Never turn a
   trend or competitor pattern into a brand rule.
5. Return SkyyRose-branded `preview.html`, schema-valid `contract.json`, and
   `evidence.json` with stable IDs and source IDs.

## Verification

Run `jq -e .`, the relevant schema validator, source-freshness checks, and HTML
ID/JSON ID parity checks. Stale research, invented facts, missing imagery
authority, and unresolved contradictions remain `BLOCKED`.

## Boundaries

Plans and routes only: no theme implementation, catalog mutation, creative
approval, uplift claim, or deployment.

## End-to-end execution

Run `fashion-e2e-task-execution` with this skill. Start a scoped task before
research or planning output is adopted; record source freshness, route matrix,
schema parity, contradictions, and independent review before handing a work
packet to implementation.
