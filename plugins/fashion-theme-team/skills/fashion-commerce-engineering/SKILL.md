---
name: fashion-commerce-engineering
description:
  Implements catalog truth, WooCommerce templates, product discovery, fit and
  returns contracts, secure customer journeys, and commerce-state behavior.
---

# Fashion Commerce Engineering

Use for catalog-backed implementation across collection, search, PDP, cart,
checkout, account, order, fit, returns, and service surfaces.

## Owners and inputs

Route `catalog-sot-integrator`, `woocommerce-theme-engineer`,
`fashion-component-commerce-engineer`,
`fashion-merchandising-conversion-architect`, and
`fashion-product-fit-returns-specialist`. Read the CSV/dossiers/image registry,
`commerce-state-contract.json`, WooCommerce coverage, page blueprints, and
current official WooCommerce/WordPress documentation.

## Procedure

1. Map SKU, variation, price, availability, imagery, provenance, fit, policy,
   and route IDs to authoritative records before editing templates.
2. Implement product cards, filters, search, PDP decision zones, variation
   selection, cart, checkout, account, order, fit, returns, and extension-absent
   states through canonical contracts.
3. Preserve server truth: authorization, HPOS-safe reads, nonces, escaping,
   prepared queries, capability checks, safe redirects, privacy, and idempotent
   actions. Client state never authorizes purchase.
4. Exercise simple, variable, unavailable, sale, sparse, long-content, guest,
   auth, failure, and recovery journeys with real approved fixtures.
5. Return source changes, state matrix, commerce-truth log, and HTML/JSON
   handoff. Rebuild tracked theme assets after CSS/JS changes.

## Verification

Run catalog consistency, provenance/image checks, PHP/theme lint, WooCommerce
template-version checks, browser purchase journeys, and generated parity. Never
invent products, scarcity, reviews, fit promises, prices, or policies.

## Boundaries

No external catalog mutation, media upload, payment action, or deployment
without explicit approval. Merchandising hypotheses remain experiments.
