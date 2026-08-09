---
name: fashion-handoff-contracts
description:
  Produces and validates SkyyRose structured HTML, JSON, and evidence handoffs
  with stable IDs, schema parity, provenance, and candidate-bound proof.
---

# Fashion Handoff Contracts

Use whenever a role returns a page plan, visual artifact, feature scaffold,
commerce contract, or release evidence.

## Contract

Every handoff contains:

- `preview.html`: branded, readable, responsive visual artifact;
- `contract.json`: machine-readable page/section/component/feature/state
  contract;
- `evidence.json`: candidate ID, source IDs, screenshots/traces, commands,
  hashes, reviewer, timestamp, status, and unresolved gaps.

The same stable IDs must appear in HTML and JSON. Product facts and imagery must
reference catalog/SOT authority. A prose-only handoff fails.

## Procedure

1. Select the relevant schema in `brain/schemas/` and load the route/page,
   commerce, motion, and acceptance contracts.
2. Render the artifact with SkyyRose tokens, fonts, imagery provenance, CTA
   states, responsive notes, and explicit loading/error/unavailable behavior.
3. Validate JSON with `jq` and the schema validator; extract HTML IDs and
   compare them to JSON IDs. Check links, asset references, and source
   freshness.
4. Attach candidate-bound browser/image evidence. Keep `UNKNOWN`, `BLOCKED`, and
   `EXPERIMENT` explicit; never upgrade them through prose.
5. Return the three files plus a short change/evidence index for the lead.

## Verification

No mixed candidate hashes, missing source/rights records, fabricated imagery,
unreadable preview, or self-certified visual claim can pass. Evidence remains
`UNVERIFIED` when a required browser, accessibility, commerce, or performance
artifact is absent.

## Boundaries

This skill validates the handoff contract; it does not approve the theme,
publish artifacts, or waive a gate.
