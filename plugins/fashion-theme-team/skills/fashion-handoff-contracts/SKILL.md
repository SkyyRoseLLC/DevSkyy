---
name: fashion-handoff-contracts
description: Produces and validates SkyyRose structured HTML, JSON, and evidence handoffs with stable IDs, schema parity, provenance, and candidate-bound proof.
---

# Fashion Handoff Contracts

Use whenever a role returns a page plan, visual artifact, feature scaffold,
commerce contract, or release evidence.

## Contract

Every handoff contains `preview.html`, `contract.json`, and `evidence.json`.
HTML and JSON share stable page/section/component/feature/state IDs. Evidence
records candidate, source, screenshot/trace, command, hash, reviewer, timestamp,
status, and unresolved gaps. Product facts and imagery reference SOT authority.

## Procedure

1. Select the relevant schema from `brain/schemas/` and load route, commerce,
   motion, and acceptance contracts.
2. Render SkyyRose tokens, fonts, provenance, CTA states, responsive notes, and
   explicit loading/error/unavailable behavior.
3. Validate JSON and schema; compare extracted HTML IDs to JSON IDs; check links,
   assets, and source freshness.
4. Attach fresh candidate-bound browser/image evidence. Keep `UNKNOWN`,
   `BLOCKED`, and `EXPERIMENT` explicit.

## Verification

Mixed candidate hashes, missing rights/source records, fabricated imagery,
unreadable previews, and self-certified claims fail. Missing browser,
accessibility, commerce, or performance artifacts remain `UNVERIFIED`.

## Boundaries

This skill validates handoffs; it does not approve, publish, or waive gates.
