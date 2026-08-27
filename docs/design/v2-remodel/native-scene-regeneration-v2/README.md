# V2 Native Commerce Scene Candidate Handoff — 2026-08-25

## Scope and candidate boundary

This handoff prepares exactly seven **local, candidate-only** native-commerce
scene contracts: `BR-COMMERCE-1`, `BR-COMMERCE-2`, `LH-COMMERCE-2`,
`LH-COMMERCE-3`, `SIG-COMMERCE-1`, `SIG-COMMERCE-2`, and `SIG-COMMERCE-3`.

It does not contain generated images, a model request, a remote upload, runtime
wiring, a staged asset, a release decision, or founder approval. The two
founder-approved scenes, `BR-COMMERCE-3` and `LH-COMMERCE-1`, are explicitly
out of scope and must not be regenerated or used as generation references.

All local authority inputs are recorded in [source-ledger.json](source-ledger.json)
with the current product-SOT hash. Each scene contract carries its own source
hashes, exact SKU and CTA mapping, product-technique locks, planned optical
contract, independent-review rubric, and promotion boundary.

## Current verdict: BLOCKED — preparation complete, generation not permitted

The prepared contracts are intentionally blocked before any provider call:

1. The current `data/product-sot.json` hash is
   `4b5799ab3e16eaf75dd3c37321339eb861176bb53e161fd18263c35bed78c9e2`.
   All preparation artifacts now bind that value; the supplied `4ccfbe…`
   handoff is retained only as stale historical evidence.
2. `gpt-image-2` now validates against the local registry's lower-level
   `scene_generation` capability. The native-scene contract is a local policy
   layer over that operation; no provider availability probe or paid request
   has been made. See [model-capability-receipt.json](model-capability-receipt.json).
   Gemini is not selected.
3. The exact native route still requires a genuine protected garment matte for every
   product in a scene and an approved on-model target. The necessary mats and,
   for several multi-product looks, a same-pose source do not presently exist.
   Each specific gap is stated in its scene contract and preflight receipt.
4. Several multi-product scenes still lack a founder-approved same-pose source.
   A packshot, ghost asset, or a rejected protected composite cannot be relabeled
   as that authority.

`BLOCKED` is not a failure to preserve the scene plan. It is the required
fail-closed state until a founder-approved source/matte route and a registry-
validated GPT Image 2 adapter are available.

## Artifact map

- `phase-ledger.json` — durable Fashion Theme Team phase state and authority boundary.
- `source-ledger.json` — current SOT lock and allowed/reference-forbidden inputs.
- `model-capability-receipt.json` — machine-readable local registry result; no live probe was made.
- `scene-contracts/*.json` — one `product-fidelity-edit.v1` native-scene contract per unfinished scene.
- `preflight-receipts/*.json` — fail-closed fidelity-gate results bound to the contracts at this baseline.
- `collections/*/{contract.json,evidence.json,preview.html}` — structured three-scene collection review handoffs. They deliberately use text status cards, not fabricated scene imagery or contact sheets.

## Required authorization and re-entry sequence

1. Re-hash `data/product-sot.json` and every bound source. Any drift supersedes this handoff.
2. Before a provider call, validate the selected GPT Image 2 adapter's
   reference-input handling and current account availability. This is distinct
   from the current local registry match and remains a paid-action boundary.
3. Produce or approve same-pose source and protected-garment mattes, inspect
   alpha, bind their hashes, and update only the affected contract.
4. Re-measure the final source-camera geometry at full size; no prompt is
   executable until the provisional geometry in the contract is confirmed.
5. Run the fidelity preflight and optimize steps again. Only then may a human
   decide whether to authorize a paid generation call.
6. After a candidate exists, create a hash-bound **independent**
   `product-fidelity-native-review.v1` receipt. Minimum scores are product
   fidelity 95; optical integration, anatomy/pose, collection story, and
   commerce readiness 90. Any hard fail blocks the candidate.
7. Stop with `founder_status: PENDING`, `wiring_allowed: false`, and
   `deployment_allowed: false`. Founder approval of a named candidate hash is
   a separate later decision; wiring, staging, deployment, and commercial
   release remain separate.

## Quarantine rules

Never use `founder-review-batch-v3`, prior rejected composites, placeholder
assets, or `*-natural-placement-v2` images as product authority, scene
authority, a generation reference, or a wireable asset. A generated candidate
may be added only below a new candidate-specific path after the preceding
steps pass; it must never overwrite a source authority.
