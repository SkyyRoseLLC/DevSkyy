# Media and Native Scene Worktree Unification

## Scope

This worktree starts from `main` at `383b77620` and consolidates the portable
media-intake and native-scene preparation work from:

- `codex/on-model-media-intake-pilot` at `e59546eac`;
- `codex/image2-native-scenes-prep` at `ca50e403e`.

It contains no provider output, runtime scene wiring, upload, promotion, or
deployment change.

## Intake-pilot assessment

The intake worktree is 114 commits divergent from current `main`; merging its
history would overwrite or resurrect old V2 storefront work. Its portable
payload is therefore limited to the following verified items:

| Item | Decision | Reason |
| --- | --- | --- |
| Hash-bound on-model and generation-receipt validators | Included | Independent Python modules and seven focused tests pass. |
| 18 geometry preflight receipts | Included and rebound | Three SKUs (`br-006`, `sg-009`, `sg-013`) times six views; all originals and product hashes match the unified 33-SKU SOT. |
| Geometry candidate runner | Included and repaired | It is dry-run by default, forbids overwrites, and now resolves repository imports when launched directly. |
| BR-004 cloud-sleeve gallery entry removal | Included | The candidate was specifically identified as the wrong Black Rose Signature Hoodie mapping. |
| Broad reconstruction/SOT history (`d3914ae` through `7f981af`) | Excluded | It changes older catalog, dossier, video, and storefront records beyond this consolidation scope. |
| Legacy credential setup script | Excluded | It targets the historical `gemini/.env`/dual-key arrangement, not the current renderer configuration. |

The 18 receipts remain `candidate_only` and `quarantined_pending_generation_and_founder_review`. A successful dry run only validates local source bindings; it does not request an image.

Independent visual review found 12 clear candidate on-model fronts, but none is
promoted by that observation alone: provenance still must pass the hash gate.
It also found eight active-plan SKUs without usable front authority and four
filename/product mismatches. Those remain excluded from scene input.

## Native-scene handoff assessment

The second worktree has no newer committed scene implementation; it contains
an untracked, candidate-only handoff with seven contracts:

- `BR-COMMERCE-1`, `BR-COMMERCE-2`
- `LH-COMMERCE-2`, `LH-COMMERCE-3`
- `SIG-COMMERCE-1`, `SIG-COMMERCE-2`, `SIG-COMMERCE-3`

It deliberately preserves `BR-COMMERCE-3` and `LH-COMMERCE-1`. The handoff’s
own preflights are blocked for exact native garment mattes and same-pose
on-model authorities. Those blockers remain in force.

All seven contracts are now portable in this worktree: their current SOT lock
is `4b5799ab3e16eaf75dd3c37321339eb861176bb53e161fd18263c35bed78c9e2`;
every contract reference exists locally and its SHA-256 matches. Three exact
lockups absent from `main` were carried with their verified bytes:

- Black Rose font-statue authority: `a879be465973a5edf2aa8fe6ae2e0eb34d66e11cf3e5081b3f4beea4d967aa3c`
- Signature font sculpture: `e259c246a320b8220044da915c3b6fa89ecbbe7ab4a74eb80bafebe668d0e274`
- Signature SR rose graphic: `2667368884d96e5d0689fdb5f6795940b0ea9aa79035d4542859c235b7864bcc`

The imported handoff remains an immutable historical record, including its
old workspace paths in preflight evidence. The new validator checks portable
contract references and current hashes instead of treating those historical
absolute paths as executable input.

## Verification

Run from this worktree:

```bash
python3 -m pytest tests/test_on_model_media_intake.py -q
python3 scripts/validate-native-scene-handoff.py
python3 scripts/run-geometry-candidate-pilot.py --max-images 18
```

Expected outcome: validator and test pass; the geometry command ends with a
dry-run message and makes no provider call. `--execute` remains a separate,
paid candidate-generation action and is not part of this consolidation.
