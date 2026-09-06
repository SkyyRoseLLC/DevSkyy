# V2 draft PR integration — 2026-09-06

Status: integrated source verified; draft review only. Mobile LCP remains above the target at the frozen completion checkpoint. Further feature-preserving performance work is active in the original V2 worktree. This report does not certify release readiness, approve deployment, or authorize merging.

## Source authority

- Frozen V2 completion checkpoint: `aabd2bffd`.
- Integrated remote main: `4da200886c73252c5271f5bf89de61c5c32ba062`.
- Isolated branch: `codex/v2-completion-pr-20260906`.
- Local main now equals remote main. Its previous ten commits were all patch-equivalent to commits already upstream and remain preserved under `codex/main-preserved-before-v2-sync-20260906`. See `main-sync.json`.
- Twelve inherited legacy conflicts were resolved to current main only after verifying that V2 had never modified those paths relative to its original main ancestor. Exact blob proofs are in `conflict-resolution.json`.
- The translation catalog was regenerated from V2 source: 579 singular messages and one plural record. It remains byte-identical to the frozen checkpoint.
- The only theme-file difference introduced by integration is the existing upstream CSP change in `inc/security.php`, from `afd03236562ae0518885f2cf669130b5866e037b`: WordPress.com font origin and EU Mixpanel connection origin. Its PHP certification hash and the policy's input hash were reconciled explicitly. Integrity checks were retained.
- All 402 protected media, font, model and decoder files remain byte-identical. All other theme files, including the nine-scene manifest and existing 3D renderer, match the frozen checkpoint. See `preservation.json`.

## Verification

- Pinned Node 22.23.2, npm 10.9.8, Python 3.12.12 and Pillow 12.3.0 build: PASS.
- Full theme verification with the pinned native fixture: PASS.
- Native fixture downloaded afresh from official WordPress/WooCommerce sources: all seven existing SHA-256 pins PASS.
- Native PHP gallery boundary with that downloaded fixture: PASS.
- Eleven negative integrity/translation/fixture tests: PASS.
- Ruff, Black, isort and focused mypy for the changed Python tooling: PASS.
- Independent Python correctness/security review: approved; requested workflow path trigger and formatting corrections applied.
- `git diff --check`: PASS.

Logs remain local under `.artifacts/v2-pr-integration-20260906/`. The initial verification log records the missing-fixture setup failure; `verify-final.log` is the corrected complete run. Earlier baseline and browser evidence is preserved in `tasks/v2-completion-20260906/` and its local artifacts. These tests are not evidence of a deployment or the subsequent performance changes.

## Clean-runner corrections

The recovered i18n regression test expected a historical line number. It now checks the actual reported source line, exact runtime file, and plural call while retaining plural rendering assertions. A separate fixture test still verifies deterministic line-number extraction.

The certification workflow previously did not provision the mandatory native gallery dependencies. It now downloads only the seven declared WordPress 7.1/WooCommerce 11.1.0 source files, validates every payload before creating a fresh fixture directory, and passes its location through `V2_WP_FIXTURE`. The PHP test independently validates those hashes again before executing fixture code. Unknown manifest paths, altered bytes and existing destinations are rejected. Changes to `tools/v2-runtime/**` now trigger certification.

## Remaining gates

The PR must stay draft while the active performance checkpoint is pending. Founder visual acceptance, mobile performance targets and any production release decision remain separate gates. No deployment, live commerce operation, paid generation, main push or automatic merge was performed.
