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

## First remote CI findings

PR #918's first run at `99fd633c0` identified three real issues:

1. Certification could not launch `cwebp`. Its approved 1.6.0/libsharpyuv 0.4.2 encoder was not installed on the clean runner. An official Linux x86 binary reproduced the 640px poster but changed the 1024px bytes. The official Linux ARM64 binary was then tested in a network-disabled Docker container and reproduced both original hashes exactly. Certification now uses the documented `ubuntu-24.04-arm` runner and SHA-pins the official archive to `69f5eebe203e0f3942fe37986209a1725741be19c152950a4283b376c95ec798`. All original files, output pins and strict regeneration checks remain unchanged. Runner architecture is part of this reproducibility boundary. Sources: [GitHub runner reference](https://docs.github.com/en/actions/reference/runners/github-hosted-runners) and [official WebP utilities](https://developers.google.com/speed/webp/docs/precompiled).
2. Ruff B905 flagged two authoring-script zip calls. The mascot neighbor iteration now uses equal-length adjacent slices with strict zip; independent review compared 363 inputs with the original and found identical weights. The scene bounds code explicitly rejects missing bounds before strict coordinate pairing, preventing silent truncation. Other changes in those two scripts are AST-verified formatting only. No authoring pipeline, Blender scene or media asset was executed or changed.
3. CodeQL reported exponential backtracking in the POT extractor at `scripts/build-pot.py:16`. Ordinary literal characters now exclude backslash, so they cannot overlap the escaped-character branch. Both plural literal groups receive the same correction. Regression coverage includes escaped literals and malformed 2,000-escape inputs in a subprocess with a bounded timeout. The exact generator hash is reconciled in build inputs. Independent source review approved this correction; no CodeQL suppression or weakened check was added.

Remote checks must rerun after these corrections. The original integration verification above remains scoped to its recorded checkpoint.

The corrected source passed the pinned full build and verification again, including unchanged assets, registries and POT. All 13 integrity/security/fixture tests and five rendition tests passed. Exact CI tool versions were reproduced locally: Ruff 0.16.6, Black 26.5.1, isort 9.0.1 and mypy 2.3.1. Black reported all 1,986 source files unchanged after scoped formatting; mypy and Ruff passed. The global isort result identified two final standard-library import-order issues in the new rendition builder/test; those were corrected and passed scoped validation. Initial apparent isort failures in four unchanged main files disappeared with the exact CI version, so those main files were not edited.

Independent Python review verified the formatting-only files' non-import ASTs and imported bindings, and checked every reconciled generator pin. The two historical film recipes document their separately prepared source directory: br-003/br-011 product references are tracked in the legacy theme and the br-008/br-009/br-010 on-model media are in V2. `assets/card-scenes` alone is not an interchangeable input. Their source names and rendering behavior are retained; neither recipe was run.


## Python runner routing

The Python job remained queued because GitHub has no registered runner matching `self-hosted, docker-arm64`. The sole online runner is macOS. The stopped Docker container retained agent 22 from July, but that registration no longer exists; its entrypoint re-registers on start and mounts the host Docker socket. It was not started or re-registered. Exit 137 is recorded, with OOMKilled false; the precise termination cause is unknown.

Commit `00dfcfa598fa3d1ae8ff28c86d6e0548b47a6d2a` originally changed this job from `ubuntu-latest` solely to exercise Linux service-container support. The job uses checkout-relative paths, a local Redis service and standard dependencies, with no private network requirement. Routing now uses `ubuntu-24.04-arm`, the same hosted Linux architecture already verified by V2 certification. Tests, services, dependencies, permissions, triggers, coverage and artifact steps remain unchanged. The remote Python run must validate this routing before a green claim.

## Runtime checkpoint integration

Parent checkpoint `437b8353e` was integrated as `0ebe92150`. Its bounded native Core stylesheet inlining, first-card frame priority, exact Inter unicode range and canonical Home label fallback retain their reviewed source changes. The existing upstream security policy and all reconciled CI generator pins remain exact; the combined PHP baseline checksum was recomputed from the merged manifest. Performance acceptance remains open.
