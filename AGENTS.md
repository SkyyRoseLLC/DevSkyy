# Repository Guidelines

## Project Structure & Module Organization

DevSkyy is a mixed Python, TypeScript, Next.js, and WordPress monorepo.

- `main_enterprise.py`, `api/`, `agents/`, `orchestration/`: FastAPI and agent
  platform.
- `src/`: shared TypeScript services, commerce utilities, hooks, and Jest tests.
- `frontend/`: Next.js application; tests live in `frontend/tests/` and
  colocated files.
- `wordpress-theme/skyyrose-flagship/`: SkyyRose WooCommerce theme, PHP
  templates, assets, and PHPUnit tests.
- `tests/`: Python unit, integration, security, and API tests.
- `docs/`: architecture, setup, testing, and operational documentation.
- `pipelines/`, `integrations/`, `security/`: 3D workflows, external systems,
  and security controls.

Keep generated files out of source directories unless build scripts
intentionally track them.

## Build, Test, and Development Commands

```bash
python -m uvicorn main_enterprise:app --reload --port 8000
pytest tests/ -v
npm run build
npm test
cd frontend && npm run dev
cd frontend && npm run test:e2e
cd wordpress-theme && npm run verify:full
```

Root `npm run lint`, `npm run type-check`, and `npm run format:check` validate
TypeScript. Use `pytest tests/ --cov --cov-report=html` for Python coverage.
Theme changes require rebuilding committed `.min.css` and `.min.js` outputs.

## Coding Style & Naming Conventions

Python uses four spaces, type hints, `snake_case`, Ruff, Black, isort, and mypy.
TypeScript uses project ESLint/Prettier rules, `camelCase` functions, and
`PascalCase` components/classes. WordPress PHP follows WPCS: prefix functions
and hooks with `skyyrose_`, escape output, sanitize input, and use tabs for
indentation.

Never add secrets, credentials, production URLs, or generated customer data.

## Testing Guidelines

Name Python tests `test_*.py`; use `@pytest.mark.unit`, `integration`,
`asyncio`, or `slow`. TypeScript tests use `*.test.ts` or `*.test.tsx`. Add
regression coverage for every bug. Run focused tests during development, then
relevant full suite before review.

## Commit & Pull Request Guidelines

History follows Conventional Commit-style subjects: `feat(theme): ...`,
`fix(theme): ...`, `docs: ...`, `chore(wolf): ...`. Use imperative, scoped
summaries.

Pull requests need problem statement, implementation summary, test evidence,
linked issue/bug ID, and screenshots or recordings for UI changes. Note
migrations, environment changes, security impact, and deployment steps. Never
deploy WordPress or production services without explicit approval.

Before any edit, check `.wolf/memory.md` for current session notes and open
work.

## Image Generation Fidelity Workflow

All product imagery is fail-closed. Product fidelity takes precedence over
scene polish, speed, cost, and generated-image aesthetics. Never infer product
details from filenames or from an earlier generated candidate.

Before every image-generation batch, complete these gates in order:

1. **Review the complete input lineage.** Inventory every file that can reach
   the model: the current canonical product SOT, the SKU dossiers from which it
   was derived, founder flatlays and physical-product photos, tech flats,
   approved logo or patch art, edit targets, proof manifests, proof boards, and
   the prompt contract. Recompute and compare all hashes. Reject missing,
   changed, stale, old-version, legacy-theme, unapproved, ambiguous-role, or
   contradictory inputs. A warning is a failure; do not generate.
2. **Visualize product truth before prompting.** Build and inspect a hash-bound
   proof/contact sheet showing every SKU and every construction, material,
   color, logo, patch, embroidery, silicone, tackle-twill, sublimation, front,
   back, or side detail the batch must preserve. Do not author the prompt until
   the proof board exists and has been visually reviewed.
3. **Store the prompt in source control.** Image-model prompts must be authored
   in an `.html` or `.json` file, never only in chat or shell history. Each job
   must include the exact positive prompt, exact negative prompt, model,
   operation, output settings, input paths and SHA-256 hashes, current SKU
   product hashes, proof bindings, and immutable visual invariants.
4. **Run a second pre-generation fidelity check.** Revalidate the current SOT,
   dossier hashes, every transitive input file, proof manifest and board,
   product hashes, prompt semantics, and output contract after the prompt is
   written. Generation is allowed only when the validator writes a
   `PASS_READY_TO_GENERATE` receipt bound to the exact prompt, proof board,
   input files, and SOT hashes. Editing any bound file invalidates the receipt.
   Before issuing that receipt, live-probe every required tournament model with
   `scripts/verify-image-judge-availability.py`; a configured policy flag is
   not evidence of actual availability. If any judge cannot be reached, do not
   generate a batch that cannot be verified afterward.
5. **Generate only the receipt-bound batch.** The model call must use the exact
   prompt and exact ordered references recorded in the prompt contract. Keep
   generated candidates in review-only state. Never describe generated imagery
   as approved, verified, exact, transparent, or SKU-correct without evidence.
6. **Run adversarial verification after every batch.** Invoke at least two
   independent skeptical visual reviewers, following
   `.agents/skills/adversarial-verification/SKILL.md`. Give them the original
   founder corrections verbatim, prompt contract, proof board, raw outputs,
   composites, and rerunnable validation commands. Each reviewer must inspect
   the pixels independently and return the structured verdict
   `clean|partially-improved|no-improvement|regressed` plus `recommend_ship`.
   The mandatory tournament is `gpt-5.5-pro` plus
   `gemini-3.1-pro-preview` as independent vision judges, followed by
   `claude-opus-5` as the synthesis judge. Every generated output must score
   at least 95 from each vision judge and at least 98 from synthesis; all
   judges must be available, source hashes must still be current, zero required
   regions may be unverifiable, and the synthesis `hallucination_veto` result
   must be `false`. The canonical machine-readable thresholds live in
   `wordpress-theme/skyyrose-flagship-2/data/image-generation-tournament-policy-v1.json`.
7. **Block downstream use unless all reviewers pass.** If either reviewer finds
   a mismatch or does not recommend shipping, the entire affected batch stays
   blocked and must not enter the scene compositor, V2 runtime, product cards,
   marketplace assets, or approval boards. Fix the source truth or prompt,
   issue a new preflight receipt, regenerate, and repeat adversarial review.

For localized corrections, preserve all approved pixels outside the measured
edit region and produce original-versus-output diff evidence. A checkerboard
appearance is not proof of transparency; inspect the alpha channel. Never
weaken or bypass a fidelity validator to obtain a passing result.
