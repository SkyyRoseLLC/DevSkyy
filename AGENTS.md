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

## Required standard for all skills: verified examples

Every skill must include task-specific examples of what to do and what not to
do, supported by identifiable evidence. Apply this standard to skill creation,
revision, audit, and adoption, including third-party skills. Read
[the verified-example standard](docs/skill-standards/verified-examples.md) when
performing that work.

Verify example provenance and distinguish source-verified guidance, recorded
observations, reproduced tests, and authenticated live executions. For workflows
requiring authentication, record redacted evidence of the intended
account/site/environment and permitted scope; never infer authenticated success
from a public page, an available credential, or a source citation. For offline
workflows, explicitly mark authentication not applicable. Never invent a
successful run or treat historical approval as current permission.

Include at least one evidence-backed correct example and one explicit incorrect
example with its correction and reason. Label illustrative negative cases as
illustrative unless actually observed or tested. Keep substantial examples in a
linked reference to avoid bloating every invocation. Missing evidence is an
explicit coverage gap, not a reason to fabricate verification or initiate an
unauthorized external action. A gap in example documentation does not itself
revoke authorization for otherwise permitted work.

Maintain examples in canonical editable sources or a clearly linked maintained
overlay for third-party distributions; do not silently modify disposable plugin
caches. Do not claim the skill library satisfies this standard until each
claimed skill has been checked.
