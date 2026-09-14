# PR 923 Copilot source repair

This repair starts from PR head `2c7644772be1d19c9218ebfb2b980350033b83f7` and
integrates CI baseline `20dec13d2`. It preserves the V2 product registry schema
and all nine founder-approved scene records. No product facts, source imagery,
scene approval records, deployment permission, or provider receipts were
changed.

The renderer now uses recovery-rail/track markup and writes to repository
`.artifacts/v2-commerce-scenes/rendered`, outside the theme package. It rejects
HTTP errors, PHP diagnostic output, page errors, missing/duplicate scenes, and
document overflow. The offline preview now includes the current collection
styles and missing WordPress helper adapters. Its prices/availability are
fixtures and cannot certify live commerce.

Approved scenes render regardless of the incidental `hero_composed` flag. The
scene template independently rejects records outside the exact approved
poster/motion/SKU contract. It translates the motion button. Only those two PHP
entries in `runtime-php-baseline.json` were refreshed, with that file's hash
updated in `build-inputs.json`. The POT was rebuilt from source.

The image preflight validator now matches the unchanged founder policy: GPT-only
vision, no synthesis model, minimum 95, source fidelity and founder approval
still required. Offline tests reject stale models and weakened gates. The
separate paid-generation readiness command remains pending: its default
`assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/image-model-prompts-v1.json`
is absent, and no tracked replacement was found. The judge-availability probe
would make a paid API call and was not run. No successful availability or
execution receipt was manufactured. This command is not part of build, verify,
package, or the source-certification CI workflow.

Standalone browser scripts create their output directories before launch; motion
testing targets the explicit motion toggle. Delivery checks reject failed HTTP
responses and unexpected redirects.

Validation performed:

- Clean `npm ci --ignore-scripts --no-audit --no-fund`: 16 packages installed
  from the already tracked `npm-shrinkwrap.json`. A second lockfile was not
  introduced; the historical missing-lock comment is stale.
- Full `npm run build`: passed using Node 22.23.2, npm 10.9.8, Python 3.12.12,
  Pillow 12.3.0, fonttools 4.59.2 and brotli 1.2.0.
- Full `npm run verify`: passed with the hash-verified native WordPress/Woo
  fixture supplied through `V2_WP_FIXTURE`.
- Full `npm run package:theme`: passed, 557 runtime entries; artifact SHA-256
  `cbe2382251f16ab944d79af40ba9eb1c8cb2234858ff9ec23e50855e0ef30692`. The
  receipt records the current uncommitted source state, not release approval.
- 19 focused Python tests: passed (policy, registry, real scene templates,
  negative poster/SKU/candidate-variant cases).
- 27 actual browser captures: passed, nine scenes across 1440×900, 768×1024, and
  390×844. Desktop/mobile contact sheets visually inspected. Existing
  portrait/landscape compositions and scene commerce details remain visible;
  fixture unavailable-product messages are not live stock evidence.
- Ruff, Black, PHP parsing through the verification suite, and diff checks:
  passed.

Standalone multi-server cadence, cross-engine and motion timing scripts require
their original before/current WordPress servers; they were not executed against
an unrelated live site. Source-only checks and screenshots grant no deployment,
paid generation, product-media promotion, or founder visual acceptance.
