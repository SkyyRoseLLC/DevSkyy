# Global tagline retirement

Founder directive: retire the brand tagline from everywhere, without introducing a replacement. Scope: current 7116 DevSkyy working tree and local preview; no publishing, database edits or external account changes.

## Applied

- Canonical `assets/brand/brand.yaml`: active tagline empty, exact former phrase recorded as retired with date2026-09-06; matching Black Rose duplicate empty.
- V2 Home: removed tagline line. About: descriptive heading “The Story of SkyyRose.” Home SEO: brand name only. Existing images, heroes, scenes, cards, Quick View, Ask Skyy and rotating marks remain.
- Legacy WordPress, SOT theme and Next.js: removed tagline output and stale metadata. No replacement slogan.
- Python:101 active files updated across defaults, email/social output, prompts, generation instructions and tests. BrandConfig now supports empty tagline; generated PHP synchronized from YAML.
- Frontend/legacy worker:43 source/generated files updated, including mascot.min.js rebuilt from source.
- Guidance/config/template sweep:147 files plus separately recorded canon/translation updates; active skill prompts now direct omission rather than mandatory inclusion. Retired hashtag removed. Old reusable seed deprecated with empty prompt fragment.
- Original source snapshots retained under `.artifacts/v2-tagline-retirement-20260906/before/` for root guidance changes; worker receipts preserve exact hashes. Unrelated pre-existing work retained.

## Intentional retained references

Retirement deny-lists, deterministic brand checks, negative regression assertions, deprecated stable IDs, old document filenames/links, frozen deployment evidence, archived reports and historical sources retain the phrase only as history or enforcement. The old blog document is explicitly marked retired/not for publishing or generation. No approved ZIP, rollback artifact, captured remote baseline or original media was altered.

## Verification

Python:303 focused tests passed,194 relevant checks rerun after review fixes.101 source files passed AST/Ruff/Black checks. Independent Python reviewer approved. Brand PHP generator --check passed. A2A full integration requires missing `a2a` dependency; five isolated actual-function guard checks passed.

Frontend/legacy:42 PHP/TS/JS parse checks plus source/min and phrase-absence evidence passed. Independent frontend review could not proceed past its full type-check gate: this checkout lacks React, Next, Framer Motion and other required dependencies/types. Parsing success is not full frontend type-check certification.

Root: edited JSON parses, V2 PHP syntax and deterministic build checks passed. Local Home/About browser checks confirm no retired phrase in rendered HTML and no horizontal overflow. Native About content retains its source preservation checks; tagline retirement is an explicit founder revision, not accidental content loss.

## Delivery boundary

Source retirement is applied. Staging and production still require a separately reviewed release; no live-site removal is claimed. The previous exact ZIP authorization does not authorize shipping the changed files.
