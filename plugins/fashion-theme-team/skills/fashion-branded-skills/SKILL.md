---
name: fashion-branded-skills
description: Routes the 234 bundled branding, content, lifecycle, commerce, funnel, SEO, and social skills through authenticated SkyyRose enhancement contracts. Use when one of those branded capabilities is requested or when auditing its gaps.
---

# Fashion Branded Skills

Use this adapter instead of preloading the bundled skill library. It adds
SkyyRose canon, SOT, authenticated-source, accessibility, security, privacy,
commerce-truth, evidence, and rollback controls without copying or modifying the
source skill bodies.

## Route

1. Read `../fashion-theme-team/brain/branded-skills/routes/route-registry.json`.
2. Match an exact skill name first. Otherwise rank conservative trigger matches;
   ambiguous matches remain `UNKNOWN` and must not cause bulk loading.
3. Read only the selected record in
   `../fashion-theme-team/brain/branded-skills/enhanced/`.
4. Verify its source path and SHA-256 against
   `../fashion-theme-team/brain/branded-skills/sources/source-registry.json`.
5. Load the referenced source `SKILL.md`, target-repository instructions,
   approved SOT, and only the route's named Brain packs.

## Execute

Apply the enhancement contract as the governing overlay. Resolve every recorded
gap through its closure control. Retrieve current primary authority at claim
time; a URL in the registry is a source requirement, not proof that a claim is
current. Record candidate ID, loaded packs, source hashes, tool expansions,
approvals, checks, unknowns, and rollback. Never invent brand, catalog, media,
policy, legal, platform, performance, or conversion facts.

Visual outputs still require the standard SkyyRose `preview.html`,
`contract.json`, and `evidence.json` handoff. Nonvisual outputs use their declared
artifact contract plus evidence conforming to
`enhanced-skill-evidence.schema.json`. A builder cannot approve its own result.

## Verification

From the plugin root, run:

```bash
python3 scripts/generate-branded-skill-enhancements.py --source-root vendor/branded-skills --output-root skills/fashion-theme-team/brain/branded-skills --check
python3 skills/fashion-theme-team/brain/branded-skills/sources/validate_source_registry.py --registry skills/fashion-theme-team/brain/branded-skills/sources/source-registry.json --plugin-root .
node skills/fashion-theme-team/brain/branded-skills/routes/validate-routes.mjs
bash scripts/verify.sh
```

These commands prove inventory, provenance, routing, schema, and package
integrity. A specific skill run requires fresh candidate-bound operational
evidence and independent review before `PASS`.

## Boundaries

The immutable source bodies are packaged under `vendor/branded-skills`; update
them only through the maintainer sync script and regenerate every bound hash.
No paid provider, credential use, upload, remote write, staging mutation,
deployment, or production mutation without its required approval. Source skill
inclusion is not factual authentication or quality certification. Missing or
stale authority is `UNKNOWN` or `BLOCKED`, never inferred.

## End-to-end execution

Run `fashion-e2e-task-execution` with every selected branded capability. Record
the exact source hash, route and packs, authority capture, gap closures, output
contract, and independent evidence; the first stale or unclosed authority gap
blocks that task scope until it is resolved.
