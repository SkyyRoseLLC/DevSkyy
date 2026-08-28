# Branded skill routing

`route-registry.json` is the deterministic routing layer for the 234 imported
skills across seven source domains. Each source skill receives one stable ID,
explicit triggers, owners, Brain packs, lifecycle stage, commerce surface,
source-authentication gates, prohibited uses, and lazy-loading instructions.

The registry routes capability; it does not approve brand facts, product facts,
external claims, imagery, visuals, publication, commerce mutation, or release.
SkyyRose repository/founder canon and SOT evidence always win. Missing evidence
is `UNVERIFIED`; filenames never establish image or product identity.

## Deterministic rebuild and validation

From the plugin root:

```bash
node skills/fashion-theme-team/brain/branded-skills/routes/build-routes.mjs
node skills/fashion-theme-team/brain/branded-skills/routes/validate-routes.mjs
```

The validator proves route shape, count, stable-ID uniqueness, seven-domain
coverage, fail-closed evidence disposition, and filename-identity protection.
It is plugin-integrity evidence only, not visual, operational, commerce, or
release certification.

## Lazy-loading rule

Choose by exact skill name first, then domain trigger, commerce surface, and
lifecycle stage. Load only the chosen source `SKILL.md`, its listed Brain packs,
target-repository canon, and applicable SOT. Ambiguity returns ranked IDs with
`UNKNOWN`; it must not cause speculative loading of multiple source skills.

## UNKNOWNs

- The target audience segment and requested journey state are unknown until a
  work order supplies them.
- Rights, consent, catalog, policy, price, stock, imagery, and publication state
  remain unknown until verified against their owning sources.
- External platform or market claims require current authenticated primary
  sources. No external claims were needed or introduced by this routing model.
- Trigger phrases are conservative retrieval hints derived from source names and
  descriptions; semantic routing quality requires evaluation on real task packets.
- Source skill quality and brand suitability are not certified by inclusion.
