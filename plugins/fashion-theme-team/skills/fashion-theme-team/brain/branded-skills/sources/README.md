# Branded Skill Source Provenance

This directory inventories the 234 `SKILL.md` source files packaged under
`vendor/branded-skills`. The registry establishes source identity from the
package-relative path, source-byte SHA-256, byte length, exact
raw frontmatter, and frontmatter SHA-256. Parsed frontmatter is explicitly
non-authoritative.

`source-registry.json` classifies each source by its bundle domain and its
fashion-commerce applicability. The classification is routing metadata, not a
brand, product, legal, platform, accessibility, or performance fact. Repository
canon and SOT always win. A filename never establishes identity or imagery.

`authority-requirements.json` defines claim-specific primary-source requirements
for all seven domains. Its URLs identify approved authority classes; they do not
constitute a cached factual claim. At the time a claim is made, retrieve the
current document over authenticated HTTPS, record the final URL, publisher,
retrieval date, version/effective date when available, and immutable revision or
content hash. If that cannot be done, the claim is `UNVERIFIED`. `repo://SOT`
means the applicable repository-owned catalog or policy source of truth.

Regenerate and validate deterministically:

```bash
python3 skills/fashion-theme-team/brain/branded-skills/sources/generate_source_registry.py \
  --source-root vendor/branded-skills \
  --authorities skills/fashion-theme-team/brain/branded-skills/sources/authority-requirements.json \
  --output skills/fashion-theme-team/brain/branded-skills/sources/source-registry.json \
  --as-of 2026-08-16

python3 skills/fashion-theme-team/brain/branded-skills/sources/validate_source_registry.py \
  --registry skills/fashion-theme-team/brain/branded-skills/sources/source-registry.json \
  --plugin-root . \
  --expected-count 234 \
  --report skills/fashion-theme-team/brain/branded-skills/sources/validation-report.json
```

`authority-capture.json` records a dated, host-pinned, CA-verified, complete
content hash and publisher-identity check for every registered public primary
artifact. This authenticates the artifact, not any future claim: claim-specific
anchors and applicability must still be refreshed at use time. Platform
dashboards or login-only evidence is
`UNVERIFIED` unless an authorized authenticated session supplies it. No bundled
skill has been substantively fact-checked merely by being inventoried.
