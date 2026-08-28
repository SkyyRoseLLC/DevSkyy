# SkyyRose bundled-skill enhancement contracts

This layer governs the 234 bundled skills inventoried from the approved source
root without copying their instruction bodies into the Fashion Theme Brain.
Each generated record is a small, versioned overlay validated by
`schemas/enhanced-skill.schema.json`; execution evidence validates separately
against `schemas/enhanced-skill-evidence.schema.json`.

## Lazy-loading architecture

The registry and route metadata are safe to inspect first. A router selects one
contract from the request, then loads that contract and its referenced source
`SKILL.md` only after a match. Source bodies, unrelated contracts, optional
provider definitions, and the full tool inventory remain unloaded. The source
path and SHA-256 bind the overlay to an exact body; `body_policy` must remain
`reference-only-never-copy`.

This contract does not change source-skill authority. Repository instructions,
approved SOT and founder decisions remain first; current official documentation
is next. SkyyRose guidance refines presentation and fashion-commerce behavior
but cannot invent facts, waive a gate, authorize a paid or external action, or
certify its own output.

## Contract lifecycle

1. **Inventory:** identify the source skill, relative path, source hash and
   inventory ID. A source-hash change makes the overlay stale.
2. **Draft:** map narrow triggers, inputs, authority, applicable Brain routes,
   SkyyRose overlay, output schema, and domain requirements. Every ethics,
   commerce, accessibility, performance, security, and privacy lane is explicit;
   `NOT_APPLICABLE` requires rationale.
3. **Validate:** validate the JSON against the strict schema, check referenced
   paths/hashes, reject copied source bodies, and confirm metadata-first loading.
4. **Review:** an owner reviews semantics; an independent reviewer owns the
   evidence verdict. Builders and generators cannot self-certify.
5. **Activate:** publish only a schema-valid `ACTIVE` contract with a changelog
   record, owners, compatibility statement, rollback procedure and provenance.
6. **Execute:** load the matched contract and source, apply the narrowest tool
   profile, log expansions, bind outputs/evidence to one candidate, and use safe
   fallbacks. Transient failures receive at most two bounded retries.
7. **Change:** use semantic versioning. Patch releases clarify or tighten without
   changing valid consumers; minor releases add backward-compatible fields or
   capabilities; major releases remove or incompatibly redefine behavior.
8. **Deprecate:** announce an owner, replacement, removal version, migration
   guide and support window. Removal is forbidden before the recorded version.
9. **Retire or roll back:** retain provenance and evidence. Restore the recorded
   last-known-good contract when a release gate fails; regenerate evidence for
   the restored candidate rather than reusing stale proof.

## Required operating controls

- Human-facing and fashion-commerce artifacts apply the canonical SkyyRose
  overlay; non-applicability is documented, never silently assumed.
- Commerce truth comes from approved SOT and authoritative server responses.
  No product, price, stock, review, rights, sustainability, or urgency claims
  may be invented.
- Accessibility uses WCAG 2.2 AA as the floor where UI is involved, including
  semantic structure, keyboard/focus behavior, reflow, reduced motion and
  equivalent fallbacks.
- Performance requirements name budgets and degraded modes. Heavy media,
  animation, WebGL, providers, and telemetry must preserve usable fallbacks.
- Security covers input/output handling, escaping, secrets, permissions,
  dependencies, provenance and supply-chain exposure. Privacy minimizes and
  redacts data; analytics require disclosed purpose, consent where applicable,
  retention and rollback.
- Provider routing is capability-based and opt-in. Paid APIs, credentials,
  uploads and external writes require approval. Cache misses must remain correct;
  context and retry ceilings are recorded, and optional providers never become
  implicit dependencies.
- Evidence is candidate-bound and invalidated by source, contract, SOT,
  approval, schema or output mutation. Missing, stale, skipped or unavailable
  evidence is `UNKNOWN` or `BLOCKED`, never `PASS`.

## Release, migration and adoption

Every activated contract needs owners, compatibility notes, a changelog entry,
support and deprecation policy, migration/rollback instructions, and deterministic
verification. Adoption may be measured with privacy-preserving counts such as
contract selection, successful completion, fallback use, failure class,
migration completion and version distribution. Do not capture prompts, secrets,
customer data or raw content. Metrics are operational signals, not proof of
quality or conversion uplift.

Release evidence distinguishes schema integrity from operational readiness.
Validation proves only that the contract is well formed. A skill run is `PASS`
only when all applicable checks pass against one candidate and an independent
reviewer records the verdict. Production use, deployment and remote publication
always remain separate approval decisions.

## Known boundaries

- The approved ledger reports 234 source skills and an aggregate inventory hash,
  but the aggregate hashing algorithm is not specified here; integrations must
  preserve the ledger value or document the canonical recomputation method.
- Per-skill contract generation, route records, source registry, changelog,
  adoption collector and verifier integration are owned by separate phases.
- Marketplace compatibility windows, support SLA, telemetry storage/retention,
  and the authoritative release signer remain `UNKNOWN` until approved.
