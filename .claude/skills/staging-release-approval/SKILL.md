---
name: staging-release-approval
description:
  Prepare a reviewed staging release packet, bind approval to exact artifact and
  rollback hashes, deploy within existing staging authorization, and verify
  filesystem, browser, feature, delivery, and performance results separately.
  Use for staging release preparation, approval, installation, or verification;
  production promotion requires a separate workflow and explicit authorization.
---

# Staging release approval

Turn an accepted candidate into a reviewable, recoverable staging release.
Preparation, authorization, installation, parity, performance readiness, and
founder acceptance are distinct states. This skill grants no deployment
permission itself.

## Verified examples

Read [verified examples](references/verified-examples.md) when preparing or
assessing a release decision. Each case separates what to do, what not to do,
the source evidence, authentication status, and unverified limits. Historical
evidence never supplies current deployment permission.

## Establish the release boundary

Read project instructions and current handoff notes before edits. For DevSkyy,
resolve the shared `.wolf/memory.md` from the parent of
`git rev-parse --path-format=absolute --git-common-dir`; retain the current
worktree identity. Verify branch, HEAD, dirty files, accepted source scope, and
existing release evidence. Preserve unrelated work.

Identify the exact staging hostname/site ID, remote theme or application root,
active version, and current file inventory through read-only inspection. Verify
that the selected connection targets staging. A theme name, URL resemblance, or
available credential is insufficient identity evidence. Label partial
inventories as partial; do not infer a remote Git commit or whole-site
certificate from them.

Use existing authorization if it covers the exact artifact, procedure, target,
and actions. Historical approvals are examples, not permission for a new
release. Read [recorded-workflow.md](references/recorded-workflow.md) only for
the SkyyRose precedent or its evidence limits.

## Prepare a concrete approval packet

Complete authorized local preparation and read-only staging inspection before
asking for approval. Use [release-packet.md](references/release-packet.md) for
the packet and result fields.

- Freeze the accepted source identity, including relevant uncommitted bytes.
  Build required runtime outputs and create one reproducible archive. Record
  SHA-256, byte size, archive root, and a path/hash manifest. Check the archive
  against that manifest and inspect unsafe paths or unexpected files before any
  extraction on staging.
- Compare the intended package with staging. Record additions, replacements,
  unchanged files, removals, and exclusions with reasons. Excluding
  authoring/tooling files from runtime packaging does not authorize deleting
  them from source or backups.
- Preserve the current staging baseline in a rollback archive; verify its
  contents against the captured path/hash inventory. Hash the archive and keep a
  durable recovery copy outside the installation target. Record limitations: a
  verified backup is not an executed restore rehearsal.
- Write a deployment/rollback procedure covering target checks, pre-install
  drift checks, exact installation, expected side effects, parity checks,
  recovery commands, recovery verification, and severe-regression triggers. If
  the release changes database/editor content, include its exact migration and
  separate backup/recovery scope; a theme ZIP does not establish database or
  managed-HTML parity.
- Review the artifact and procedure for source preservation, scope, recovery,
  and verification gaps. Retain any independent review evidence and bind it to
  the reviewed hashes; do not label a self-review independent.
- Present the archive, procedure, rollback hashes, expected changes, review
  result, known failures, and permitted verification actions. Approval may cover
  staging diagnosis while overall readiness remains `NEEDS_MORE_WORK`.

If exact authorization is missing, ask one concise question referencing the
completed packet and explain that staging mutation needs explicit approval. Once
that authorization exists, continue without asking again for each covered step.
Changed artifact/procedure hashes, target, baseline, or scope invalidate the
affected approval binding; prepare an updated packet rather than silently
substituting bytes.

## Execute the authorized staging release

Immediately before installation, rehash the artifact, procedure, and rollback;
verify remote identity and compare the current baseline to the reviewed
baseline. Stop on unexplained drift. Do not overwrite a newer deployment or
manufacture matching state.

Use the reviewed installer and exact ZIP. Preserve a receipt of the target,
time, hashes, baseline, installer outcome, and platform side effects. Never
rebuild from a moving branch during installation. A failed or uncertain
installer response requires inspection of actual remote state before any retry.

Keep manual cache purges, platform settings, repository connections, paid
upgrades, product/media replacement, orders/payments, and production changes
outside scope unless explicitly authorized. If the normal installer itself
invalidates cache, record that observed side effect separately from a manually
initiated purge.

If an approved recovery trigger occurs, execute the reviewed rollback and verify
restored identity and functionality. Distinguish baseline performance failures
from installation-created severe regressions. Do not invent a rollback trigger
merely because a previously known performance gate remains open. Stop after a
failed recovery or unexplained identity mismatch; preserve evidence and report
the precise next decision instead of retrying indefinitely.

## Verify separately on actual staging

| Evidence layer    | Required check                                                                                                                                            | Claim boundary                                                                                             |
| ----------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------- |
| Filesystem parity | Installed path/hash map versus release manifest, including unexpected/missing paths and declared exclusions; recheck at end of testing                    | Does not establish cached HTML, database, optimized-resource, or CDN response identity                     |
| Browser parity    | Actual staging routes, correct asset selections, desktop/mobile composition, console/resource errors, overflow, reduced motion and no-JS where applicable | Label sampled routes, emulated devices, and untested states explicitly                                     |
| Feature parity    | Accepted features and interactions against the declared release contract                                                                                  | Rendered UI or a sampled frame does not prove all interactions, animation seams, or garment fidelity       |
| Platform delivery | Actual response headers, request timing, cache/encoding behavior, media ranges, and deferred-resource loading                                             | Preserve contradictory responses; do not purge or edit configuration to erase diagnostic evidence          |
| Performance       | Fresh deployed measurements using declared browser/device/network/cache profiles and acceptance thresholds                                                | Local timings and unthrottled feature checks are not deployed performance or physical-device certification |

For SkyyRose V2, select the applicable accepted surfaces: Home and collection
animated heroes, nine Scroll World scenes when in scope, responsive cards,
native WooCommerce Quick View, Ask Skyy intent-triggered loading, Shop,
representative PDPs, Bag, Cart, Search, and Checkout. On untouched Home, inspect
whether GLB/Three/Draco/3D textures remain deferred until the intended
interaction. Verify PDP selections against authoritative product/media records;
packaging does not approve replacement imagery.

An empty Checkout redirect to Cart proves only that state. Keep populated
checkout, payment, cart mutation, and chat submission untested unless
specifically authorized. Collect redacted evidence; omit cookies, tokens,
nonces, and personalized form values.

If files match but managed HTML/resources are stale, report filesystem `PASS`
and browser `BLOCKED` or `FAIL` as evidenced. Permit bounded normal revalidation
under the approved plan, preserving earlier evidence. Do not silently authorize
a manual purge. Recheck installed identity after browser/performance testing to
detect concurrent deployment changes.

## Return the staging result

Report target, artifact/procedure/rollback hashes, preflight and final
identities, installer and rollback outcomes, evidence links, and one status per
layer: `PASS`, `FAIL`, `BLOCKED`, or `UNVERIFIED`, with explicit bounds.
Preserve earlier failures when later checks pass and identify which final
evidence supersedes them.

State overall readiness independently. Filesystem and bounded browser parity may
pass while performance remains `FAIL / NEEDS_MORE_WORK`. Keep physical-device,
field-INP, full accessibility, product/media, payment, and founder acceptance
gaps explicit when relevant. Return results for founder review and stop at the
authorized staging boundary. Staging success does not approve production or
later optimization work.
