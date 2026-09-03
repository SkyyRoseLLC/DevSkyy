---
name: fashion-e2e-task-execution
description: Runs any Fashion Theme Team capability as an end-to-end, evidence-backed SkyyRose task from scope and source authority through implementation, independent verification, release decision, and handoff. Use whenever work spans product media, catalog, design, frontend, motion, accessibility, commerce, visual QA, staging, packaging, release, or a corrective retry; invoke it alongside the narrow capability skill even when the request only names that capability.
---

# Fashion Theme Team End-to-End Execution

This is the shared execution contract for every Fashion Theme Team capability.
It turns a scoped request into a durable task with declared evidence, first-seen
issue capture, accountable resolution, and a completion decision. It is the
control layer around the specialist skills; it does not replace their product,
brand, commerce, motion, accessibility, or release expertise.

## Strict E2E policy

Every Fashion Theme Team task uses strict E2E. A task is `BLOCKED` unless it
has all of the following before any completion or release claim:

- a multi-field scope that identifies the candidate/source and target surface;
- explicit, non-empty acceptance criteria;
- non-empty, named evidence requirements including `independent_review`;
- a named project manager;
- a task roster that includes a reviewer independent of the manager;
- no unresolved scope issue;
- manager re-verification; and
- founder approval by someone other than the manager and independent reviewer.

No role may reduce an acceptance criterion, replace an independent review with
its own implementation note, or mark a missing check as not applicable merely
to close the task.

## Start every task before side effects

1. Issue a named project manager before work begins. The manager owns scope,
   dependencies, evidence completeness, issue recurrence, and the final
   handoff. A manager cannot self-approve a candidate or replace an independent
   reviewer.
2. Define the task roster. List every agent or automation used with its stable
   name, job title, assigned capabilities, and owned deliverable. Do the same
   for each remediation loop; do not collapse independent implementation and
   review work into an unnamed Team.
3. Freeze the exact scope: candidate/ref, theme path, route/feature, SKU and
   view when relevant, target environment, and explicit exclusions.
4. Bind authority: source hashes, catalog/dossier/media SOT, plugin/package
   version, platform contract, and current candidate identity.
5. Declare the evidence required to complete the task from the matrix below.
6. Create the task before editing, rendering, uploading, deploying, or calling
   a paid provider. In DevSkyy use `skyyrose.core.task_control` and
   `scripts/task_control.py`; the authenticated MCP surface provides the same
   task start, evidence, issue, remediation, verification, approval-request,
   and status operations. MCP writes require a durable
   `TASK_CONTROL_MCP_LEDGER_PATH`; do not use an ephemeral service filesystem
   as a release ledger. MCP may request founder approval but cannot grant it
   or close a task. Founder approval remains a deliberate CLI action with
   `--confirm-founder-approval`. Another repository must use an equivalent
   append-only task ledger with the same fields and approval boundary.
7. Log every meaningful result: command, artifact/hash, reviewer, state,
   scope, timestamp, and any unknown. Never put credentials, customer data,
   prompt/source bytes, or provider secrets in a ledger.

## First-seen issue rule

Record a defect at the first observation, using a stable issue fingerprint
made from scope plus the failed invariant. A recurrence must link to the
original issue rather than create an unrelated note. An unresolved issue blocks
completion and a re-run of the same scope until a reviewer records the exact
resolution. A changed candidate invalidates stale evidence; it does not erase
the original incident.

For paid generation, the first candidate is a one-SKU pilot. Automatic paid
retries are disabled by default. A fidelity, QC, source, provider, or review
failure opens an issue and blocks the same scope. An approved pilot never clears
an unresolved fidelity issue.

## Manager verification and Team loops

The project manager owns the verification loop for each task:

1. Verify every declared evidence item against the current candidate and check
   that no scoped issue remains open.
2. If anything is absent, stale, failed, or `UNKNOWN`, issue a remediation loop
   to the responsible specialist Team with the exact failed invariant and
   required replacement evidence. Do not close, release, or spend through it.
3. After the Team returns, attach new evidence, resolve only the issue that the
   evidence actually addresses, and verify again. Continue the loop until all
   declared requirements pass.
4. When the manager's verification is clear, raise the task as
   `READY_FOR_FOUNDER_APPROVAL`. The task is not complete at this point.
5. Record explicit founder approval. Only then may the manager record task
   completion and issue a final handoff.

The manager can coordinate and verify but cannot approve their own candidate.
No automatic loop may weaken an acceptance condition, change source authority,
or reinterpret a failed review as a pass.

## Capability evidence matrix

| Capability                       | Required task evidence before completion                                                                            | Independent verdict              |
| -------------------------------- | ------------------------------------------------------------------------------------------------------------------- | -------------------------------- |
| `fashion-strategy-brain`         | source registry, route/job matrix, source freshness, contract/schema parity                                         | strategy/knowledge review        |
| `fashion-brand-experience`       | brand direction, image provenance, crop/type rules, logo-off review                                                 | brand review                     |
| `product-fidelity-image-edits`   | source-authority receipt, contract, mask or optical plan, provider receipt, visual review                           | product-fidelity review          |
| `fashion-design-system`          | token/component contract, fixtures, generated parity, adoption impact                                               | design-system red team           |
| `fashion-commerce-engineering`   | catalog mapping, state matrix, server-truth checks, purchase-journey proof                                          | commerce QA                      |
| `fashion-frontend-motion`        | source/build hash, minified parity, 390/768/1440 captures, reduced-motion and fallback proof, console/network trace | frontend/motion review           |
| `fashion-a11y-performance`       | keyboard/focus, screen-reader/reflow, reduced-motion, CWV/budget, route/state trace                                 | accessibility/performance review |
| `fashion-visual-commerce-qa`     | eyes-on captures, candidate hash, SKU/provenance, customer-journey trace, failures                                  | independent QA verdict           |
| `fashion-handoff-contracts`      | `preview.html`, `contract.json`, `evidence.json`, schema/ID/source parity                                           | handoff validation               |
| `fashion-premium-feature-system` | feature ID/state matrix, fallback/security/performance/commerce proof                                               | cross-discipline review          |
| `fashion-branded-skills`         | source hash, selected route/pack, gap closures, authenticated authority capture                                     | independent evidence review      |
| `elite-builder-runtime`          | approved PRD/routing/budget, execution record, self-heal trace, downstream evidence                                 | independent Team gates           |
| `fashion-release-evidence`       | frozen candidate, all applicable gate records, compatibility/package/rollback packet                                | release engineer verdict         |
| `fashion-theme-team`             | phase ledger, ownership/handoffs, integrated candidate manifest, complete evidence set                              | founder-review readiness         |

## Staging and release work

Keep staging and production as different scopes. A staging task must include
candidate/source hashes, deployment receipt, desktop capture, 390px capture,
reduced-motion and fallback proof when motion exists, plus a failed-resource
report. Local build output and a provider success response are evidence only;
they are not staging proof. Never deploy or promote because a task record exists.

## Task state

Use only these dispositions in a task handoff:

- `IN_PROGRESS` — work has begun and evidence is being collected.
- `BLOCKED` — required authority, evidence, environment, or a prior issue
  prevents safe continuation.
- `NEEDS_REVIEW` — a candidate exists but needs independent or founder review.
- `FAIL` — a required invariant failed; retain it as issue evidence.
- `READY_FOR_FOUNDER_APPROVAL` — manager verification is clear; explicit founder
  approval is still required.
- `PASS` — founder approval, independent review, and all declared evidence are recorded.

`PASS` is never a deployment, promotion, merge, provider-spend, or production
authorization.

## Required handoff

```text
Task ID and scope: <stable ID; candidate/ref; target; SKU/view/route as applicable>
Project manager: <named manager accountable for scope, dependencies, evidence, and handoff>
Task roster: <agent/automation; job title; assigned capabilities; owned deliverable>
Authority: <source and candidate hashes, SOT and platform references>
Evidence: <named evidence with command, artifact path/hash, reviewer, timestamp>
Issues: <first-seen ID, recurrence count, resolution or active blocker>
Verdict: <IN_PROGRESS | BLOCKED | NEEDS_REVIEW | FAIL | PASS>
Release decision: <authorized target/scope, or explicit reason not authorized>
```

## Boundaries

- Do not turn a plan, prompt, provider receipt, clean working tree, or local
  build into a product or release claim.
- Do not replace verified product imagery with plausible substitutes.
- Do not re-run a paid or remote operation merely because a prior output failed.
- Reviewers do not repair or approve their own candidate.
