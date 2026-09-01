---
name: skyyrose-task-execution-controller
description: Govern SkyyRose product, catalog, staging, deployment, visual-QA, and paid-generation work with a durable task contract. Use this skill whenever work must be released, verified, staged, rendered, deployed, promoted, corrected after a missread, or reported as complete; use it even when the request does not explicitly mention task logging. It creates evidence requirements before side effects, records the first occurrence of any issue, blocks repeated spend or completion while the issue remains open, and produces an accountable release record.
---

# SkyyRose Task Execution Controller

Turn a request into an accountable, evidence-backed work record before any
meaningful side effect. The purpose is not to slow delivery; it is to avoid a
second spend, a false “complete,” or a lost founder correction after the first
time a defect appears.

Use the implementation in `skyyrose.core.task_control` and its operator entry
point `python scripts/task_control.py`. The task ledger is append-only runtime
data under `var/task-control/` unless a controlled entrypoint specifies its own
ledger.

## Operating contract

1. Define the exact task scope before work begins. Use stable identifiers such
   as SKU, view, style, branch/ref, target environment, and release surface.
   Do not replace physical product truth with a loose name or collection label.
2. Issue a named project manager for the task. The manager owns scope,
   dependencies, evidence completeness, issue recurrence, and the handoff; the
   manager cannot self-approve a candidate.
3. Define the task roster. For every agent or automation used, record its stable
   name, job title, and assigned capabilities. Do the same for a remediation
   loop so an issue cannot lose accountability between rounds.
4. Declare the evidence required to call the task complete. Each evidence name
   must be concrete and reviewable, for example `source_hash`,
   `desktop_capture`, `mobile_capture`, `network_report`, `candidate_sha256`,
   or `catalog_diff`.
5. Start the task before an external side effect. Attach source fingerprints,
   preflight results, provider receipt IDs, screenshots, hashes, and test
   reports as evidence during execution.
6. When a defect, mismatch, blocked request, or unexpected result appears,
   record it immediately with a stable fingerprint. If it appears again, use
   the same fingerprint so the ledger creates a linked recurrence rather than
   hiding the first event under a new note.
7. Do not mark the task complete while declared evidence is missing or a scoped
   issue is unresolved. Resolution names the reviewer and records the precise
   corrective action; it is not a silent retry.
8. The issued project manager verifies the completed evidence set. When it is
   incomplete, issue a remediation loop to the responsible Team and re-verify
   after its fresh evidence arrives. When it is complete, request founder
   approval. Only an explicit founder approval record allows task completion.

For strict E2E work, declare explicit acceptance criteria and include
`independent_review` in the evidence requirements. The independent reviewer
must be in the roster and must not be the project manager or the founder
approver. A missing criterion, reviewer, or evidence item is `BLOCKED`.
6. Report the task ID, evidence, active issue IDs, resolutions, and remaining
   blockers in the final handoff. Never claim release, promotion, staging, or
   product fidelity from a plan alone.

## Task patterns

### Paid product generation

For every paid render, include SKU, presentation style, requested view, and a
source fingerprint in the scope. Require `source_fingerprint`, `candidate`,
and `visual_qa`.

The OpenAI render pipeline already has a release-controller adapter. It:

- permits one SKU pilot from a multi-SKU request;
- records the task before a provider call;
- defaults to zero automatic paid retries;
- creates a first-seen incident for QC failure, unjudged output, provider
  failure, or human fidelity miss;
- blocks the same render scope until an accountable resolution; and
- requires visual review before a pilot is approved.

Do not bypass that path by calling a provider client directly. Do not use an
unapproved candidate as storefront media.

### Staging release and visual QA

Scope the task to `theme/ref`, `target=staging`, and the exact surface(s). At a
minimum require `source_hash`, `deployment_receipt`, `desktop_capture`,
`mobile_capture`, and `network_report`. For motion surfaces also require
`reduced_motion_capture` and `fallback_capture`.

Record each failed resource, crop regression, visual mismatch, or cache
discrepancy as an issue before attempting a second deployment. A passing local
build is evidence, not staging validation.

### Catalog and source-of-truth corrections

Scope the task to SKU plus the affected data surface (catalog, dossier, source
manifest, image registry, or runtime mapping). Require the original source
fingerprint, exact changed fields, downstream validation, and a source/runtime
consistency check. Treat a founder correction as binding source material, not
an optional prompt preference.

## Operator commands

Start a controlled task:

```bash
python scripts/task_control.py start \
  --kind product_render \
  --project-manager fashion_theme_team_project_manager \
  --scope-json '{"sku":"br-008","style":"on-model","view":"front"}' \
  --require source_fingerprint --require candidate --require visual_qa
```

Record evidence without secret values or image bytes:

```bash
python scripts/task_control.py evidence --task-id TASK_ID \
  --name source_fingerprint --details-json '{"sha256":"..."}'
```

Record the first sighting of a defect. Reuse the same fingerprint if it
reappears:

```bash
python scripts/task_control.py issue --task-id TASK_ID \
  --category product_fidelity \
  --fingerprint br-008-front-zero-fill-v1 \
  --summary 'Rose artwork appears inside the plain front zero'
```

Check or authorize completion:

```bash
python scripts/task_control.py status --task-id TASK_ID
python scripts/task_control.py complete --task-id TASK_ID --reviewer founder
```

## Handoff format

Use this structure whenever the skill governs work:

```text
Task: <task ID and exact scope>
Project manager: <manager who ran the evidence and remediation loop>
Task roster: <agent/automation; job title; assigned capabilities>
Outcome: <completed | blocked | needs review>
Evidence: <named evidence and stable paths/hashes>
Issues: <first-seen ID, recurrence count, resolution or blocker>
Release decision: <authorized scope and target, or why it is blocked>
```

## Boundaries

- A record of source identity does not prove a visual output is correct; obtain
  the required visual evidence.
- A successful provider response does not authorize promotion or deployment.
- Never store API keys, customer data, unredacted provider payloads, or image
  bytes in the ledger.
- Keep staging and production scopes distinct. Staging authorization does not
  become production authorization.
