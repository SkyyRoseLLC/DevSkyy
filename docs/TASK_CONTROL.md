# SkyyRose Task Execution Controller

`skyyrose.core.task_control` is the SkyyRose Task Execution Controller: the
production control plane for product renders, catalog corrections, staging
deployments, visual QA, and future automated jobs.

It is intentionally small and deterministic. It does not pretend that a log
can judge fashion fidelity. Instead, it makes a claim of completion dependent
on concrete evidence and makes the first observed issue durable.

Every task is issued a named **project manager** when it begins. That person or
role owns task scope, dependencies, evidence completeness, issue recurrence,
and handoff. A project manager cannot replace an independent reviewer or
self-approve a candidate.

Every task record also carries a roster of every participating agent or
automation. Each entry has a stable name, job title, and the capabilities it
was assigned for that task. Remediation loops carry their own roster so
responsibility remains visible across rounds.

For work governed as strict E2E, the controller also requires a non-empty
acceptance contract and an assigned independent reviewer. Its `independent_review`
evidence is mandatory, its reviewer must differ from both the project manager
and founder approver, and its receipt must attest `"PASS"` for **each exact
acceptance criterion** issued when the task started. A broad “looks good” note
cannot satisfy a specific desktop/mobile/fallback contract.

## Lifecycle

```text
task_started
  -> evidence_recorded (each declared requirement)
  -> issue_opened (or issue_seen_again linked to the first issue)
  -> issue_resolved
  -> task_completed
```

`task_completed` cannot be written when a required evidence name is missing or
an issue in the task scope is still open, or the founder has not explicitly
approved the manager-verified task. A second observation of the same issue
fingerprint creates `issue_seen_again`; it cannot overwrite, hide, or reset the
original record.

The ledger is append-only JSONL and contains bounded task metadata, hashes,
reviewer names, and evidence references only. It rejects API keys, tokens,
passwords, cookies, credentials, deeply nested data, oversized strings, and
oversized records at the append boundary. Do not put image bytes, customer
data, or unredacted provider payloads in `--details-json`.

## Use it for any task

The command interface writes runtime data to `var/task-control/task-events.jsonl`
by default (ignored by Git):

```bash
python scripts/task_control.py start \
  --kind staging_deploy \
  --project-manager fashion_theme_team_project_manager \
  --agent-json '{"name":"fashion_visual_commerce_qa","job_title":"Visual Commerce QA Reviewer","capabilities":["desktop and mobile visual review","network failure review"]}' \
  --strict-e2e \
  --acceptance 'Desktop and 390px evidence pass the product-card framing contract.' \
  --acceptance 'Reduced-motion and media-fallback evidence pass.' \
  --require independent_review \
  --scope-json '{"theme":"v2","target":"staging"}' \
  --require source_hash --require desktop_capture --require mobile_capture

python scripts/task_control.py evidence --task-id TASK_ID --name source_hash \
  --details-json '{"sha256":"..."}'

# This command must be run by the separately configured reviewer identity.
python scripts/task_control.py independent-review --task-id TASK_ID \
  --reviewer fashion_visual_commerce_qa \
  --details-json '{"criteria":{"Desktop and 390px evidence pass the product-card framing contract.":"PASS","Reduced-motion and media-fallback evidence pass.":"PASS"},"desktop_capture":"sha256:...","mobile_capture":"sha256:..."}'

python scripts/task_control.py issue --task-id TASK_ID \
  --category visual_regression --fingerprint v2-stone-frame-crop-v1 \
  --summary 'Stone frame clips product on 390px view'

python scripts/task_control.py status --task-id TASK_ID
```

When a task is not ready, the issued manager creates a correction loop for the
responsible Team, then re-verifies the required evidence after the loop:

```bash
python scripts/task_control.py remediate --task-id TASK_ID \
  --owner-team fashion-frontend-motion \
  --objective 'Repair the 390px stone-frame crop and attach fresh evidence.' \
  --issue-id ISSUE_ID

python scripts/task_control.py request-founder-approval --task-id TASK_ID \
  --project-manager fashion_theme_team_project_manager
python scripts/task_control.py founder-approve --task-id TASK_ID --founder founder \
  --confirm-founder-approval
python scripts/task_control.py complete --task-id TASK_ID \
  --reviewer fashion_theme_team_project_manager
```

Only a named reviewer can resolve an issue or mark a task complete. The release
entrypoint for OpenAI product rendering uses the same controller automatically;
it adds a one-SKU pilot policy and a paid-spend circuit breaker on top.

## Trusted actor configuration

The controller never treats `--reviewer`, `--founder`, `project_manager`, or an
MCP request body as authentication. Mutating CLI commands require
`TASK_CONTROL_CLI_ACTOR_ID`, which must match the issued task participant for
the requested action. Founder approval additionally requires
`TASK_CONTROL_FOUNDER_ID` to equal both that configured CLI actor and the
requested founder. Read-only `status` remains available without an actor.

MCP mutations require `TASK_CONTROL_MCP_ACTOR_ID`. Configure each
role-sensitive MCP deployment with exactly one issued actor identity and a
separate `MCP_SERVICE_TOKEN`; for example, the manager service is not also the
independent-reviewer service. The service rejects a caller-provided role that
does not match its configured actor. This is a deployment requirement, not a
substitute for the existing transport authentication.

## Entry-point rule

New mutating, paid, deployment, or asset-promotion entrypoints must create a
task before their first external side effect, issue a project manager, and call
`verify` / `request_founder_approval` / `complete` after attaching their
declared evidence. The controller is deliberately a library rather than a
background process so it can be tested, versioned, and invoked from Python, CI,
or operator scripts without hidden state.

## MCP integration

The DevSkyy MCP service exposes the same strict controller as typed operator
tools. This makes task state available to dashboards and approved automated
workflows without allowing a tool call to silently close a task.

| MCP tool | Outcome |
| --- | --- |
| `devskyy_task_start` | Creates a strict E2E task; a manager, independent reviewer, evidence contract, and acceptance criteria are required. |
| `devskyy_task_record_evidence` | Appends one named evidence receipt. |
| `devskyy_task_record_independent_review` | Appends the separate reviewer’s verdict. |
| `devskyy_task_open_issue` / `devskyy_task_resolve_issue` | Records the first failure and its accountable resolution. |
| `devskyy_task_open_remediation_loop` | Issues a bounded repair loop with a named Team roster. |
| `devskyy_task_verify` | Runs the issued manager’s deterministic evidence/issue audit. |
| `devskyy_task_request_founder_approval` | Raises an evidence-complete task for approval; it does not approve it. |
| `devskyy_task_status` | Returns the task contract, current evidence, issue recurrence, and eligibility as JSON. |

The current FastMCP generated schema wraps each typed request in a `params`
object. MCP clients should discover the input schema via `tools/list` and send
for example `{"params":{"task_id":"...","name":"source_hash","details":{...}}}`;
they must not flatten that object and silently bypass schema validation.

The MCP surface intentionally exposes **no** founder-approval or completion
tool. Founder approval is a human-controlled CLI action and now requires the
additional `--confirm-founder-approval` acknowledgement. A valid founder
approval still does not release, deploy, merge, promote, or spend; those are
separate authorization decisions.

MCP task-control writes require `TASK_CONTROL_MCP_LEDGER_PATH` to point at a
durable JSONL volume shared by the MCP service’s workers and
`TASK_CONTROL_MCP_ACTOR_ID` to identify the deployment’s one trusted actor.
The MCP service fails closed when either variable is not configured; it never
falls back to an ephemeral container filesystem. The existing
`MCP_SERVICE_TOKEN` transport authentication remains required outside
development.

For a locally operated CLI ledger, `var/task-control/task-events.jsonl` remains
the default. Both interfaces append the same `skyyrose-task-control.v1` events,
so an operator can inspect a task started via MCP with the CLI, and vice versa.
