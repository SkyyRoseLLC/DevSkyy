# Fashion Theme Team End-to-End Task Contract

Every Fashion Theme Team capability uses the `fashion-e2e-task-execution`
control layer. A task is complete only when its exact scope, source authority,
declared evidence, first-seen issues, resolutions, and independent disposition
are recorded against one candidate.

Each task has one named project manager. The manager owns the task plan,
dependency resolution, evidence completeness, issue recurrence, and handoff;
the manager cannot independently certify their own candidate.

The task roster names every participating agent or automation with its job
title, assigned capabilities, and owned deliverable. A remediation loop has its
own roster; a role cannot be called independent when it implemented the same
candidate it reviews.

The manager verifies every declared requirement. A missing, failed, stale, or
unknown item opens a remediation loop to the responsible Team. The task can
become `READY_FOR_FOUNDER_APPROVAL` only after re-verification finds all
requirements satisfied and no open scoped issue. It becomes `PASS` only after
explicit founder approval is recorded.

Strict E2E requires a multi-field scope, non-empty acceptance criteria,
non-empty evidence requirements including `independent_review`, and an
independent reviewer in the task roster. The project manager and independent
reviewer cannot record founder approval.

## Executable interfaces

In DevSkyy, use `scripts/task_control.py` for local, CI, and human-operated
workflow control. The `devskyy_task_*` MCP tools expose the same start,
evidence, independent-review, first-seen-issue, remediation, verification,
approval-request, and status operations to authenticated integrations. The MCP
service must set `TASK_CONTROL_MCP_LEDGER_PATH` to a durable shared volume; it
must fail closed rather than write task truth to an ephemeral container.

MCP cannot record founder approval or task completion. After manager
verification, the named founder records their decision through the trusted CLI
using `founder-approve --confirm-founder-approval`; that explicit action is
still distinct from deployment, promotion, merge, spend, or production
release authorization.

The task ledger is append-only. A recurrence links to the first issue through a
stable fingerprint; a new candidate invalidates stale proof but never deletes
the earlier record. Staging and production are separate task scopes.

## Minimum fields

```json
{
  "task_id": "task_...",
  "project_manager": "fashion_theme_team_project_manager",
  "agents": [{"name": "fashion_frontend_engineer", "job_title": "Frontend Engineer", "capabilities": ["responsive implementation", "motion fallback"], "owned_deliverable": "V2 card surface"}],
  "kind": "fashion_frontend_motion",
  "scope": {"candidate": "git:...", "route": "shop", "target": "staging"},
  "requirements": ["source_hash", "desktop_capture", "mobile_capture", "independent_review"],
  "acceptance_criteria": ["desktop and mobile framing pass", "no failed scoped resource"],
  "evidence": [{"name": "source_hash", "artifact": "...", "sha256": "..."}],
  "issues": [{"id": "issue_...", "fingerprint": "...", "state": "OPEN"}],
  "remediation_loops": [{"owner_team": "fashion-frontend-motion", "agents": [{"name": "fashion_motion_responsive_engineer", "job_title": "Motion and Responsive Engineer", "capabilities": ["mobile crop correction"]}], "state": "OPEN"}],
  "founder_approval": "PENDING",
  "verdict": "BLOCKED"
}
```

Do not store secrets, customer data, source bytes, or provider payloads in the
task record. The associated candidate-bound handoff still uses `preview.html`,
`contract.json`, and `evidence.json` where the capability requires them.
