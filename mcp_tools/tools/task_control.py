"""Strict E2E task-control tools for the DevSkyy MCP server.

The server-side controller is intentionally an execution surface rather than a
claiming surface.  It can create strict tasks, collect immutable evidence,
record first-seen defects, dispatch remediation, and request founder approval.
It does not expose a tool that grants founder approval or closes a task: those
two decisions remain an explicit human-controlled CLI action on a trusted
operator machine.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from mcp.types import ToolAnnotations
from pydantic import Field

from mcp_tools.security import secure_tool
from mcp_tools.server import mcp
from mcp_tools.types import BaseAgentInput
from skyyrose.core.task_control import TaskExecutionController, TaskLedger

_MCP_LEDGER_PATH_ENV = "TASK_CONTROL_MCP_LEDGER_PATH"
_MCP_ACTOR_ID_ENV = "TASK_CONTROL_MCP_ACTOR_ID"


def _ledger_path() -> Path:
    """Return the explicitly configured durable MCP ledger path.

    An HTTP service may run in an ephemeral container.  Falling back to the
    local CLI ledger would lose evidence when that container restarts, so the
    MCP interface fails closed until operations configure a durable volume.
    """
    configured = os.getenv(_MCP_LEDGER_PATH_ENV, "").strip()
    if not configured:
        raise RuntimeError(
            f"{_MCP_LEDGER_PATH_ENV} is required for task-control MCP writes; "
            "configure a durable JSONL volume first"
        )
    return Path(configured).expanduser().resolve()


def _controller() -> TaskExecutionController:
    return TaskExecutionController(TaskLedger(_ledger_path()))


def _actor_id() -> str:
    """Return the server-configured service identity for role-sensitive writes.

    The transport bearer authenticates a client to this MCP service, not a
    user-selected role carried in tool input.  Each role-sensitive deployment
    must therefore bind its service token and this identity to one issued task
    participant.  We fail closed rather than trusting an arbitrary `reviewer`
    or `project_manager` parameter from a caller.
    """
    actor = os.getenv(_MCP_ACTOR_ID_ENV, "").strip()
    if not actor:
        raise RuntimeError(
            f"{_MCP_ACTOR_ID_ENV} is required for task-control MCP writes; "
            "configure one trusted task-actor identity per service deployment"
        )
    return actor


def _require_actor(expected: str) -> str:
    actor = _actor_id()
    if actor != expected.strip():
        raise RuntimeError("configured MCP actor is not authorized for this task action")
    return actor


def _require_task_participant(controller: TaskExecutionController, task_id: str) -> str:
    actor = _actor_id()
    if not controller.is_task_participant(task_id, actor):
        raise RuntimeError("configured MCP actor is not an issued participant for this task")
    return actor


def _result(*, ok: bool, **payload: Any) -> str:
    """Keep MCP responses structured and never serialize exceptions as traces."""
    return json.dumps({"ok": ok, **payload}, sort_keys=True, ensure_ascii=False, default=str)


def _failure(exc: ValueError | RuntimeError) -> str:
    return _result(ok=False, error={"code": "TASK_CONTROL_REJECTED", "message": str(exc)})


class TaskAgentInput(BaseAgentInput):
    """A named participant whose capabilities are preserved in the task ledger."""

    name: str = Field(min_length=1, max_length=120)
    job_title: str = Field(min_length=1, max_length=160)
    capabilities: list[str] = Field(min_length=1, max_length=20)


class TaskStartInput(BaseAgentInput):
    """Strict E2E contract required before any operational side effect."""

    kind: str = Field(min_length=1, max_length=100)
    scope: dict[str, Any] = Field(min_length=2)
    project_manager: str = Field(min_length=1, max_length=120)
    requirements: list[str] = Field(min_length=1, max_length=40)
    acceptance_criteria: list[str] = Field(min_length=1, max_length=40)
    agents: list[TaskAgentInput] = Field(min_length=1, max_length=30)


class TaskIdInput(BaseAgentInput):
    """A task ledger identifier."""

    task_id: str = Field(min_length=1, max_length=128)


class TaskStatusInput(TaskIdInput):
    """Read-only task-status request."""


class TaskEvidenceInput(TaskIdInput):
    """One named evidence item; never include secret or source-byte payloads."""

    name: str = Field(min_length=1, max_length=120)
    details: dict[str, Any] = Field(min_length=1)


class TaskIndependentReviewInput(TaskIdInput):
    """A reviewer distinct from the issued project manager."""

    reviewer: str = Field(min_length=1, max_length=120)
    details: dict[str, Any] = Field(min_length=1)


class TaskIssueInput(TaskIdInput):
    """First-observed defect data and a stable invariant fingerprint."""

    category: str = Field(min_length=1, max_length=100)
    fingerprint: str = Field(min_length=1, max_length=200)
    summary: str = Field(min_length=1, max_length=1000)
    details: dict[str, Any] = Field(default_factory=dict)


class TaskResolveIssueInput(BaseAgentInput):
    """Evidence-backed issue resolution recorded by a named reviewer."""

    issue_id: str = Field(min_length=1, max_length=128)
    reviewer: str = Field(min_length=1, max_length=120)
    resolution: str = Field(min_length=1, max_length=2000)


class TaskRemediationInput(TaskIdInput):
    """Bounded correction loop routed to an accountable specialist Team."""

    owner_team: str = Field(min_length=1, max_length=240)
    objective: str = Field(min_length=1, max_length=2000)
    issue_id: str | None = Field(default=None, max_length=128)
    agents: list[TaskAgentInput] = Field(default_factory=list, max_length=30)


class TaskManagerInput(TaskIdInput):
    """Action requested by the project manager issued with the task."""

    project_manager: str = Field(min_length=1, max_length=120)


@mcp.tool(
    name="devskyy_task_start",
    annotations=ToolAnnotations(
        title="Strict E2E Task — Start",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
@secure_tool("task_start")
async def task_start(params: TaskStartInput) -> str:
    """Start a strict E2E task with its scope, manager, roster, and proof contract."""
    try:
        _require_actor(params.project_manager)
        task_id = _controller().begin_e2e(
            kind=params.kind,
            scope=params.scope,
            project_manager=params.project_manager,
            requirements=params.requirements,
            acceptance_criteria=params.acceptance_criteria,
            agents=[agent.model_dump(exclude={"response_format"}) for agent in params.agents],
        )
        return _result(ok=True, task_id=task_id, disposition="IN_PROGRESS")
    except (ValueError, RuntimeError) as exc:
        return _failure(exc)


@mcp.tool(
    name="devskyy_task_record_evidence",
    annotations=ToolAnnotations(
        title="Strict E2E Task — Record Evidence",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
@secure_tool("task_record_evidence")
async def task_record_evidence(params: TaskEvidenceInput) -> str:
    """Append one named evidence item to a strict E2E task."""
    try:
        controller = _controller()
        actor = _require_task_participant(controller, params.task_id)
        event_id = controller.record_evidence(
            params.task_id, params.name, params.details, recorded_by=actor
        )
        return _result(ok=True, event_id=event_id, task_id=params.task_id)
    except (ValueError, RuntimeError) as exc:
        return _failure(exc)


@mcp.tool(
    name="devskyy_task_record_independent_review",
    annotations=ToolAnnotations(
        title="Strict E2E Task — Record Independent Review",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
@secure_tool("task_record_independent_review")
async def task_record_independent_review(params: TaskIndependentReviewInput) -> str:
    """Record the required independent review; manager self-review is rejected."""
    try:
        _require_actor(params.reviewer)
        event_id = _controller().record_independent_review(
            params.task_id, reviewer=params.reviewer, evidence=params.details
        )
        return _result(ok=True, event_id=event_id, task_id=params.task_id)
    except (ValueError, RuntimeError) as exc:
        return _failure(exc)


@mcp.tool(
    name="devskyy_task_open_issue",
    annotations=ToolAnnotations(
        title="Strict E2E Task — Log First-Seen Issue",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
@secure_tool("task_open_issue")
async def task_open_issue(params: TaskIssueInput) -> str:
    """Log the first occurrence of an invariant failure, or link its recurrence."""
    try:
        controller = _controller()
        _require_task_participant(controller, params.task_id)
        issue_id = controller.record_issue(
            params.task_id,
            category=params.category,
            fingerprint=params.fingerprint,
            summary=params.summary,
            details=params.details,
        )
        return _result(ok=True, issue_id=issue_id, task_id=params.task_id)
    except (ValueError, RuntimeError) as exc:
        return _failure(exc)


@mcp.tool(
    name="devskyy_task_resolve_issue",
    annotations=ToolAnnotations(
        title="Strict E2E Task — Resolve Issue",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
@secure_tool("task_resolve_issue")
async def task_resolve_issue(params: TaskResolveIssueInput) -> str:
    """Record the reviewed resolution for a first-seen issue; it does not complete the task."""
    try:
        _require_actor(params.reviewer)
        _controller().resolve_issue(
            params.issue_id, reviewer=params.reviewer, resolution=params.resolution
        )
        return _result(ok=True, issue_id=params.issue_id)
    except (ValueError, RuntimeError) as exc:
        return _failure(exc)


@mcp.tool(
    name="devskyy_task_open_remediation_loop",
    annotations=ToolAnnotations(
        title="Strict E2E Task — Open Remediation Loop",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
@secure_tool("task_open_remediation_loop")
async def task_open_remediation_loop(params: TaskRemediationInput) -> str:
    """Assign a correction loop with its accountable Team and named roster."""
    try:
        controller = _controller()
        _require_actor(controller.project_manager_for_task(params.task_id))
        loop_id = controller.open_remediation_loop(
            params.task_id,
            owner_team=params.owner_team,
            objective=params.objective,
            issue_id=params.issue_id,
            agents=[agent.model_dump(exclude={"response_format"}) for agent in params.agents],
        )
        return _result(ok=True, loop_id=loop_id, task_id=params.task_id)
    except (ValueError, RuntimeError) as exc:
        return _failure(exc)


@mcp.tool(
    name="devskyy_task_verify",
    annotations=ToolAnnotations(
        title="Strict E2E Task — Manager Verify",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
@secure_tool("task_verify")
async def task_verify(params: TaskManagerInput) -> str:
    """Run the issued manager's deterministic evidence and issue audit."""
    try:
        _require_actor(params.project_manager)
        decision = _controller().verify(params.task_id, project_manager=params.project_manager)
        return _result(
            ok=True,
            task_id=params.task_id,
            disposition=(
                "READY_FOR_FOUNDER_APPROVAL" if decision.ready_for_founder_approval else "BLOCKED"
            ),
            completion={
                "allowed": decision.allowed,
                "missing_requirements": list(decision.missing_requirements),
                "open_issue_ids": list(decision.open_issue_ids),
                "ready_for_founder_approval": decision.ready_for_founder_approval,
                "founder_approved": decision.founder_approved,
            },
        )
    except (ValueError, RuntimeError) as exc:
        return _failure(exc)


@mcp.tool(
    name="devskyy_task_request_founder_approval",
    annotations=ToolAnnotations(
        title="Strict E2E Task — Request Founder Approval",
        readOnlyHint=False,
        destructiveHint=False,
        idempotentHint=False,
        openWorldHint=False,
    ),
)
@secure_tool("task_request_founder_approval")
async def task_request_founder_approval(params: TaskManagerInput) -> str:
    """Raise an evidence-complete task for human founder approval; does not approve it."""
    try:
        _require_actor(params.project_manager)
        requested = _controller().request_founder_approval(
            params.task_id, project_manager=params.project_manager
        )
        return _result(
            ok=requested,
            task_id=params.task_id,
            disposition="READY_FOR_FOUNDER_APPROVAL" if requested else "BLOCKED",
        )
    except (ValueError, RuntimeError) as exc:
        return _failure(exc)


@mcp.tool(
    name="devskyy_task_status",
    annotations=ToolAnnotations(
        title="Strict E2E Task — Status",
        readOnlyHint=True,
        destructiveHint=False,
        idempotentHint=True,
        openWorldHint=False,
    ),
)
@secure_tool("task_status")
async def task_status(params: TaskStatusInput) -> str:
    """Read the current task contract, evidence, first-seen issues, and eligibility."""
    try:
        return _result(ok=True, **_controller().snapshot(params.task_id))
    except (ValueError, RuntimeError) as exc:
        return _failure(exc)


__all__ = [
    "TaskAgentInput",
    "TaskEvidenceInput",
    "TaskIdInput",
    "TaskIndependentReviewInput",
    "TaskIssueInput",
    "TaskManagerInput",
    "TaskRemediationInput",
    "TaskResolveIssueInput",
    "TaskStartInput",
    "TaskStatusInput",
    "task_open_issue",
    "task_open_remediation_loop",
    "task_record_evidence",
    "task_record_independent_review",
    "task_request_founder_approval",
    "task_resolve_issue",
    "task_start",
    "task_status",
    "task_verify",
]
