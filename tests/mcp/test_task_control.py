"""End-to-end tests for the strict task-control MCP surface."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from mcp_tools.server import mcp
from mcp_tools.tools import task_control as task_control_tools
from mcp_tools.tools.task_control import (
    TaskEvidenceInput,
    TaskIndependentReviewInput,
    TaskStartInput,
    TaskStatusInput,
    task_record_evidence,
    task_record_independent_review,
    task_start,
    task_status,
)
from skyyrose.core.task_control import TaskExecutionController, TaskLedger


def _controller(tmp_path: Path) -> TaskExecutionController:
    return TaskExecutionController(TaskLedger(tmp_path / "task-events.jsonl"))


def test_mcp_writes_fail_closed_without_a_durable_ledger_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restartable HTTP container must not become the only source of task truth."""
    monkeypatch.delenv("TASK_CONTROL_MCP_LEDGER_PATH", raising=False)

    with pytest.raises(RuntimeError, match="TASK_CONTROL_MCP_LEDGER_PATH"):
        task_control_tools._ledger_path()

    monkeypatch.delenv("TASK_CONTROL_MCP_ACTOR_ID", raising=False)
    with pytest.raises(RuntimeError, match="TASK_CONTROL_MCP_ACTOR_ID"):
        task_control_tools._actor_id()


@pytest.mark.asyncio
async def test_mcp_can_start_strict_task_and_report_machine_readable_blockers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP starts the exact strict task contract, then exposes its real state."""
    controller = _controller(tmp_path)
    monkeypatch.setattr(task_control_tools, "_controller", lambda: controller)
    monkeypatch.setattr(
        task_control_tools, "_actor_id", lambda: "fashion_theme_team_project_manager"
    )
    params = TaskStartInput(
        kind="staging_deploy",
        scope={"theme": "v2", "target": "staging"},
        project_manager="fashion_theme_team_project_manager",
        requirements=["source_hash", "independent_review"],
        acceptance_criteria=["Desktop and mobile proof passes."],
        agents=[
            {
                "name": "fashion_visual_commerce_qa",
                "job_title": "Visual Commerce QA Reviewer",
                "capabilities": ["independent visual review"],
            }
        ],
    )

    started = json.loads(await task_start(params))
    status = json.loads(await task_status(TaskStatusInput(task_id=started["task_id"])))

    assert started["ok"] is True
    assert status["ok"] is True
    assert status["task"]["project_manager"] == "fashion_theme_team_project_manager"
    assert status["completion"]["allowed"] is False
    assert status["completion"]["missing_requirements"] == [
        "source_hash",
        "independent_review",
    ]


@pytest.mark.asyncio
async def test_mcp_evidence_and_independent_review_leave_founder_approval_manual(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MCP can prepare review, but may not grant final founder approval."""
    controller = _controller(tmp_path)
    monkeypatch.setattr(task_control_tools, "_controller", lambda: controller)
    monkeypatch.setattr(
        task_control_tools, "_actor_id", lambda: "fashion_theme_team_project_manager"
    )
    task_id = controller.begin_e2e(
        kind="product_render",
        scope={"sku": "br-009", "target": "review-only"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("candidate", "independent_review"),
        acceptance_criteria=("Source-bound candidate passes fidelity review.",),
        agents=(
            {
                "name": "product_fidelity_independent_reviewer",
                "job_title": "Product Fidelity Independent Reviewer",
                "capabilities": ["SKU fidelity review"],
            },
        ),
    )

    evidence = json.loads(
        await task_record_evidence(
            TaskEvidenceInput(
                task_id=task_id,
                name="candidate",
                details={"sha256": "a" * 64, "actor": "oai_render_release_controller"},
            )
        )
    )
    monkeypatch.setattr(
        task_control_tools, "_actor_id", lambda: "product_fidelity_independent_reviewer"
    )
    review = json.loads(
        await task_record_independent_review(
            TaskIndependentReviewInput(
                task_id=task_id,
                reviewer="product_fidelity_independent_reviewer",
                details={
                    "candidate_sha256": "a" * 64,
                    "verdict": "PASS",
                    "criteria": {"Source-bound candidate passes fidelity review.": "PASS"},
                },
            )
        )
    )
    status = json.loads(await task_status(TaskStatusInput(task_id=task_id)))

    assert evidence["ok"] is True
    assert review["ok"] is True
    assert status["completion"]["ready_for_founder_approval"] is True
    assert status["completion"]["founder_approved"] is False
    assert not hasattr(task_control_tools, "task_approve_founder")
    assert not hasattr(task_control_tools, "task_complete")


@pytest.mark.asyncio
async def test_mcp_rejects_a_manager_forging_independent_review_as_generic_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The generic MCP evidence tool cannot impersonate the review operation."""
    controller = _controller(tmp_path)
    monkeypatch.setattr(task_control_tools, "_controller", lambda: controller)
    monkeypatch.setattr(
        task_control_tools, "_actor_id", lambda: "fashion_theme_team_project_manager"
    )
    task_id = controller.begin_e2e(
        kind="staging_deploy",
        scope={"theme": "v2", "target": "staging"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("source_hash", "independent_review"),
        acceptance_criteria=("Independent review passes.",),
        agents=(
            {
                "name": "fashion_visual_commerce_qa",
                "job_title": "Visual Commerce QA Reviewer",
                "capabilities": ["independent review"],
            },
        ),
    )

    forged = json.loads(
        await task_record_evidence(
            TaskEvidenceInput(
                task_id=task_id,
                name="independent_review",
                details={
                    "reviewer": "fashion_theme_team_project_manager",
                    "outcome": "PASS",
                },
            )
        )
    )
    status = json.loads(await task_status(TaskStatusInput(task_id=task_id)))

    assert forged["ok"] is False
    assert forged["error"]["code"] == "TASK_CONTROL_REJECTED"
    assert status["completion"]["missing_requirements"] == [
        "source_hash",
        "independent_review",
    ]


@pytest.mark.asyncio
async def test_mcp_rejects_body_claims_that_do_not_match_the_configured_actor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A shared transport token cannot select a reviewer identity in tool input."""
    controller = _controller(tmp_path)
    monkeypatch.setattr(task_control_tools, "_controller", lambda: controller)
    monkeypatch.setattr(
        task_control_tools, "_actor_id", lambda: "fashion_theme_team_project_manager"
    )
    task_id = controller.begin_e2e(
        kind="staging_deploy",
        scope={"theme": "v2", "target": "staging"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("source_hash", "independent_review"),
        acceptance_criteria=("Independent review passes.",),
        agents=(
            {
                "name": "fashion_visual_commerce_qa",
                "job_title": "Visual Commerce QA Reviewer",
                "capabilities": ["independent review"],
            },
        ),
    )

    rejected = json.loads(
        await task_record_independent_review(
            TaskIndependentReviewInput(
                task_id=task_id,
                reviewer="fashion_visual_commerce_qa",
                details={"criteria": {"Independent review passes.": "PASS"}},
            )
        )
    )

    assert rejected["ok"] is False
    assert rejected["error"]["code"] == "TASK_CONTROL_REJECTED"


@pytest.mark.asyncio
async def test_task_control_tools_are_registered_on_the_mcp_server() -> None:
    """The integration is discoverable through the actual MCP tool registry."""
    names = {tool.name for tool in await mcp.list_tools()}

    assert {
        "devskyy_task_start",
        "devskyy_task_record_evidence",
        "devskyy_task_record_independent_review",
        "devskyy_task_open_issue",
        "devskyy_task_resolve_issue",
        "devskyy_task_open_remediation_loop",
        "devskyy_task_verify",
        "devskyy_task_request_founder_approval",
        "devskyy_task_status",
    } <= names
