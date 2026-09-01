"""Tests for the durable SkyyRose Task Execution Controller.

Rendering, deployment, catalog work, and future controlled entrypoints use the
same first-observed-issue and evidence rules. It contains metadata and hashes
only; credentials and asset bytes are never persisted.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from skyyrose.core.task_control import LedgerIntegrityError, TaskExecutionController, TaskLedger


def _controller(tmp_path: Path) -> TaskExecutionController:
    return TaskExecutionController(TaskLedger(tmp_path / "task-events.jsonl"))


def test_completion_requires_all_declared_evidence(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    task_id = controller.begin(
        kind="staging_deploy",
        scope={"theme": "v2", "target": "staging"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("source_hash", "desktop_capture", "mobile_capture"),
    )
    controller.record_evidence(task_id, "source_hash", {"sha256": "a" * 64})

    decision = controller.can_complete(task_id)

    assert decision.allowed is False
    assert decision.missing_requirements == ("desktop_capture", "mobile_capture")


def test_task_start_requires_and_records_a_project_manager(tmp_path: Path) -> None:
    import pytest

    controller = _controller(tmp_path)
    with pytest.raises(ValueError, match="project manager"):
        controller.begin(kind="catalog_sync", scope={"sku": "br-008"}, project_manager="")

    task_id = controller.begin(
        kind="catalog_sync",
        scope={"sku": "br-008"},
        project_manager="fashion_theme_team_project_manager",
    )

    assert controller.ledger.events()[-1]["task_id"] == task_id
    assert controller.ledger.events()[-1]["project_manager"] == "fashion_theme_team_project_manager"
    assert controller.ledger.events()[-1]["agents"] == [
        {
            "name": "fashion_theme_team_project_manager",
            "job_title": "Project Manager",
            "capabilities": [
                "scope ownership",
                "dependency coordination",
                "evidence verification",
                "remediation-loop routing",
                "founder-approval handoff",
            ],
        }
    ]


def test_task_roster_requires_named_title_and_capabilities(tmp_path: Path) -> None:
    import pytest

    controller = _controller(tmp_path)
    with pytest.raises(ValueError, match="name, job_title, and capabilities"):
        controller.begin(
            kind="product_render",
            scope={"sku": "br-008"},
            project_manager="fashion_theme_team_project_manager",
            agents=({"name": "visual_qa"},),
        )

    task_id = controller.begin(
        kind="product_render",
        scope={"sku": "br-008"},
        project_manager="fashion_theme_team_project_manager",
        agents=(
            {
                "name": "product_fidelity_reviewer",
                "job_title": "Product Fidelity Reviewer",
                "capabilities": ["SKU construction review", "patch and lettering verification"],
            },
        ),
    )
    roster = controller.ledger.events()[-1]["agents"]
    assert controller.ledger.events()[-1]["task_id"] == task_id
    assert roster[1]["job_title"] == "Product Fidelity Reviewer"


def test_first_issue_is_retained_and_repeat_is_linked_to_it(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    task_id = controller.begin(
        kind="product_render",
        scope={"sku": "br-008", "style": "on-model", "view": "front"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("candidate", "visual_qa"),
    )

    first = controller.record_issue(
        task_id,
        category="product_fidelity",
        summary="rose artwork leaked into the plain 0",
        fingerprint="br-008-front-zero-leak-v1",
    )
    repeated = controller.record_issue(
        task_id,
        category="product_fidelity",
        summary="same defect observed again",
        fingerprint="br-008-front-zero-leak-v1",
    )

    assert repeated == first
    events = controller.ledger.events()
    assert [event["event"] for event in events] == [
        "task_started",
        "issue_opened",
        "issue_seen_again",
    ]
    assert events[-1]["issue_id"] == first


def test_open_issue_blocks_completion_until_resolved_and_evidence_exists(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    task_id = controller.begin(
        kind="product_render",
        scope={"sku": "br-008", "style": "on-model", "view": "front"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("candidate", "visual_qa"),
    )
    controller.record_evidence(task_id, "candidate", {"sha256": "b" * 64})
    controller.record_evidence(task_id, "visual_qa", {"reviewer": "founder", "passed": True})
    issue_id = controller.record_issue(
        task_id,
        category="product_fidelity",
        summary="patch size inconsistent",
        fingerprint="br-008-patch-footprint-v1",
    )

    assert controller.can_complete(task_id).allowed is False
    controller.resolve_issue(issue_id, reviewer="founder", resolution="corrected source bound")

    decision = controller.verify(task_id, project_manager="fashion_theme_team_project_manager")
    assert decision.ready_for_founder_approval is True
    assert (
        controller.request_founder_approval(
            task_id, project_manager="fashion_theme_team_project_manager"
        )
        is True
    )
    controller.approve_founder(task_id, founder="founder")
    assert controller.can_complete(task_id).allowed is True
    assert controller.complete(task_id, reviewer="founder") is True


def test_new_task_in_same_scope_is_blocked_by_open_first_seen_issue(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    scope = {"sku": "br-008", "style": "on-model", "view": "front"}
    task_id = controller.begin(
        kind="product_render",
        scope=scope,
        project_manager="fashion_theme_team_project_manager",
        requirements=("candidate",),
    )
    controller.record_issue(
        task_id,
        category="product_fidelity",
        summary="wrong numeral fill",
        fingerprint="br-008-number-fill-v1",
    )

    next_task = controller.begin(
        kind="product_render",
        scope=scope,
        project_manager="fashion_theme_team_project_manager",
        requirements=("candidate",),
    )

    assert controller.is_scope_blocked(next_task) is True


def test_manager_issues_a_remediation_loop_then_reverifies_before_founder_approval(
    tmp_path: Path,
) -> None:
    controller = _controller(tmp_path)
    task_id = controller.begin(
        kind="staging_deploy",
        scope={"theme": "v2", "target": "staging"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("source_hash", "desktop_capture"),
    )
    issue_id = controller.record_issue(
        task_id,
        category="visual_regression",
        summary="stone frame clips the product at 390px",
        fingerprint="v2-card-stone-frame-390px-v1",
    )

    blocked = controller.verify(task_id, project_manager="fashion_theme_team_project_manager")
    loop_id = controller.open_remediation_loop(
        task_id,
        owner_team="fashion-frontend-motion + fashion-visual-commerce-qa",
        objective="Repair the 390px crop and attach a fresh mobile capture.",
        issue_id=issue_id,
    )

    assert blocked.ready_for_founder_approval is False
    assert loop_id.startswith("loop_")
    controller.record_evidence(task_id, "source_hash", {"sha256": "c" * 64})
    controller.record_evidence(task_id, "desktop_capture", {"path": "desktop.png"})
    controller.resolve_issue(issue_id, reviewer="independent_qa", resolution="verified at 390px")
    assert (
        controller.verify(
            task_id, project_manager="fashion_theme_team_project_manager"
        ).ready_for_founder_approval
        is True
    )


def test_project_manager_cannot_self_approve_as_founder(tmp_path: Path) -> None:
    import pytest

    controller = _controller(tmp_path)
    task_id = controller.begin(
        kind="staging_deploy",
        scope={"theme": "v2", "target": "staging"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("source_hash",),
    )
    controller.record_evidence(task_id, "source_hash", {"sha256": "d" * 64})
    assert (
        controller.request_founder_approval(
            task_id, project_manager="fashion_theme_team_project_manager"
        )
        is True
    )

    with pytest.raises(ValueError, match="cannot record founder approval"):
        controller.approve_founder(task_id, founder="fashion_theme_team_project_manager")


def test_strict_e2e_requires_contract_reviewer_and_separate_founder(tmp_path: Path) -> None:
    import pytest

    controller = _controller(tmp_path)
    with pytest.raises(ValueError, match="independent_review"):
        controller.begin_e2e(
            kind="staging_deploy",
            scope={"theme": "v2", "target": "staging"},
            project_manager="fashion_theme_team_project_manager",
            requirements=("source_hash",),
            acceptance_criteria=("desktop and mobile evidence pass",),
            agents=(),
        )

    task_id = controller.begin_e2e(
        kind="staging_deploy",
        scope={"theme": "v2", "target": "staging"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("source_hash", "independent_review"),
        acceptance_criteria=("desktop and mobile evidence pass",),
        agents=(
            {
                "name": "fashion_visual_commerce_qa",
                "job_title": "Visual Commerce QA Reviewer",
                "capabilities": ["desktop and mobile review"],
            },
        ),
    )
    controller.record_evidence(task_id, "source_hash", {"sha256": "e" * 64})
    controller.record_independent_review(
        task_id,
        reviewer="fashion_visual_commerce_qa",
        evidence={
            "capture": "review.png",
            "criteria": {"desktop and mobile evidence pass": "PASS"},
        },
    )
    assert (
        controller.request_founder_approval(
            task_id, project_manager="fashion_theme_team_project_manager"
        )
        is True
    )

    with pytest.raises(ValueError, match="independent reviewer cannot"):
        controller.approve_founder(task_id, founder="fashion_visual_commerce_qa")
    controller.approve_founder(task_id, founder="founder")
    assert controller.complete(task_id, reviewer="fashion_theme_team_project_manager") is True


def test_strict_e2e_rejects_forged_independent_review_evidence(tmp_path: Path) -> None:
    """A manager cannot satisfy review evidence by relabeling a generic receipt."""
    import pytest

    controller = _controller(tmp_path)
    task_id = controller.begin_e2e(
        kind="staging_deploy",
        scope={"theme": "v2", "target": "staging"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("source_hash", "independent_review"),
        acceptance_criteria=("Independent review must pass.",),
        agents=(
            {
                "name": "fashion_visual_commerce_qa",
                "job_title": "Visual Commerce QA Reviewer",
                "capabilities": ["independent review"],
            },
        ),
    )
    controller.record_evidence(task_id, "source_hash", {"sha256": "f" * 64})

    with pytest.raises(ValueError, match="must use record_independent_review"):
        controller.record_evidence(
            task_id,
            "independent_review",
            {"reviewer": "fashion_theme_team_project_manager", "outcome": "PASS"},
        )
    with pytest.raises(ValueError, match="named independent reviewer"):
        controller.record_independent_review(
            task_id,
            reviewer="unrostered_reviewer",
            evidence={"verdict": "PASS"},
        )

    assert (
        controller.request_founder_approval(
            task_id, project_manager="fashion_theme_team_project_manager"
        )
        is False
    )


def test_strict_e2e_requires_an_independent_criterion_by_criterion_attestation(
    tmp_path: Path,
) -> None:
    """A passing label is insufficient without every issued criterion mapped to PASS."""
    import pytest

    controller = _controller(tmp_path)
    task_id = controller.begin_e2e(
        kind="staging_deploy",
        scope={"theme": "v2", "target": "staging"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("source_hash", "independent_review"),
        acceptance_criteria=("Desktop proof passes.", "390px proof passes."),
        agents=(
            {
                "name": "fashion_visual_commerce_qa",
                "job_title": "Visual Commerce QA Reviewer",
                "capabilities": ["independent visual review"],
            },
        ),
    )

    with pytest.raises(ValueError, match="every acceptance criterion"):
        controller.record_independent_review(
            task_id,
            reviewer="fashion_visual_commerce_qa",
            evidence={"criteria": {"Desktop proof passes.": "PASS"}},
        )


def test_task_ledger_rejects_sensitive_or_unbounded_metadata(tmp_path: Path) -> None:
    """Task receipts cannot become a secret, payload, or unbounded-data store."""
    import pytest

    controller = _controller(tmp_path)
    task_id = controller.begin(
        kind="catalog_sync",
        scope={"sku": "br-009"},
        project_manager="fashion_theme_team_project_manager",
    )

    with pytest.raises(ValueError, match="sensitive key"):
        controller.record_evidence(task_id, "source_hash", {"api_key": "not-allowed"})
    with pytest.raises(ValueError, match="credential-like value"):
        controller.record_evidence(
            task_id,
            "source_hash",
            {"opaque_note": "Bearer this-is-a-test-secret-value"},
        )
    with pytest.raises(ValueError, match="credential-like value"):
        controller.record_evidence(
            task_id,
            "source_hash",
            {"references": [{"note": "authorization=not-allowed"}]},
        )
    with pytest.raises(ValueError, match="too long"):
        controller.record_evidence(task_id, "source_hash", {"note": "x" * 4097})


def test_cli_requires_explicit_confirmation_for_founder_approval(
    tmp_path: Path, monkeypatch
) -> None:
    """A typed founder identity alone cannot finalize a strict task by accident."""
    from scripts.task_control import main

    ledger_path = tmp_path / "task-events.jsonl"
    controller = TaskExecutionController(TaskLedger(ledger_path))
    task_id = controller.begin_e2e(
        kind="staging_deploy",
        scope={"theme": "v2", "target": "staging"},
        project_manager="fashion_theme_team_project_manager",
        requirements=("source_hash", "independent_review"),
        acceptance_criteria=("The staging capture passes review.",),
        agents=(
            {
                "name": "fashion_visual_commerce_qa",
                "job_title": "Visual Commerce QA Reviewer",
                "capabilities": ["independent review"],
            },
        ),
    )
    controller.record_evidence(task_id, "source_hash", {"sha256": "f" * 64})
    controller.record_independent_review(
        task_id,
        reviewer="fashion_visual_commerce_qa",
        evidence={
            "capture": "staging.png",
            "criteria": {"The staging capture passes review.": "PASS"},
        },
    )
    assert (
        controller.request_founder_approval(
            task_id, project_manager="fashion_theme_team_project_manager"
        )
        is True
    )

    assert (
        main(
            [
                "--ledger",
                str(ledger_path),
                "founder-approve",
                "--task-id",
                task_id,
                "--founder",
                "founder",
            ]
        )
        == 2
    )
    assert controller.can_complete(task_id).founder_approved is False

    monkeypatch.setenv("TASK_CONTROL_CLI_ACTOR_ID", "founder")
    monkeypatch.setenv("TASK_CONTROL_FOUNDER_ID", "founder")

    assert (
        main(
            [
                "--ledger",
                str(ledger_path),
                "founder-approve",
                "--task-id",
                task_id,
                "--founder",
                "founder",
                "--confirm-founder-approval",
            ]
        )
        == 0
    )
    assert controller.can_complete(task_id).founder_approved is True


def test_documented_direct_cli_command_resolves_the_project_package(tmp_path: Path) -> None:
    """`python scripts/task_control.py` must work without a PYTHONPATH workaround."""
    project_root = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [
            sys.executable,
            "scripts/task_control.py",
            "--ledger",
            str(tmp_path / "task-events.jsonl"),
            "status",
            "--task-id",
            "task_missing",
        ],
        cwd=project_root,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 2
    assert "unknown task id" in result.stderr
    assert "ModuleNotFoundError" not in result.stderr


def test_malformed_task_ledger_fails_closed_instead_of_hiding_an_open_issue(tmp_path: Path) -> None:
    """Corruption cannot be interpreted as an empty, approval-safe ledger."""
    import pytest

    ledger_path = tmp_path / "task-events.jsonl"
    ledger_path.write_text('{"schema":"skyyrose-task-control.v1"}\n{"event":"issue_opened"')

    with pytest.raises(LedgerIntegrityError, match="incomplete|malformed"):
        TaskLedger(ledger_path).events()
