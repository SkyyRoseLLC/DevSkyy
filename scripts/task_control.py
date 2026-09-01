#!/usr/bin/env python3
"""Operate the evidence-first task control ledger.

Examples (all arguments are data, not shell commands):

  python scripts/task_control.py start --kind staging_deploy \
    --scope-json '{"theme":"v2","target":"staging"}' \
    --require source_hash --require desktop_capture --require mobile_capture
  python scripts/task_control.py evidence --task-id task_... --name source_hash \
    --details-json '{"sha256":"..."}'
  python scripts/task_control.py issue --task-id task_... --category visual_regression \
    --fingerprint card-crop-v1 --summary 'Stone frame clips the product'
  python scripts/task_control.py status --task-id task_...

The default ledger is ignored runtime data under ``var/task-control``.  Pass
``--ledger PATH`` to use a project-, branch-, or environment-specific ledger.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

# Direct script execution puts ``scripts/`` on sys.path, not the repository
# root. Keep the documented ``python scripts/task_control.py`` entrypoint
# self-contained without requiring an operator to set PYTHONPATH manually.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from skyyrose.core.task_control import TaskExecutionController, TaskLedger

DEFAULT_LEDGER = PROJECT_ROOT / "var/task-control/task-events.jsonl"
_CLI_ACTOR_ID_ENV = "TASK_CONTROL_CLI_ACTOR_ID"
_FOUNDER_ID_ENV = "TASK_CONTROL_FOUNDER_ID"


def _object(value: str, *, label: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except json.JSONDecodeError as exc:
        raise argparse.ArgumentTypeError(f"{label} must be valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict) or not decoded:
        raise argparse.ArgumentTypeError(f"{label} must be a non-empty JSON object")
    return decoded


def _controller(args: argparse.Namespace) -> TaskExecutionController:
    return TaskExecutionController(TaskLedger(Path(args.ledger)))


def _actor_id() -> str:
    """Return the operator identity bound by the trusted CLI environment."""
    actor = os.getenv(_CLI_ACTOR_ID_ENV, "").strip()
    if not actor:
        raise RuntimeError(
            f"{_CLI_ACTOR_ID_ENV} is required for task-control write actions; "
            "configure the trusted operator identity first"
        )
    return actor


def _require_actor(expected: str) -> str:
    actor = _actor_id()
    if actor != expected.strip():
        raise RuntimeError("configured CLI actor is not authorized for this task action")
    return actor


def _require_task_participant(controller: TaskExecutionController, task_id: str) -> str:
    actor = _actor_id()
    if not controller.is_task_participant(task_id, actor):
        raise RuntimeError("configured CLI actor is not an issued participant for this task")
    return actor


def _require_founder(founder: str) -> None:
    configured_founder = os.getenv(_FOUNDER_ID_ENV, "").strip()
    if not configured_founder:
        raise RuntimeError(f"{_FOUNDER_ID_ENV} is required before founder approval can be recorded")
    _require_actor(founder)
    if founder.strip() != configured_founder:
        raise RuntimeError("configured founder identity does not match the approval request")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ledger", default=str(DEFAULT_LEDGER), help="Path to append-only JSONL ledger."
    )
    commands = parser.add_subparsers(dest="command", required=True)

    start = commands.add_parser("start", help="Record a task before work begins.")
    start.add_argument("--kind", required=True)
    start.add_argument(
        "--project-manager",
        required=True,
        help="Named manager accountable for scope, evidence, issues, and handoff.",
    )
    start.add_argument("--scope-json", required=True)
    start.add_argument("--require", action="append", default=[])
    start.add_argument("--acceptance", action="append", default=[])
    start.add_argument(
        "--strict-e2e",
        action="store_true",
        help="Require acceptance criteria and a named independent reviewer before approval.",
    )
    start.add_argument(
        "--agent-json",
        action="append",
        default=[],
        help='Task participant JSON: {"name":"...","job_title":"...","capabilities":["..."]}.',
    )

    evidence = commands.add_parser("evidence", help="Attach one evidence item to a task.")
    evidence.add_argument("--task-id", required=True)
    evidence.add_argument("--name", required=True)
    evidence.add_argument("--details-json", required=True)

    independent_review = commands.add_parser(
        "independent-review", help="Attach a passing independent review to a strict E2E task."
    )
    independent_review.add_argument("--task-id", required=True)
    independent_review.add_argument("--reviewer", required=True)
    independent_review.add_argument("--details-json", required=True)

    issue = commands.add_parser("issue", help="Open or link a first-seen issue.")
    issue.add_argument("--task-id", required=True)
    issue.add_argument("--category", required=True)
    issue.add_argument("--fingerprint", required=True)
    issue.add_argument("--summary", required=True)
    issue.add_argument("--details-json", default="{}")

    resolve = commands.add_parser(
        "resolve", help="Resolve a tracked issue with an accountable note."
    )
    resolve.add_argument("--issue-id", required=True)
    resolve.add_argument("--reviewer", required=True)
    resolve.add_argument("--resolution", required=True)

    remediation = commands.add_parser(
        "remediate", help="Issue a managed correction loop to a Team."
    )
    remediation.add_argument("--task-id", required=True)
    remediation.add_argument("--owner-team", required=True)
    remediation.add_argument("--objective", required=True)
    remediation.add_argument("--issue-id")
    remediation.add_argument(
        "--agent-json",
        action="append",
        default=[],
        help="Participant JSON for the remediation loop.",
    )

    verify = commands.add_parser(
        "verify", help="Run the issued manager's deterministic evidence audit."
    )
    verify.add_argument("--task-id", required=True)
    verify.add_argument("--project-manager", required=True)

    request_approval = commands.add_parser(
        "request-founder-approval", help="Raise an evidence-complete task for founder approval."
    )
    request_approval.add_argument("--task-id", required=True)
    request_approval.add_argument("--project-manager", required=True)

    founder_approve = commands.add_parser(
        "founder-approve", help="Record the founder's explicit approval after manager verification."
    )
    founder_approve.add_argument("--task-id", required=True)
    founder_approve.add_argument("--founder", required=True)
    founder_approve.add_argument(
        "--confirm-founder-approval",
        action="store_true",
        help="Required acknowledgement that this is the founder's explicit approval.",
    )

    status = commands.add_parser("status", help="Show whether the task is eligible for completion.")
    status.add_argument("--task-id", required=True)

    complete = commands.add_parser(
        "complete", help="Record completion only if evidence and issues allow it."
    )
    complete.add_argument("--task-id", required=True)
    complete.add_argument("--reviewer", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    controller = _controller(args)
    try:
        if args.command == "start":
            _require_actor(args.project_manager)
            kwargs = {
                "kind": args.kind,
                "scope": _object(args.scope_json, label="--scope-json"),
                "project_manager": args.project_manager,
                "requirements": args.require,
                "agents": [_object(value, label="--agent-json") for value in args.agent_json],
            }
            task_id = (
                controller.begin_e2e(acceptance_criteria=args.acceptance, **kwargs)
                if args.strict_e2e
                else controller.begin(acceptance_criteria=args.acceptance, **kwargs)
            )
            print(task_id)
            return 0
        if args.command == "evidence":
            actor = _require_task_participant(controller, args.task_id)
            event_id = controller.record_evidence(
                args.task_id,
                args.name,
                _object(args.details_json, label="--details-json"),
                recorded_by=actor,
            )
            print(event_id)
            return 0
        if args.command == "independent-review":
            _require_actor(args.reviewer)
            event_id = controller.record_independent_review(
                args.task_id,
                reviewer=args.reviewer,
                evidence=_object(args.details_json, label="--details-json"),
            )
            print(event_id)
            return 0
        if args.command == "issue":
            _require_task_participant(controller, args.task_id)
            issue_id = controller.record_issue(
                args.task_id,
                category=args.category,
                fingerprint=args.fingerprint,
                summary=args.summary,
                details=_object(args.details_json, label="--details-json"),
            )
            print(issue_id)
            return 0
        if args.command == "resolve":
            _require_actor(args.reviewer)
            controller.resolve_issue(
                args.issue_id, reviewer=args.reviewer, resolution=args.resolution
            )
            print(args.issue_id)
            return 0
        if args.command == "remediate":
            _require_actor(controller.project_manager_for_task(args.task_id))
            loop_id = controller.open_remediation_loop(
                args.task_id,
                owner_team=args.owner_team,
                objective=args.objective,
                issue_id=args.issue_id,
                agents=[_object(value, label="--agent-json") for value in args.agent_json],
            )
            print(loop_id)
            return 0
        if args.command == "verify":
            _require_actor(args.project_manager)
            decision = controller.verify(args.task_id, project_manager=args.project_manager)
            print(
                json.dumps(
                    {
                        "allowed": decision.allowed,
                        "ready_for_founder_approval": decision.ready_for_founder_approval,
                        "missing_requirements": list(decision.missing_requirements),
                        "open_issue_ids": list(decision.open_issue_ids),
                    },
                    sort_keys=True,
                )
            )
            return 0 if decision.ready_for_founder_approval or decision.allowed else 2
        if args.command == "request-founder-approval":
            _require_actor(args.project_manager)
            requested = controller.request_founder_approval(
                args.task_id, project_manager=args.project_manager
            )
            print(json.dumps({"requested": requested}, sort_keys=True))
            return 0 if requested else 2
        if args.command == "founder-approve":
            if not args.confirm_founder_approval:
                print(
                    "ERROR: founder approval requires --confirm-founder-approval",
                    file=sys.stderr,
                )
                return 2
            _require_founder(args.founder)
            controller.approve_founder(args.task_id, founder=args.founder)
            print(args.task_id)
            return 0
        if args.command == "status":
            snapshot = controller.snapshot(args.task_id)
            print(json.dumps(snapshot, sort_keys=True))
            decision = controller.can_complete(args.task_id)
            return 0 if decision.allowed else 2
        if args.command == "complete":
            _require_actor(args.reviewer)
            completed = controller.complete(args.task_id, reviewer=args.reviewer)
            if completed:
                print(args.task_id)
                return 0
            decision = controller.can_complete(args.task_id)
            print(
                json.dumps(
                    {
                        "blocked": True,
                        "missing_requirements": list(decision.missing_requirements),
                        "open_issue_ids": list(decision.open_issue_ids),
                    },
                    sort_keys=True,
                ),
                file=sys.stderr,
            )
            return 2
    except (OSError, RuntimeError, ValueError, argparse.ArgumentTypeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    raise AssertionError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
