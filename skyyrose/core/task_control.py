"""SkyyRose Task Execution Controller.

This production control service gives an operational task a small lifecycle
that is enforced by its entrypoint:

``start -> evidence / first-seen issue -> resolve -> complete``.

Completion is denied while declared evidence is missing or an issue for the
same scope remains unresolved.  Repeating an already-known issue never erases
the first observation: it creates an ``issue_seen_again`` event pointing to the
original issue instead.  The JSONL ledger contains only task metadata, hashes,
and reviewer references -- never source bytes, customer data, or credentials.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
import uuid
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _canonical(value: Mapping[str, Any]) -> str:
    """Return a stable scope/fingerprint representation safe for JSONL."""
    return json.dumps(dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("\x1f".join(parts).encode("utf-8")).hexdigest()[:20]
    return f"{prefix}_{digest}"


@dataclass(frozen=True)
class CompletionDecision:
    allowed: bool
    missing_requirements: tuple[str, ...] = ()
    open_issue_ids: tuple[str, ...] = ()
    ready_for_founder_approval: bool = False
    founder_approved: bool = False


@dataclass(frozen=True)
class TaskAgentAssignment:
    """Named task participant, its job title, and bounded responsibilities."""

    name: str
    job_title: str
    capabilities: tuple[str, ...]

    def record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "job_title": self.job_title,
            "capabilities": list(self.capabilities),
        }


class LedgerIntegrityError(RuntimeError):
    """The append-only task ledger is unreadable or internally inconsistent."""


_SENSITIVE_METADATA_KEY = re.compile(
    r"(?:^|[_-])(api[_-]?key|access[_-]?key|private[_-]?key|password|secret|"
    r"authorization|cookie|credential|bearer|token)(?:$|[_-])",
    re.IGNORECASE,
)
_SENSITIVE_METADATA_VALUE = re.compile(
    r"(?:\bbearer\s+[a-z0-9._~+/=-]{8,}|\b(?:sk|rk|pk|ghp|xoxb|xoxa|xoxp)[_-][a-z0-9_-]{8,}|"
    r"-----begin(?: [a-z0-9_-]+)? private key-----|\b(?:api[_ -]?key|authorization|"
    r"password|secret|token)\s*[:=]\s*\S+)",
    re.IGNORECASE,
)
_MAX_METADATA_DEPTH = 8
_MAX_METADATA_ITEMS = 200
_MAX_METADATA_STRING_LENGTH = 4096
_MAX_LEDGER_RECORD_BYTES = 64 * 1024


def _validate_metadata(value: Any, *, path: str = "record", depth: int = 0) -> None:
    """Reject unbounded or sensitive metadata before it reaches the ledger.

    The task ledger is an evidence index, not a transport for provider payloads,
    source bytes, customer data, or credentials.  Validation at the single
    append boundary covers both the CLI and the MCP surface.
    """
    if depth > _MAX_METADATA_DEPTH:
        raise ValueError(f"task metadata exceeds maximum nesting at {path}")
    if isinstance(value, str):
        if len(value) > _MAX_METADATA_STRING_LENGTH:
            raise ValueError(f"task metadata string is too long at {path}")
        if _SENSITIVE_METADATA_VALUE.search(value):
            raise ValueError(f"task metadata may not contain credential-like value at {path}")
        return
    if value is None or isinstance(value, (bool, int, float)):
        return
    if isinstance(value, Mapping):
        if len(value) > _MAX_METADATA_ITEMS:
            raise ValueError(f"task metadata has too many keys at {path}")
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"task metadata key must be a string at {path}")
            if _SENSITIVE_METADATA_KEY.search(key):
                raise ValueError(f"task metadata may not contain sensitive key: {key}")
            _validate_metadata(item, path=f"{path}.{key}", depth=depth + 1)
        return
    if isinstance(value, (list, tuple)):
        if len(value) > _MAX_METADATA_ITEMS:
            raise ValueError(f"task metadata has too many items at {path}")
        for index, item in enumerate(value):
            _validate_metadata(item, path=f"{path}[{index}]", depth=depth + 1)
        return
    raise ValueError(f"task metadata type is not JSON-safe at {path}")


def _assignments(
    project_manager: str, agents: Sequence[Mapping[str, Any]]
) -> tuple[TaskAgentAssignment, ...]:
    """Validate a task roster and always retain the issued manager role."""
    manager = project_manager.strip()
    assignments: list[TaskAgentAssignment] = [
        TaskAgentAssignment(
            name=manager,
            job_title="Project Manager",
            capabilities=(
                "scope ownership",
                "dependency coordination",
                "evidence verification",
                "remediation-loop routing",
                "founder-approval handoff",
            ),
        )
    ]
    seen = {manager}
    for agent in agents:
        name = str(agent.get("name", "")).strip()
        job_title = str(agent.get("job_title", "")).strip()
        raw_capabilities = agent.get("capabilities", [])
        if not isinstance(raw_capabilities, (list, tuple)):
            raise ValueError("agent capabilities must be a list")
        capabilities = tuple(str(item).strip() for item in raw_capabilities if str(item).strip())
        if not name or not job_title or not capabilities:
            raise ValueError("each agent requires name, job_title, and capabilities")
        if name in seen:
            raise ValueError(f"duplicate task agent: {name}")
        assignments.append(
            TaskAgentAssignment(name=name, job_title=job_title, capabilities=capabilities)
        )
        seen.add(name)
    return tuple(assignments)


class TaskLedger:
    """Append-only JSONL store with a small cross-process append lock."""

    schema = "skyyrose-task-control.v1"

    def __init__(self, path: Path) -> None:
        self.path = Path(path)

    @property
    def lock_path(self) -> Path:
        """A transition lock separate from the JSONL data stream."""
        return self.path.with_name(f"{self.path.name}.lock")

    @contextmanager
    def transition(self):
        """Serialize read-validate-append lifecycle transitions across processes."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
                pass
            try:
                yield
            finally:
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
                    pass

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        try:
            with self.path.open("r", encoding="utf-8") as handle:
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_SH)
                except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
                    pass
                try:
                    lines = handle.read().splitlines()
                finally:
                    try:
                        import fcntl

                        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                    except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
                        pass
        except OSError as exc:
            raise LedgerIntegrityError(f"unable to read task ledger: {exc}") from exc
        records: list[dict[str, Any]] = []
        for line_number, line in enumerate(lines, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise LedgerIntegrityError(
                    f"malformed task ledger record at line {line_number}"
                ) from exc
            if not isinstance(value, dict):
                raise LedgerIntegrityError(f"non-object task ledger record at line {line_number}")
            if value.get("schema") != self.schema:
                continue
            if not all(key in value for key in ("event", "event_id", "ts")):
                raise LedgerIntegrityError(f"incomplete task ledger record at line {line_number}")
            records.append(value)
        return records

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        """Append one event atomically enough for local multi-process tools."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema": self.schema,
            "event": event,
            "event_id": str(uuid.uuid4()),
            "ts": round(time.time(), 3),
            **fields,
        }
        _validate_metadata(record)
        serialized = json.dumps(record, sort_keys=True, ensure_ascii=False)
        if len(serialized.encode("utf-8")) > _MAX_LEDGER_RECORD_BYTES:
            raise ValueError("task ledger record exceeds maximum size")
        # macOS/Linux local worktrees support fcntl.  The fallback still writes
        # safely for the single-process case used by portable test runners.
        with self.path.open("a", encoding="utf-8") as handle:
            try:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
                pass
            try:
                handle.write(serialized + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            finally:
                try:
                    import fcntl

                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except (ImportError, OSError):  # pragma: no cover - non-POSIX fallback
                    pass
        return record


class TaskExecutionController:
    """Record and enforce evidence-first completion for a scoped task."""

    def __init__(self, ledger: TaskLedger) -> None:
        self.ledger = ledger

    def begin(
        self,
        *,
        kind: str,
        scope: Mapping[str, Any],
        project_manager: str,
        requirements: tuple[str, ...] | list[str] = (),
        agents: Sequence[Mapping[str, Any]] = (),
        acceptance_criteria: tuple[str, ...] | list[str] = (),
        e2e_strict: bool = False,
    ) -> str:
        if not kind.strip():
            raise ValueError("task kind is required")
        if not scope:
            raise ValueError("task scope is required")
        if not project_manager.strip():
            raise ValueError("project manager is required")
        requirement_names = tuple(
            dict.fromkeys(item.strip() for item in requirements if item.strip())
        )
        assignments = _assignments(project_manager, agents)
        criteria = tuple(
            dict.fromkeys(item.strip() for item in acceptance_criteria if item.strip())
        )
        if e2e_strict:
            if len(scope) < 2:
                raise ValueError("strict E2E task scope requires at least two identifying fields")
            if not requirement_names:
                raise ValueError("strict E2E task requires declared evidence")
            if "independent_review" not in requirement_names:
                raise ValueError("strict E2E task requires independent_review evidence")
            if not criteria:
                raise ValueError("strict E2E task requires acceptance criteria")
            if not any(
                assignment.name != project_manager.strip()
                and (
                    "review" in assignment.job_title.lower()
                    or "qa" in assignment.job_title.lower()
                    or any(
                        "review" in capability.lower() or "qa" in capability.lower()
                        for capability in assignment.capabilities
                    )
                )
                for assignment in assignments
            ):
                raise ValueError("strict E2E task requires a named independent reviewer")
        scope_key = _canonical(scope)
        # A UUID makes independent attempts auditable.  Scope-level issue checks
        # still join subsequent attempts to the first recorded problem.
        task_id = f"task_{uuid.uuid4()}"
        with self.ledger.transition():
            self.ledger.append(
                "task_started",
                task_id=task_id,
                kind=kind.strip(),
                project_manager=project_manager.strip(),
                agents=[assignment.record() for assignment in assignments],
                scope=dict(scope),
                scope_key=scope_key,
                requirements=list(requirement_names),
                acceptance_criteria=list(criteria),
                e2e_strict=e2e_strict,
            )
        return task_id

    def begin_e2e(
        self,
        *,
        kind: str,
        scope: Mapping[str, Any],
        project_manager: str,
        requirements: tuple[str, ...] | list[str],
        acceptance_criteria: tuple[str, ...] | list[str],
        agents: Sequence[Mapping[str, Any]],
    ) -> str:
        """Start a strict E2E task; no acceptance or reviewer shortcut exists."""
        return self.begin(
            kind=kind,
            scope=scope,
            project_manager=project_manager,
            requirements=requirements,
            agents=agents,
            acceptance_criteria=acceptance_criteria,
            e2e_strict=True,
        )

    def _task_started(self, task_id: str) -> dict[str, Any]:
        for event in reversed(self.ledger.events()):
            if event.get("event") == "task_started" and event.get("task_id") == task_id:
                return event
        raise ValueError(f"unknown task id: {task_id}")

    def _open_issues(self, scope_key: str) -> list[dict[str, Any]]:
        events = self.ledger.events()
        resolved = {
            event.get("issue_id") for event in events if event.get("event") == "issue_resolved"
        }
        return [
            event
            for event in events
            if event.get("event") == "issue_opened"
            and event.get("scope_key") == scope_key
            and event.get("issue_id") not in resolved
        ]

    def is_scope_blocked(self, task_id: str) -> bool:
        started = self._task_started(task_id)
        return bool(self._open_issues(str(started["scope_key"])))

    def is_task_participant(self, task_id: str, actor: str) -> bool:
        """Return whether an issued task roster contains the named actor."""
        started = self._task_started(task_id)
        return any(
            isinstance(assignment, Mapping) and assignment.get("name") == actor.strip()
            for assignment in started.get("agents", [])
        )

    def project_manager_for_task(self, task_id: str) -> str:
        """Return the project manager issued in the immutable task contract."""
        return str(self._task_started(task_id)["project_manager"])

    @staticmethod
    def _is_named_independent_reviewer(started: Mapping[str, Any], reviewer: str) -> bool:
        """Return whether ``reviewer`` is an issued, review-qualified task role."""
        candidate = reviewer.strip()
        if not candidate or candidate == started["project_manager"]:
            return False
        for assignment in started.get("agents", []):
            if not isinstance(assignment, Mapping) or assignment.get("name") != candidate:
                continue
            job_title = str(assignment.get("job_title", "")).lower()
            capabilities = assignment.get("capabilities", [])
            return (
                "review" in job_title
                or "qa" in job_title
                or any(
                    "review" in str(capability).lower() or "qa" in str(capability).lower()
                    for capability in capabilities
                )
            )
        return False

    def _valid_independent_reviewers(self, task_id: str, started: Mapping[str, Any]) -> set[str]:
        """Return reviewers backed by the dedicated, passing review event."""
        reviewers: set[str] = set()
        for event in self.ledger.events():
            if (
                event.get("event") != "evidence_recorded"
                or event.get("task_id") != task_id
                or event.get("name") != "independent_review"
            ):
                continue
            details = event.get("details")
            if not isinstance(details, Mapping) or details.get("outcome") != "PASS":
                continue
            if started.get("e2e_strict"):
                criteria = details.get("criteria")
                if not isinstance(criteria, Mapping) or any(
                    criteria.get(str(criterion)) != "PASS"
                    for criterion in started.get("acceptance_criteria", [])
                ):
                    continue
            reviewer = str(details.get("reviewer", "")).strip()
            if self._is_named_independent_reviewer(started, reviewer):
                reviewers.add(reviewer)
        return reviewers

    def record_evidence(
        self,
        task_id: str,
        name: str,
        details: Mapping[str, Any],
        *,
        recorded_by: str | None = None,
    ) -> str:
        with self.ledger.transition():
            started = self._task_started(task_id)
            if not name.strip():
                raise ValueError("evidence name is required")
            if recorded_by is not None and not self.is_task_participant(task_id, recorded_by):
                raise ValueError("evidence recorder must be a named task participant")
            if started.get("e2e_strict") and name.strip() == "independent_review":
                raise ValueError(
                    "strict E2E independent_review evidence must use record_independent_review"
                )
            record = self.ledger.append(
                "evidence_recorded",
                task_id=task_id,
                scope_key=started["scope_key"],
                name=name.strip(),
                details=dict(details),
                recorded_by=recorded_by.strip() if recorded_by else None,
            )
            return str(record["event_id"])

    def record_independent_review(
        self, task_id: str, *, reviewer: str, evidence: Mapping[str, Any]
    ) -> str:
        """Attach a passing independent review required by strict E2E tasks."""
        with self.ledger.transition():
            started = self._task_started(task_id)
            if not reviewer.strip() or reviewer.strip() == started["project_manager"]:
                raise ValueError("independent reviewer must differ from the project manager")
            if started.get("e2e_strict") and not self._is_named_independent_reviewer(
                started, reviewer
            ):
                raise ValueError(
                    "strict E2E reviewer must be a named independent reviewer in the roster"
                )
            criteria = evidence.get("criteria")
            required_criteria = tuple(str(item) for item in started.get("acceptance_criteria", []))
            if started.get("e2e_strict"):
                if not isinstance(criteria, Mapping):
                    raise ValueError("strict E2E independent review requires a criteria PASS map")
                failed = [
                    criterion
                    for criterion in required_criteria
                    if criteria.get(criterion) != "PASS"
                ]
                if failed:
                    raise ValueError(
                        "strict E2E independent review must attest PASS for every acceptance criterion"
                    )
            details = {**dict(evidence), "reviewer": reviewer.strip(), "outcome": "PASS"}
            record = self.ledger.append(
                "evidence_recorded",
                task_id=task_id,
                scope_key=started["scope_key"],
                name="independent_review",
                details=details,
                recorded_by=reviewer.strip(),
            )
            return str(record["event_id"])

    def record_issue(
        self,
        task_id: str,
        *,
        category: str,
        summary: str,
        fingerprint: str,
        details: Mapping[str, Any] | None = None,
    ) -> str:
        """Open the first unique issue, or link a recurrence to that record."""
        with self.ledger.transition():
            started = self._task_started(task_id)
            if not category.strip() or not summary.strip() or not fingerprint.strip():
                raise ValueError("issue category, summary, and fingerprint are required")
            scope_key = str(started["scope_key"])
            issue_key = _id("issue", scope_key, category.strip(), fingerprint.strip())
            for event in self.ledger.events():
                if event.get("event") != "issue_opened" or event.get("issue_id") != issue_key:
                    continue
                self.ledger.append(
                    "issue_seen_again",
                    task_id=task_id,
                    scope_key=scope_key,
                    issue_id=issue_key,
                    category=category.strip(),
                    fingerprint=fingerprint.strip(),
                    summary=summary.strip(),
                    details=dict(details or {}),
                )
                return issue_key
            self.ledger.append(
                "issue_opened",
                task_id=task_id,
                scope_key=scope_key,
                issue_id=issue_key,
                category=category.strip(),
                fingerprint=fingerprint.strip(),
                summary=summary.strip(),
                details=dict(details or {}),
            )
            return issue_key

    def resolve_issue(self, issue_id: str, *, reviewer: str, resolution: str) -> None:
        if not reviewer.strip() or not resolution.strip():
            raise ValueError("issue resolution requires reviewer and resolution")
        with self.ledger.transition():
            issue = next(
                (
                    event
                    for event in self.ledger.events()
                    if event.get("event") == "issue_opened" and event.get("issue_id") == issue_id
                ),
                None,
            )
            if issue is None:
                raise ValueError(f"unknown issue id: {issue_id}")
            started = self._task_started(str(issue["task_id"]))
            if started.get("e2e_strict") and not self._is_named_independent_reviewer(
                started, reviewer
            ):
                raise ValueError(
                    "strict E2E issue resolution requires a named independent reviewer"
                )
            self.ledger.append(
                "issue_resolved",
                issue_id=issue_id,
                reviewer=reviewer.strip(),
                resolution=resolution.strip(),
            )

    def open_remediation_loop(
        self,
        task_id: str,
        *,
        owner_team: str,
        objective: str,
        issue_id: str | None = None,
        agents: Sequence[Mapping[str, Any]] = (),
    ) -> str:
        """Assign a bounded correction loop before the manager re-verifies.

        A loop is an auditable handoff to the responsible specialist team. It
        does not resolve an issue automatically; the manager must re-verify
        supplied evidence and a reviewer must record the resolution.
        """
        with self.ledger.transition():
            started = self._task_started(task_id)
            if not owner_team.strip() or not objective.strip():
                raise ValueError("remediation loop requires owner team and objective")
            open_issue_ids = {
                str(event["issue_id"]) for event in self._open_issues(str(started["scope_key"]))
            }
            if issue_id is not None and issue_id not in open_issue_ids:
                raise ValueError("remediation loop issue must be open in the task scope")
            assignments = _assignments(str(started["project_manager"]), agents)
            loop_id = f"loop_{uuid.uuid4()}"
            self.ledger.append(
                "remediation_loop_opened",
                loop_id=loop_id,
                task_id=task_id,
                project_manager=started["project_manager"],
                issue_id=issue_id,
                owner_team=owner_team.strip(),
                objective=objective.strip(),
                agents=[assignment.record() for assignment in assignments],
            )
            return loop_id

    def _verify_locked(self, task_id: str, *, project_manager: str) -> CompletionDecision:
        """Write a manager verification while the lifecycle transition is locked."""
        started = self._task_started(task_id)
        if project_manager.strip() != started["project_manager"]:
            raise ValueError("only the issued project manager may verify this task")
        decision = self.can_complete(task_id)
        if decision.missing_requirements or decision.open_issue_ids:
            state = "BLOCKED"
        elif decision.founder_approved:
            state = "VERIFIED_AFTER_FOUNDER_APPROVAL"
        else:
            state = "READY_FOR_FOUNDER_APPROVAL"
        reviewers = sorted(self._valid_independent_reviewers(task_id, started))
        criteria_attestations = [
            {
                "criterion": criterion,
                "status": "PASS" if reviewers else "BLOCKED",
                "reviewers": reviewers,
            }
            for criterion in started.get("acceptance_criteria", [])
        ]
        self.ledger.append(
            "manager_verification",
            task_id=task_id,
            project_manager=started["project_manager"],
            state=state,
            missing_requirements=list(decision.missing_requirements),
            open_issue_ids=list(decision.open_issue_ids),
            acceptance_criteria=criteria_attestations,
        )
        return decision

    def verify(self, task_id: str, *, project_manager: str) -> CompletionDecision:
        """Record the manager's evidence audit and its next state.

        Verification is deterministic: every declared evidence name must be
        present and no scoped issue may remain open. It does not turn a task
        into a founder-approved release; that remains a separate human action.
        """
        with self.ledger.transition():
            return self._verify_locked(task_id, project_manager=project_manager)

    def request_founder_approval(self, task_id: str, *, project_manager: str) -> bool:
        """Raise a fully evidenced, issue-free task for founder approval."""
        with self.ledger.transition():
            decision = self._verify_locked(task_id, project_manager=project_manager)
            if not decision.ready_for_founder_approval:
                return False
            self.ledger.append(
                "founder_approval_requested",
                task_id=task_id,
                project_manager=project_manager.strip(),
            )
            return True

    def approve_founder(self, task_id: str, *, founder: str) -> None:
        """Record explicit founder approval after a manager verification pass."""
        if not founder.strip():
            raise ValueError("founder approval requires founder identity")
        with self.ledger.transition():
            started = self._task_started(task_id)
            if founder.strip() == started["project_manager"]:
                raise ValueError("project manager cannot record founder approval")
            events = self.ledger.events()
            requested = any(
                event.get("event") == "founder_approval_requested"
                and event.get("task_id") == task_id
                for event in events
            )
            if not requested:
                raise ValueError("founder approval requires a manager approval request")
            decision = self.can_complete(task_id)
            if decision.missing_requirements or decision.open_issue_ids:
                raise ValueError("cannot approve a task with missing evidence or open issues")
            if started.get("e2e_strict"):
                reviewers = self._valid_independent_reviewers(task_id, started)
                if not reviewers:
                    raise ValueError(
                        "strict E2E task requires an independent review before founder approval"
                    )
                if founder.strip() in reviewers:
                    raise ValueError("independent reviewer cannot record founder approval")
            self.ledger.append(
                "founder_approved",
                task_id=task_id,
                founder=founder.strip(),
                project_manager=started["project_manager"],
            )

    def can_complete(self, task_id: str) -> CompletionDecision:
        started = self._task_started(task_id)
        evidence_names = {
            str(event.get("name"))
            for event in self.ledger.events()
            if event.get("event") == "evidence_recorded" and event.get("task_id") == task_id
        }
        required = tuple(str(item) for item in started.get("requirements", []))
        missing_names = set(item for item in required if item not in evidence_names)
        if started.get("e2e_strict") and not self._valid_independent_reviewers(task_id, started):
            missing_names.add("independent_review")
        missing = tuple(item for item in required if item in missing_names)
        open_issue_ids = tuple(
            str(event["issue_id"]) for event in self._open_issues(str(started["scope_key"]))
        )
        founder_approved = any(
            event.get("event") == "founder_approved" and event.get("task_id") == task_id
            for event in self.ledger.events()
        )
        evidence_clear = not missing and not open_issue_ids
        return CompletionDecision(
            allowed=evidence_clear and founder_approved,
            missing_requirements=missing,
            open_issue_ids=open_issue_ids,
            ready_for_founder_approval=evidence_clear and not founder_approved,
            founder_approved=founder_approved,
        )

    def snapshot(self, task_id: str) -> dict[str, Any]:
        """Return the task's present, ledger-derived state for CLI/MCP readers.

        This method deliberately derives status from immutable events rather
        than maintaining mutable state.  It is safe for operator dashboards:
        it reports required evidence, scoped issues, and completion eligibility
        without changing the task or treating a read as a verification pass.
        """
        started = self._task_started(task_id)
        events = self.ledger.events()
        scope_key = str(started["scope_key"])
        scope_events = [event for event in events if event.get("scope_key") == scope_key]
        resolved = {
            str(event["issue_id"])
            for event in events
            if event.get("event") == "issue_resolved" and event.get("issue_id")
        }
        issues: list[dict[str, Any]] = []
        for event in scope_events:
            if event.get("event") != "issue_opened":
                continue
            issue_id = str(event["issue_id"])
            issues.append(
                {
                    "issue_id": issue_id,
                    "category": event.get("category"),
                    "fingerprint": event.get("fingerprint"),
                    "summary": event.get("summary"),
                    "open": issue_id not in resolved,
                    "recurrence_count": sum(
                        1
                        for candidate in scope_events
                        if candidate.get("event") == "issue_seen_again"
                        and candidate.get("issue_id") == issue_id
                    ),
                }
            )
        evidence = [
            {
                "event_id": event["event_id"],
                "name": event.get("name"),
                "details": event.get("details", {}),
            }
            for event in events
            if event.get("event") == "evidence_recorded" and event.get("task_id") == task_id
        ]
        decision = self.can_complete(task_id)
        return {
            "task": {
                "task_id": task_id,
                "kind": started["kind"],
                "scope": started["scope"],
                "project_manager": started["project_manager"],
                "agents": started["agents"],
                "requirements": started["requirements"],
                "acceptance_criteria": started.get("acceptance_criteria", []),
                "e2e_strict": bool(started.get("e2e_strict")),
            },
            "evidence": evidence,
            "issues": issues,
            "completion": {
                "allowed": decision.allowed,
                "missing_requirements": list(decision.missing_requirements),
                "open_issue_ids": list(decision.open_issue_ids),
                "ready_for_founder_approval": decision.ready_for_founder_approval,
                "founder_approved": decision.founder_approved,
            },
        }

    def complete(self, task_id: str, *, reviewer: str) -> bool:
        if not reviewer.strip():
            raise ValueError("completion requires reviewer")
        with self.ledger.transition():
            started = self._task_started(task_id)
            if started.get("e2e_strict") and reviewer.strip() != started["project_manager"]:
                raise ValueError(
                    "strict E2E completion must be recorded by the issued project manager"
                )
            decision = self.can_complete(task_id)
            if not decision.allowed:
                self.ledger.append(
                    "completion_blocked",
                    task_id=task_id,
                    missing_requirements=list(decision.missing_requirements),
                    open_issue_ids=list(decision.open_issue_ids),
                )
                return False
            self.ledger.append(
                "task_completed",
                task_id=task_id,
                reviewer=reviewer.strip(),
                project_manager=started["project_manager"],
            )
            return True


__all__ = [
    "CompletionDecision",
    "LedgerIntegrityError",
    "TaskAgentAssignment",
    "TaskExecutionController",
    "TaskLedger",
]
