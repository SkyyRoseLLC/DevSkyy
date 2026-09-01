"""Deterministic release control for paid product-render batches.

This module is deliberately not a vision judge. It enforces the business rule
that the renderer alone cannot prove: when a product candidate reveals a
misread, stop spending and retain the exact evidence needed to repair it.

Every decision is appended to a local JSONL ledger. The ledger never contains
credentials, prompts, or image bytes; it stores stable scope/source hashes,
provider request IDs, failure tags, and estimated spend only.
"""

from __future__ import annotations

import hashlib
import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from skyyrose.core.task_control import TaskExecutionController, TaskLedger

if TYPE_CHECKING:
    from .pipeline import RenderResult, SkuPlan


INCIDENT_STATUSES = frozenset({"needs_review", "qc_failed", "error"})


@dataclass(frozen=True)
class ReleaseDecision:
    """The subset of an ordered plan list that may make a paid provider call."""

    allowed_plans: tuple[int, ...]
    blocked_plans: tuple[int, ...]
    reason: str


def _scope(plan: SkuPlan) -> dict[str, str]:
    return {"sku": plan.sku, "style": plan.style, "view": plan.view}


def _fingerprint(plan: SkuPlan) -> str:
    """Hash the render-relevant plan shape without storing prompt/source bytes."""
    references: list[dict[str, str | None]] = []
    for reference in plan.references:
        digest: str | None = None
        try:
            digest = hashlib.sha256(reference.path.read_bytes()).hexdigest()
        except OSError:
            # Planning tests and a missing source are still loggable. The paid
            # source-integrity gate remains responsible for blocking that path.
            digest = None
        references.append({"path": str(reference.path), "kind": reference.kind, "sha256": digest})
    payload = {
        "scope": _scope(plan),
        "prompt_sha256": hashlib.sha256(plan.prompt.encode("utf-8")).hexdigest(),
        "references": references,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _file_sha256(path: Path) -> str:
    """Hash the exact persisted candidate without loading an image into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RenderMisreadLogger:
    """Render adapter for the shared task ledger plus render-specific incidents.

    The companion task ledger gives every controlled render a governed lifecycle
    (started, evidence, issue, resolution, completion).  This focused JSONL
    keeps the release-controller queries inexpensive and backwards-compatible.
    """

    def __init__(self, path: Path) -> None:
        self.path = path
        self.task_control = TaskExecutionController(
            TaskLedger(path.parent / "task-execution.jsonl")
        )
        self._task_ids: dict[tuple[str, str, str], str] = {}

    def _task_id(self, plan: SkuPlan) -> str:
        scope = {**_scope(plan), "target": "review-only"}
        key = (scope["sku"], scope["style"], scope["view"])
        if key not in self._task_ids:
            task_id = self.task_control.begin_e2e(
                kind="product_render",
                scope=scope,
                project_manager="fashion_theme_team_project_manager",
                requirements=(
                    "source_fingerprint",
                    "candidate",
                    "visual_qa",
                    "independent_review",
                ),
                acceptance_criteria=(
                    "one source-bound candidate only",
                    "named visual QA evidence passes",
                    "independent product-fidelity review passes",
                    "founder approval is distinct from manager and reviewer",
                ),
                agents=(
                    {
                        "name": "oai_render_release_controller",
                        "job_title": "Render Release Controller",
                        "capabilities": [
                            "one-SKU pilot enforcement",
                            "paid-spend circuit breaking",
                            "first-seen incident logging",
                        ],
                    },
                    {
                        "name": "product_fidelity_independent_reviewer",
                        "job_title": "Product Fidelity Independent Reviewer",
                        "capabilities": [
                            "SKU construction verification",
                            "patch, lettering, and number-placement review",
                            "candidate-bound visual evidence review",
                        ],
                    },
                ),
            )
            self.task_control.record_evidence(
                task_id,
                "source_fingerprint",
                {"sha256": _fingerprint(plan)},
            )
            self._task_ids[key] = task_id
        return self._task_ids[key]

    def _events(self) -> list[dict[str, Any]]:
        try:
            lines = self.path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return []
        events: list[dict[str, Any]] = []
        for line in lines:
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(record, dict):
                events.append(record)
        return events

    def canonical_candidate_sha256(self, task_id: str) -> str:
        """Return the single immutable candidate hash eligible for handoff."""
        candidates = [
            event
            for event in self.task_control.ledger.events()
            if event.get("event") == "evidence_recorded"
            and event.get("task_id") == task_id
            and event.get("name") == "candidate"
        ]
        if len(candidates) != 1:
            raise ValueError("pilot handoff requires exactly one candidate evidence receipt")
        details = candidates[0].get("details")
        candidate_sha256 = details.get("candidate_sha256") if isinstance(details, dict) else None
        if not isinstance(candidate_sha256, str) or len(candidate_sha256) != 64:
            raise ValueError("pilot handoff requires a persisted 64-character candidate hash")
        return candidate_sha256

    def candidate_review_is_bound(self, task_id: str, candidate_sha256: str) -> bool:
        """Require both QA receipts to identify the persisted candidate exactly."""
        names = {"visual_qa", "independent_review"}
        bound_names = {
            str(event.get("name"))
            for event in self.task_control.ledger.events()
            if event.get("event") == "evidence_recorded"
            and event.get("task_id") == task_id
            and event.get("name") in names
            and isinstance(event.get("details"), dict)
            and event["details"].get("candidate_sha256") == candidate_sha256
        }
        return bound_names == names

    def append(self, event: str, **fields: Any) -> dict[str, Any]:
        record = {
            "schema": "skyyrose-render-misread-ledger.v1",
            "event": event,
            "event_id": str(uuid.uuid4()),
            "ts": round(time.time(), 3),
            **fields,
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, ensure_ascii=False) + "\n")
        return record

    def unresolved_incidents(self, plan: SkuPlan) -> list[dict[str, Any]]:
        scope = _scope(plan)
        resolved = {
            event.get("incident_id")
            for event in self._events()
            if event.get("event") == "incident_resolved"
        }
        return [
            event
            for event in self._events()
            if event.get("event") == "incident_opened"
            and event.get("scope") == scope
            and event.get("incident_id", event.get("event_id")) not in resolved
        ]

    def begin_task(self, plan: SkuPlan) -> str:
        """Log the render work before a provider call can happen."""
        return self._task_id(plan)

    def pending_pilot(self, plan: SkuPlan) -> bool:
        fingerprint = _fingerprint(plan)
        events = self._events()
        approved = {
            (event.get("scope_key"), event.get("source_fingerprint"))
            for event in events
            if event.get("event") == "pilot_approved"
        }
        scope_key = json.dumps(_scope(plan), sort_keys=True)
        return any(
            event.get("event") == "pilot_review_required"
            and event.get("scope_key") == scope_key
            and event.get("source_fingerprint") == fingerprint
            and (scope_key, fingerprint) not in approved
            for event in events
        )


class ReleaseController:
    """Permit one visual pilot, then halt on review debt or an incident.

    A multi-SKU CLI request is an ordering convenience, never permission to
    buy a whole batch before the first candidate has been evaluated. The first
    paid output creates explicit pilot-review debt. A rejected, unjudged, or
    provider-failed output becomes an unresolved incident that blocks the same
    SKU/style/view until a reviewer records a resolution.
    """

    def __init__(self, logger: RenderMisreadLogger) -> None:
        self.logger = logger

    def authorize(self, plans: list[SkuPlan]) -> ReleaseDecision:
        # A plan is a task the moment it is considered for a paid action.  This
        # makes a blocked request auditable instead of silently disappearing.
        for plan in plans:
            self.logger.begin_task(plan)
        if not plans:
            return ReleaseDecision((), (), "NO_PLANS")
        if any(self.logger.unresolved_incidents(plan) for plan in plans):
            return ReleaseDecision((), tuple(range(len(plans))), "UNRESOLVED_MISREAD")
        if any(self.logger.pending_pilot(plan) for plan in plans):
            return ReleaseDecision((), tuple(range(len(plans))), "PILOT_REVIEW_REQUIRED")
        if len(plans) == 1:
            return ReleaseDecision((0,), (), "SINGLE_SKU_PILOT")
        return ReleaseDecision((0,), tuple(range(1, len(plans))), "PILOT_REQUIRED")

    def record_result(
        self,
        plan: SkuPlan,
        result: RenderResult,
        *,
        provider_request_id: str | None,
        estimated_spend_usd: float,
        failure_tags: tuple[str, ...] = (),
    ) -> str:
        scope = _scope(plan)
        task_id = self.logger.begin_task(plan)
        candidate_sha256: str | None = None
        if result.output_path is not None:
            try:
                candidate_sha256 = _file_sha256(result.output_path)
            except OSError as exc:
                if result.status == "rendered":
                    raise ValueError(
                        "rendered result must retain a readable candidate output"
                    ) from exc
        if result.status == "rendered" and candidate_sha256 is None:
            raise ValueError("rendered result must retain a candidate output path")
        self.logger.task_control.record_evidence(
            task_id,
            "candidate",
            {
                "status": result.status,
                "candidate_sha256": candidate_sha256,
                "provider_request_id": provider_request_id,
                "estimated_spend_usd": round(estimated_spend_usd, 6),
            },
        )
        common = {
            "task_id": task_id,
            "scope": scope,
            "scope_key": json.dumps(scope, sort_keys=True),
            "source_fingerprint": _fingerprint(plan),
            "result_status": result.status,
            "reason": result.reason,
            "provider_request_id": provider_request_id,
            "estimated_spend_usd": round(estimated_spend_usd, 6),
        }
        if result.status in INCIDENT_STATUSES:
            incident_id = self.logger.task_control.record_issue(
                task_id,
                category="render_misread",
                summary=result.reason or result.status,
                fingerprint=hashlib.sha256(
                    json.dumps(
                        {
                            "status": result.status,
                            "failure_tags": sorted(set(failure_tags)),
                            "reason": result.reason,
                        },
                        sort_keys=True,
                    ).encode("utf-8")
                ).hexdigest(),
                details={
                    "provider_request_id": provider_request_id,
                    "failure_tags": sorted(set(failure_tags)),
                    "estimated_spend_usd": round(estimated_spend_usd, 6),
                },
            )
            record = self.logger.append(
                "incident_opened",
                incident_id=incident_id,
                failure_tags=sorted(set(failure_tags)),
                **common,
            )
        else:
            record = self.logger.append("pilot_review_required", **common)
        # The caller must receive the durable issue identifier so a later
        # resolution closes both the task ledger and this adapter.
        return incident_id if result.status in INCIDENT_STATUSES else str(record["event_id"])

    def resolve_incident(self, incident_id: str, *, reviewer: str, resolution: str) -> None:
        if not reviewer.strip() or not resolution.strip():
            raise ValueError("incident resolution requires reviewer and resolution")
        self.logger.task_control.resolve_issue(
            incident_id, reviewer=reviewer.strip(), resolution=resolution.strip()
        )
        self.logger.append(
            "incident_resolved",
            incident_id=incident_id,
            reviewer=reviewer.strip(),
            resolution=resolution.strip(),
        )

    def request_pilot_founder_approval(self, plan: SkuPlan) -> str:
        """Request founder approval only after a separate actor records review.

        This adapter owns paid-render control and candidate evidence. It is not
        an identity provider, so it cannot create independent-review evidence
        from a caller-supplied label. The rostered reviewer must record that
        receipt through the actor-bound CLI or MCP controller before this
        adapter can request a founder-approval handoff.
        """
        scope = _scope(plan)
        task_id = self.logger.begin_task(plan)
        if self.logger.unresolved_incidents(plan):
            raise ValueError("pilot cannot be approved while an issue remains open")
        candidate_sha256 = self.logger.canonical_candidate_sha256(task_id)
        if not self.logger.candidate_review_is_bound(task_id, candidate_sha256):
            raise ValueError(
                "pilot cannot advance until visual QA and independent review reference the "
                "persisted candidate hash"
            )
        if not self.logger.task_control.request_founder_approval(
            task_id, project_manager="fashion_theme_team_project_manager"
        ):
            raise ValueError(
                "pilot cannot advance until separately recorded visual QA, independent review, "
                "and all other required evidence are complete"
            )
        self.logger.append(
            "pilot_ready_for_founder_approval",
            task_id=task_id,
            scope=scope,
            candidate_sha256=candidate_sha256,
        )
        return task_id


__all__ = ["ReleaseController", "ReleaseDecision", "RenderMisreadLogger"]
