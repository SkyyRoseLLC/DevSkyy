"""Regression coverage for the paid-render release controller.

The controller exists to stop the exact failure that wastes render credits:
one SKU reveals a source or fidelity problem, while the batch keeps paying for
later SKUs anyway. It is deterministic and local; it never calls a provider.
"""

from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path

from scripts.oai_render.pipeline import RenderResult, SkuPlan
from scripts.oai_render.references import ReferenceImage
from scripts.oai_render.release_controller import ReleaseController, RenderMisreadLogger


def _plan(sku: str, *, style: str = "on-model", view: str = "front") -> SkuPlan:
    return SkuPlan(
        sku=sku,
        name=sku,
        collection="black-rose",
        output_slug=sku,
        style=style,
        view=view,
        prompt=f"locked source prompt for {sku}",
        references=[
            ReferenceImage(label="product truth", path=Path(f"/{sku}.png"), kind="garment")
        ],
    )


def _record_external_visual_review(
    logger: RenderMisreadLogger, plan: SkuPlan, *, candidate_sha256: str
) -> str:
    """Simulate the separately actor-bound reviewer receipt, not adapter work."""
    task_id = logger.begin_task(plan)
    logger.task_control.record_evidence(
        task_id,
        "visual_qa",
        {"candidate_sha256": candidate_sha256, "surface": "product_render"},
    )
    logger.task_control.record_independent_review(
        task_id,
        reviewer="product_fidelity_independent_reviewer",
        evidence={
            "candidate_sha256": candidate_sha256,
            "surface": "product_render",
            "criteria": {
                "one source-bound candidate only": "PASS",
                "named visual QA evidence passes": "PASS",
                "independent product-fidelity review passes": "PASS",
                "founder approval is distinct from manager and reviewer": "PASS",
            },
        },
    )
    return task_id


def test_batch_is_reduced_to_one_pilot_before_any_paid_call(tmp_path: Path) -> None:
    controller = ReleaseController(RenderMisreadLogger(tmp_path / "misreads.jsonl"))

    decision = controller.authorize([_plan("br-008"), _plan("br-009")])

    assert decision.allowed_plans == (0,)
    assert decision.blocked_plans == (1,)
    assert decision.reason == "PILOT_REQUIRED"


def test_quarantined_pilot_opens_incident_and_blocks_future_scope(tmp_path: Path) -> None:
    logger = RenderMisreadLogger(tmp_path / "misreads.jsonl")
    controller = ReleaseController(logger)
    plan = _plan("br-008")

    controller.record_result(
        plan,
        RenderResult(sku="br-008", status="needs_review", reason="patch scale inconsistent"),
        provider_request_id="req_123",
        estimated_spend_usd=0.40,
        failure_tags=("manual_review_required", "patch_scale"),
    )
    decision = controller.authorize([plan])

    assert decision.allowed_plans == ()
    assert decision.blocked_plans == (0,)
    assert decision.reason == "UNRESOLVED_MISREAD"
    incidents = logger.unresolved_incidents(plan)
    assert len(incidents) == 1
    assert incidents[0]["provider_request_id"] == "req_123"
    assert incidents[0]["failure_tags"] == ["manual_review_required", "patch_scale"]


def test_open_misread_prevents_pilot_approval_and_does_not_clear_incident(tmp_path: Path) -> None:
    logger = RenderMisreadLogger(tmp_path / "misreads.jsonl")
    controller = ReleaseController(logger)
    plan = _plan("br-008")
    controller.record_result(
        plan,
        RenderResult(sku="br-008", status="qc_failed", reason="rose fill leaked into 0"),
        provider_request_id="req_abc",
        estimated_spend_usd=0.40,
        failure_tags=("branding_drift",),
    )
    import pytest

    with pytest.raises(ValueError, match="issue remains open"):
        controller.request_pilot_founder_approval(
            plan,
        )

    assert controller.authorize([plan]).reason == "UNRESOLVED_MISREAD"


def test_resolved_incident_allows_a_new_single_sku_pilot(tmp_path: Path) -> None:
    logger = RenderMisreadLogger(tmp_path / "misreads.jsonl")
    controller = ReleaseController(logger)
    plan = _plan("br-008")
    incident_id = controller.record_result(
        plan,
        RenderResult(sku="br-008", status="error", reason="source drift"),
        provider_request_id=None,
        estimated_spend_usd=0.0,
        failure_tags=("source_drift",),
    )

    controller.resolve_incident(
        incident_id,
        reviewer="product_fidelity_independent_reviewer",
        resolution="bound corrected source",
    )

    assert controller.authorize([plan]).reason == "SINGLE_SKU_PILOT"


def test_pilot_handoff_requires_separate_review_and_never_accepts_founder_or_reviewer(
    tmp_path: Path,
) -> None:
    """The release adapter cannot manufacture review, founder approval, or completion."""
    import pytest

    logger = RenderMisreadLogger(tmp_path / "misreads.jsonl")
    controller = ReleaseController(logger)
    plan = _plan("br-008")
    candidate_path = tmp_path / "br-008-candidate.webp"
    candidate_path.write_bytes(b"candidate-output")
    candidate_sha256 = sha256(candidate_path.read_bytes()).hexdigest()
    controller.record_result(
        plan,
        RenderResult(
            sku="br-008",
            status="rendered",
            reason=None,
            output_path=candidate_path,
        ),
        provider_request_id="req_approved_candidate",
        estimated_spend_usd=0.40,
    )

    expected_task_id = _record_external_visual_review(
        logger, plan, candidate_sha256=candidate_sha256
    )
    task_id = controller.request_pilot_founder_approval(plan)
    events = logger.task_control.ledger.events()

    assert task_id == expected_task_id
    assert any(event["event"] == "pilot_ready_for_founder_approval" for event in logger._events())
    assert not any(
        event["event"] in {"founder_approved", "task_completed"}
        for event in events
        if event.get("task_id") == task_id
    )
    with pytest.raises(TypeError):
        controller.request_pilot_founder_approval(  # type: ignore[call-arg]
            plan,
            founder="founder",
        )
    with pytest.raises(TypeError):
        controller.request_pilot_founder_approval(  # type: ignore[call-arg]
            plan,
            reviewer="product_fidelity_independent_reviewer",
        )


def test_pilot_handoff_rejects_review_bound_to_another_candidate(tmp_path: Path) -> None:
    """A reviewer receipt for candidate A cannot unlock candidate B's founder handoff."""
    import pytest

    logger = RenderMisreadLogger(tmp_path / "misreads.jsonl")
    controller = ReleaseController(logger)
    plan = _plan("br-008")
    candidate_path = tmp_path / "br-008-candidate.webp"
    candidate_path.write_bytes(b"canonical-candidate")
    canonical_sha256 = sha256(candidate_path.read_bytes()).hexdigest()
    controller.record_result(
        plan,
        RenderResult(sku="br-008", status="rendered", output_path=candidate_path),
        provider_request_id="req_candidate",
        estimated_spend_usd=0.40,
    )
    _record_external_visual_review(logger, plan, candidate_sha256="b" * 64)

    with pytest.raises(ValueError, match="persisted candidate hash"):
        controller.request_pilot_founder_approval(plan)

    assert logger.canonical_candidate_sha256(logger.begin_task(plan)) == canonical_sha256


def test_provider_choke_point_stops_a_multi_sku_request_after_one_pilot(
    tmp_path: Path, monkeypatch
) -> None:
    """The policy must hold where paid provider calls happen, not only in a CLI."""
    from scripts.oai_render import config, pipeline

    class Client:
        calls = 0

        def edit(self, *, prompt: str, image_paths: list[Path]) -> bytes:
            self.calls += 1
            return b"candidate"

    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "oai")
    monkeypatch.setattr(config, "REJECTED_DIR", tmp_path / "oai" / "_rejected")
    monkeypatch.setattr(config, "QC_ENABLED", False)
    client = Client()

    results = pipeline.render_all([_plan("br-008"), _plan("br-009")], client, verify_assets=False)

    assert client.calls == 1
    assert [result.status for result in results] == ["rendered", "blocked"]
    task_events = [
        json.loads(line)
        for line in (tmp_path / "oai" / "_control" / "task-execution.jsonl")
        .read_text()
        .splitlines()
    ]
    assert any(event["event"] == "task_started" for event in task_events)
    assert any(event["event"] == "evidence_recorded" for event in task_events)
