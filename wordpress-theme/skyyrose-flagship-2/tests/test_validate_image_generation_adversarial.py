"""Focused tests for the GPT-then-founder image review gate."""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest
from PIL import Image

THEME_DIR = Path(__file__).resolve().parents[1]
SCRIPT = THEME_DIR / "scripts/validate-image-generation-adversarial.py"
SPEC = importlib.util.spec_from_file_location("image_adversarial", SCRIPT)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def build_fixture(tmp_path: Path, score: int = 95) -> dict[str, Path]:
    policy_path = tmp_path / "policy.json"
    write_json(
        policy_path,
        {
            "required_vision_judges": ["gpt-5.5-pro"],
            "synthesis_model": None,
            "minimum_each_vision_score": 95,
            "minimum_final_score": 95,
            "required_hallucination_veto_result": False,
            "all_judges_available": True,
            "unverifiable_required_regions": 0,
            "source_hashes_current": True,
            "founder_approval_required": True,
            "review_authorities": ["gpt-5.5-pro", "founder"],
        },
    )
    output_path = tmp_path / "candidate.png"
    Image.new("RGB", (8, 8), "red").save(output_path)
    output_hash = VALIDATOR.sha256(output_path)
    preflight_path = tmp_path / "preflight.json"
    preflight = {
        "status": "PASS_READY_TO_GENERATE",
        "prompt_contract": {"sha256": "a" * 64},
        "product_sot": {"sha256": "b" * 64},
        "active_generation_jobs": ["job-1"],
        "tournament_policy": {
            "path": "policy.json",
            "sha256": VALIDATOR.sha256(policy_path),
        },
    }
    write_json(preflight_path, preflight)
    batch_path = tmp_path / "batch.json"
    batch = {
        "schema": "skyyrose.image-generation-batch.v1",
        "approval_state": "FOUNDER_REVIEW_REQUIRED",
        "preflight_receipt_sha256": VALIDATOR.sha256(preflight_path),
        "prompt_contract_sha256": "a" * 64,
        "product_sot_sha256": "b" * 64,
        "jobs": {
            "job-1": {
                "output": {"path": "candidate.png", "sha256": output_hash},
            }
        },
    }
    write_json(batch_path, batch)
    reviews = tmp_path / "reviews"
    review = {
        "schema": "skyyrose.adversarial-vision-review.v1",
        "reviewer_id": "gpt-5.5-pro",
        "available": True,
        "generation_batch_manifest_sha256": VALIDATOR.sha256(batch_path),
        "independent_visual_inspection": True,
        "proof_board_visually_compared": True,
        "original_founder_corrections_verbatim": "Exact product and logo fidelity required.",
        "reviewed_outputs": {"job-1": output_hash},
        "scores": {
            "job-1": {
                "source_hashes_current": True,
                "unverifiable_required_regions": 0,
                "overall": score,
                "hallucination_veto": False,
                "overall_verdict": "clean",
                "recommend_ship": True,
                "evidence": ["Product regions visually compared against bound proof."],
            }
        },
    }
    write_json(reviews / "gpt-5.5-pro.json", review)
    return {
        "preflight": preflight_path,
        "batch": batch_path,
        "reviews": reviews,
        "receipt": tmp_path / "receipt.json",
    }


def run_main(
    fixture: dict[str, Path],
    monkeypatch: pytest.MonkeyPatch,
    root: Path,
    founder_override: Path | None = None,
) -> dict:
    monkeypatch.setattr(VALIDATOR, "ROOT", root)
    argv = [
        str(SCRIPT),
        "--preflight",
        str(fixture["preflight"]),
        "--batch",
        str(fixture["batch"]),
        "--reviews",
        str(fixture["reviews"]),
        "--receipt",
        str(fixture["receipt"]),
        "--write-receipt",
    ]
    if founder_override is not None:
        argv.extend(["--founder-override", str(founder_override)])
    monkeypatch.setattr(sys, "argv", argv)
    assert VALIDATOR.main() == 0
    return json.loads(fixture["receipt"].read_text())


def test_gpt_pass_stays_blocked_until_explicit_founder_approval(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    receipt = run_main(build_fixture(tmp_path), monkeypatch, tmp_path)

    assert [item["reviewer_id"] for item in receipt["reviewers"]] == ["gpt-5.5-pro"]
    assert receipt["founder_approval_required"] is True
    assert receipt["founder_approval_recorded"] is False
    assert receipt["status"] == "PASS_GPT_VISION_REVIEW"
    assert receipt["downstream_state"] == "BLOCKED_PENDING_EXPLICIT_FOUNDER_APPROVAL"


def test_gemini_or_opus_review_file_is_rejected(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_fixture(tmp_path)
    write_json(
        fixture["reviews"] / "opus.json",
        {"schema": "skyyrose.adversarial-synthesis-review.v1"},
    )
    monkeypatch.setattr(VALIDATOR, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--preflight",
            str(fixture["preflight"]),
            "--batch",
            str(fixture["batch"]),
            "--reviews",
            str(fixture["reviews"]),
        ],
    )

    with pytest.raises(SystemExit, match="exactly one GPT vision report"):
        VALIDATOR.main()


def test_gpt_score_below_95_fails_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    fixture = build_fixture(tmp_path, score=94)
    monkeypatch.setattr(VALIDATOR, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--preflight",
            str(fixture["preflight"]),
            "--batch",
            str(fixture["batch"]),
            "--reviews",
            str(fixture["reviews"]),
        ],
    )

    with pytest.raises(SystemExit, match="explicit founder override is required"):
        VALIDATOR.main()


def test_gpt_hallucination_veto_fails_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_fixture(tmp_path)
    review_path = fixture["reviews"] / "gpt-5.5-pro.json"
    review = json.loads(review_path.read_text())
    review["scores"]["job-1"]["hallucination_veto"] = True
    write_json(review_path, review)
    monkeypatch.setattr(VALIDATOR, "ROOT", tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--preflight",
            str(fixture["preflight"]),
            "--batch",
            str(fixture["batch"]),
            "--reviews",
            str(fixture["reviews"]),
        ],
    )

    with pytest.raises(SystemExit, match="explicit founder override is required"):
        VALIDATOR.main()


def founder_override_for(fixture: dict[str, Path], score: int) -> Path:
    review_path = fixture["reviews"] / "gpt-5.5-pro.json"
    review = json.loads(review_path.read_text())
    output_hash = review["reviewed_outputs"]["job-1"]
    path = fixture["receipt"].parent / "founder-override.json"
    write_json(
        path,
        {
            "schema": "skyyrose.founder-image-review-override.v1",
            "founder_override": True,
            "founder_approved": True,
            "deployment_authorized": False,
            "approved_at": datetime.now(UTC).isoformat(),
            "founder_instruction_verbatim": "Use the selected image despite the GPT rejection.",
            "generation_batch_manifest_sha256": VALIDATOR.sha256(fixture["batch"]),
            "gpt_review_sha256": VALIDATOR.sha256(review_path),
            "selected_outputs": {"job-1": output_hash},
            "machine_assessments": {
                "job-1": {
                    "overall": score,
                    "minimum_required": 95,
                    "hallucination_veto": False,
                    "overall_verdict": "clean",
                    "recommend_ship": True,
                    "passed_gpt_gate": False,
                }
            },
        },
    )
    return path


def test_explicit_founder_override_preserves_gpt_rejection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_fixture(tmp_path, score=90)
    override = founder_override_for(fixture, score=90)

    receipt = run_main(fixture, monkeypatch, tmp_path, override)

    assert receipt["status"] == "FOUNDER_OVERRIDE_RECORDED_GPT_REJECTION_PRESERVED"
    assert receipt["machine_rejected_outputs"]["job-1"]["overall"] == 90
    assert receipt["machine_rejected_outputs"]["job-1"]["passed_gpt_gate"] is False
    assert receipt["founder_approval_recorded"] is True
    assert receipt["approved_output_hashes"] == receipt["reviewed_outputs"]
    assert (
        receipt["downstream_state"]
        == "COMPOSITOR_ALLOWED_FOR_EXPLICIT_FOUNDER_OVERRIDE_SELECTION_ONLY"
    )


def test_founder_override_cannot_rewrite_gpt_score(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = build_fixture(tmp_path, score=90)
    override = founder_override_for(fixture, score=90)
    value = json.loads(override.read_text())
    value["machine_assessments"]["job-1"]["overall"] = 99
    write_json(override, value)

    with pytest.raises(SystemExit, match="rewrites or omits the GPT assessment"):
        run_main(fixture, monkeypatch, tmp_path, override)
