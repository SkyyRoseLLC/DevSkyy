"""Focused fail-closed tests for the LH native-scene planning package."""

from __future__ import annotations

import copy
import importlib.util
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

THEME_DIR = Path(__file__).resolve().parents[1]
SCRIPT = THEME_DIR / "scripts/validate-native-scene-regeneration-preflight.py"
CONTRACT = (
    THEME_DIR / "assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/"
    "preflight-v1/vision-authored-prompts/"
    "lh-commerce-1-native-scene-regeneration-plan-v1.json"
)

SPEC = importlib.util.spec_from_file_location("native_scene_preflight", SCRIPT)
assert SPEC and SPEC.loader
VALIDATOR = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(VALIDATOR)


def contract_copy() -> dict:
    return copy.deepcopy(VALIDATOR.load_json(CONTRACT))


def load_json_with_current_blocked_gpt_receipt(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if path.name == "judge-availability-receipt-v1.json":
        value["status"] = "BLOCKED_REQUIRED_JUDGE_UNAVAILABLE"
        value["all_judges_available"] = False
        value["judges"] = [
            {
                "model": "gpt-5.5-pro",
                "available": False,
                "reason": "credential_not_configured",
                "configured_credentials_tried": 0,
            }
        ]
    return value


def test_current_contract_is_ready_only_for_opus_review(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VALIDATOR, "load_json", load_json_with_current_blocked_gpt_receipt)
    receipt = VALIDATOR.validate(CONTRACT)

    assert receipt["status"] == "PASS_PROMPT_PACKAGE_READY_FOR_OPUS_REVIEW"
    assert receipt["selected_branch"] == "NATIVE_SCENE_REGENERATION"
    assert receipt["generation_permitted"] is False
    assert receipt["round_2_invariant"]["current_pair_verdict"] == "UNRESOLVABLE"
    assert receipt["round_2_invariant"]["max_abs_log_gain"] == 0
    assert receipt["visual_correctness_claimed"] is False


def test_generation_gate_has_material_blockers(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(VALIDATOR, "load_json", load_json_with_current_blocked_gpt_receipt)
    receipt = VALIDATOR.validate(CONTRACT)

    assert "GENERATION_NOT_AUTHORIZED" in receipt["generation_blockers"]
    assert "RESPONSIVE_SAFE_ZONES_ARE_PROVISIONAL" in receipt["generation_blockers"]
    assert "MANDATORY_TOURNAMENT_JUDGES_UNAVAILABLE" in receipt["generation_blockers"]
    assert "MODEL_REFERENCE_MACHINE_FOUNDER_APPROVAL_NOT_RECORDED" in receipt["generation_blockers"]
    assert "PASS_READY_TO_GENERATE_RECEIPT_NOT_ISSUED" in receipt["generation_blockers"]


@pytest.mark.parametrize(
    ("field", "value", "expected_message"),
    [
        ("calibration_evidence_for_both_assets", True, "may not claim calibration"),
        ("relight_authority", True, "relight authority"),
        ("raster_only_may_grant_relight_authority", True, "relight authority"),
    ],
)
def test_uncalibrated_pair_can_never_gain_relight_authority(
    field: str, value: bool, expected_message: str
) -> None:
    contract = contract_copy()
    contract["illumination_identifiability"][field] = value

    with pytest.raises(VALIDATOR.ValidationError, match=expected_message):
        VALIDATOR.validate_illumination_invariant(contract)


def test_calibration_records_cannot_be_invented_to_unlock_relighting() -> None:
    contract = contract_copy()
    illumination = contract["illumination_identifiability"]
    illumination["calibration_evidence_for_both_assets"] = True
    illumination["calibration_records"]["records"] = [
        {
            "asset_role": "locked_environment_composition_reference",
            "sha256": "0" * 64,
        },
        {
            "asset_role": "approved_two_model_identity_pose_and_product_placement_reference",
            "sha256": "1" * 64,
        },
    ]
    illumination["relight_authority"] = True
    illumination["protected_pixel_policy"]["max_abs_log_gain"] = 0.5

    with pytest.raises(VALIDATOR.ValidationError, match="may not claim calibration"):
        VALIDATOR.validate_illumination_invariant(contract)


def test_no_observed_contradiction_is_also_zero_gain() -> None:
    contract = contract_copy()
    illumination = contract["illumination_identifiability"]
    illumination["current_pair_verdict"] = "NO_OBSERVED_CONTRADICTION"
    illumination["relight_authority"] = True
    illumination["protected_pixel_policy"]["max_abs_log_gain"] = 0.01

    with pytest.raises(VALIDATOR.ValidationError, match="relight authority"):
        VALIDATOR.validate_illumination_invariant(contract)


def test_raster_only_verdicts_can_never_include_compatible() -> None:
    contract = contract_copy()
    contract["illumination_identifiability"]["raster_only_allowed_outcomes"].append(
        "VALIDATED_COMPATIBLE"
    )

    with pytest.raises(VALIDATOR.ValidationError, match="verdict set"):
        VALIDATOR.validate_illumination_invariant(contract)


def test_zero_gain_and_approved_low_frequency_source_are_tamper_evident() -> None:
    contract = contract_copy()
    illumination = contract["illumination_identifiability"]
    illumination["protected_pixel_policy"]["max_abs_log_gain"] = 0.000001

    with pytest.raises(VALIDATOR.ValidationError, match="max_abs_log_gain"):
        VALIDATOR.validate_illumination_invariant(contract)

    contract = contract_copy()
    illumination = contract["illumination_identifiability"]
    illumination["protected_pixel_policy"][
        "protected_low_frequency_band_source"
    ] = "locked_environment_composition_reference"

    with pytest.raises(VALIDATOR.ValidationError, match="low-frequency source"):
        VALIDATOR.validate_illumination_invariant(contract)


def test_json_boolean_aliases_cannot_bypass_numeric_or_boolean_rules() -> None:
    contract = contract_copy()
    contract["illumination_identifiability"]["protected_pixel_policy"]["max_abs_log_gain"] = False

    with pytest.raises(VALIDATOR.ValidationError, match="max_abs_log_gain"):
        VALIDATOR.validate_illumination_invariant(contract)

    contract = contract_copy()
    contract["illumination_identifiability"]["calibration_records"][
        "all_records_must_be_hash_verified"
    ] = 1

    with pytest.raises(VALIDATOR.ValidationError, match="calibration records"):
        VALIDATOR.validate_illumination_invariant(contract)

    contract = contract_copy()
    contract["native_scene_requirements"]["floor"]["both_shoes_make_visible_contact"] = 1

    with pytest.raises(VALIDATOR.ValidationError, match="native-scene"):
        VALIDATOR.validate_contract_semantics(contract)

    contract = contract_copy()
    contract["responsive_safe_zones"]["breakpoints"][0]["keep_clear"][0]["rect"][0] = True

    with pytest.raises(VALIDATOR.ValidationError, match="safe-zone rect"):
        VALIDATOR.validate_contract_semantics(contract)


def test_stale_generator_reference_hash_fails_closed() -> None:
    contract = contract_copy()
    item = contract["bindings"]["generator_conditioning_references"][0]
    item["sha256"] = "0" * 64

    with pytest.raises(VALIDATOR.ValidationError, match="stale hash"):
        VALIDATOR.audit_binding(item, "generator_conditioning")


def test_live_judge_receipt_hash_may_rotate_but_bytes_are_still_audited() -> None:
    contract = contract_copy()
    item = next(
        evidence
        for evidence in contract["bindings"]["validation_only_evidence"]
        if evidence["role"] == "live_judge_availability_receipt"
    )
    item["sha256"] = "0" * 64

    audit = VALIDATOR.audit_binding(
        item,
        "validation_only",
        rotating_hash_allowed=True,
    )

    assert audit["sha256"] == VALIDATOR.sha256(VALIDATOR.root_path(item["path"]))


def test_review_policy_hash_may_rotate_but_path_and_current_bytes_are_audited() -> None:
    contract = contract_copy()
    item = next(
        evidence
        for evidence in contract["bindings"]["validation_only_evidence"]
        if evidence["role"] == "mandatory_tournament_policy"
    )
    item["sha256"] = "0" * 64

    VALIDATOR.validate_binding_manifest_contract(contract["bindings"])
    audit = VALIDATOR.audit_binding(item, "validation_only", rotating_hash_allowed=True)

    assert audit["sha256"] == VALIDATOR.sha256(VALIDATOR.root_path(item["path"]))


@pytest.mark.parametrize(
    ("section", "mutation"),
    [
        ("prompt_payload", lambda value: value.__setitem__("target_model", "other-model")),
        ("output_contract", lambda value: value.__setitem__("format", "webp")),
        ("product_bindings", lambda value: value["lh-004"].__setitem__("required_regions", [])),
        ("per_region_fidelity_checks", lambda value: value[0].__setitem__("regions", [])),
    ],
)
def test_generation_semantics_are_sealed(section: str, mutation) -> None:
    contract = contract_copy()
    mutation(contract[section])

    with pytest.raises(VALIDATOR.ValidationError):
        VALIDATOR.validate_contract_semantics(contract)


def test_native_scene_review_sequence_excludes_gemini_and_synthesis() -> None:
    contract = contract_copy()

    assert contract["batch_and_review_contract"]["mandatory_review_sequence"] == [
        "mechanical_hash_and_role_validation",
        "per_region_product_fidelity_validation",
        "pasted_cutout_and_scene_coherence_rejection",
        "gpt-5.5-pro_independent_vision_score_at_least_95",
        "founder_contact_sheet_review_and_explicit_hash_approval",
    ]

    contract["batch_and_review_contract"]["mandatory_review_sequence"].insert(
        4,
        "gemini-3.1-pro-preview_independent_vision_score_at_least_95",
    )

    with pytest.raises(VALIDATOR.ValidationError, match="GPT-only"):
        VALIDATOR.validate_contract_semantics(contract)


def test_generator_conditioning_reference_order_is_sealed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = contract_copy()
    references = contract["bindings"]["generator_conditioning_references"]
    references[0], references[1] = references[1], references[0]
    original_load_json = VALIDATOR.load_json

    def load_json_with_reordered_references(path: Path) -> dict:
        if path == CONTRACT:
            return contract
        return original_load_json(path)

    monkeypatch.setattr(VALIDATOR, "load_json", load_json_with_reordered_references)

    with pytest.raises(VALIDATOR.ValidationError, match="sealed ordered"):
        VALIDATOR.validate(CONTRACT)


def test_canonical_dossier_roles_cannot_share_or_relabel_one_path() -> None:
    contract = contract_copy()
    evidence = contract["bindings"]["validation_only_evidence"]
    dossiers = [item for item in evidence if item["role"].endswith("_canonical_dossier")]
    for item in dossiers[1:]:
        item["path"] = dossiers[0]["path"]
        item["sha256"] = dossiers[0]["sha256"]

    with pytest.raises(VALIDATOR.ValidationError, match="duplicate validation-only"):
        VALIDATOR.validate_binding_manifest_contract(contract["bindings"])


def test_static_evidence_paths_cannot_be_swapped() -> None:
    contract = contract_copy()
    evidence = contract["bindings"]["validation_only_evidence"]
    first = evidence[0]
    second = evidence[1]
    first["path"], second["path"] = second["path"], first["path"]
    first["sha256"], second["sha256"] = second["sha256"], first["sha256"]

    with pytest.raises(VALIDATOR.ValidationError, match="static validation-only"):
        VALIDATOR.validate_binding_manifest_contract(contract["bindings"])


def test_current_product_sot_cannot_be_redirected() -> None:
    contract = contract_copy()
    contract["bindings"]["current_product_sot"][
        "path"
    ] = "wordpress-theme/skyyrose-flagship-2/data/product-presentation-registry.json"

    with pytest.raises(VALIDATOR.ValidationError, match="current product SOT binding"):
        VALIDATOR.validate_binding_manifest_contract(contract["bindings"])


def test_receipt_path_cannot_escape_scene_preflight_directory() -> None:
    with pytest.raises(VALIDATOR.ValidationError, match="inside the native-scene preflight"):
        VALIDATOR.resolve_receipt_path(Path("/tmp/native-scene-receipt.json"))


def test_live_judge_receipt_must_be_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    contract = contract_copy()
    generator_items = contract["bindings"]["generator_conditioning_references"]
    evidence_items = contract["bindings"]["validation_only_evidence"]
    generator_by_role = {item["role"]: item for item in generator_items}
    evidence_by_role = {item["role"]: item for item in evidence_items}
    judge_path = VALIDATOR.root_path(evidence_by_role["live_judge_availability_receipt"]["path"])
    original_load_json = VALIDATOR.load_json

    def load_json_with_stale_live_judges(path: Path) -> dict:
        value = original_load_json(path)
        if path == judge_path:
            value["status"] = "PASS_ALL_JUDGES_AVAILABLE"
            value["all_judges_available"] = True
            value["checked_at"] = (datetime.now(UTC) - timedelta(hours=1)).isoformat()
            value["judges"] = [{"model": "gpt-5.5-pro", "available": True}]
        return value

    monkeypatch.setattr(VALIDATOR, "load_json", load_json_with_stale_live_judges)
    product_sot_hash = contract["bindings"]["current_product_sot"]["sha256"]

    with pytest.raises(VALIDATOR.ValidationError, match="judge receipt is stale"):
        VALIDATOR.validate_receipt_lineage(
            evidence_by_role,
            generator_by_role,
            product_sot_hash,
        )


@pytest.mark.parametrize(
    ("field", "weakened_value"),
    [
        ("schema", "other-policy"),
        ("all_judges_available", False),
        ("source_hashes_current", False),
        ("evaluation_scope", "selected outputs only"),
        ("downstream_rule", "founder approval optional"),
    ],
)
def test_complete_tournament_policy_is_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    weakened_value,
) -> None:
    contract = contract_copy()
    generator_items = contract["bindings"]["generator_conditioning_references"]
    evidence_items = contract["bindings"]["validation_only_evidence"]
    generator_by_role = {item["role"]: item for item in generator_items}
    evidence_by_role = {item["role"]: item for item in evidence_items}
    policy_path = VALIDATOR.root_path(evidence_by_role["mandatory_tournament_policy"]["path"])
    original_load_json = VALIDATOR.load_json

    def load_json_with_weakened_policy(path: Path) -> dict:
        value = original_load_json(path)
        if path == policy_path:
            value[field] = weakened_value
        return value

    monkeypatch.setattr(VALIDATOR, "load_json", load_json_with_weakened_policy)
    product_sot_hash = contract["bindings"]["current_product_sot"]["sha256"]

    with pytest.raises(VALIDATOR.ValidationError, match="complete fail-closed policy"):
        VALIDATOR.validate_receipt_lineage(
            evidence_by_role,
            generator_by_role,
            product_sot_hash,
        )


def test_candidate_and_founder_reapproval_state_cannot_be_relaxed() -> None:
    contract = contract_copy()
    contract["output_contract"]["approval_state"] = "APPROVED"

    with pytest.raises(VALIDATOR.ValidationError, match="approval state"):
        VALIDATOR.validate_contract_semantics(contract)

    contract = contract_copy()
    contract["authorization"]["founder_reapproval_required"] = False

    with pytest.raises(VALIDATOR.ValidationError, match="founder reapproval"):
        VALIDATOR.validate_contract_semantics(contract)


def test_unverifiable_product_region_is_always_fatal() -> None:
    contract = contract_copy()
    contract["per_region_fidelity_checks"][0]["unverifiable_is_failure"] = False

    with pytest.raises(VALIDATOR.ValidationError, match="unverifiable product region"):
        VALIDATOR.validate_contract_semantics(contract)


def test_native_scene_physics_and_hero_rose_rules_are_tamper_evident() -> None:
    contract = contract_copy()
    contract["native_scene_requirements"]["shadows"]["floating_feet_fatal"] = False

    with pytest.raises(VALIDATOR.ValidationError, match="native-scene"):
        VALIDATOR.validate_contract_semantics(contract)

    contract = contract_copy()
    contract["native_scene_requirements"]["hero_rose"][
        "minimum_saliency_ratio_vs_dominant_model"
    ] = 0.1

    with pytest.raises(VALIDATOR.ValidationError, match="native-scene"):
        VALIDATOR.validate_contract_semantics(contract)


def test_wiring_and_deployment_remain_forbidden() -> None:
    contract = contract_copy()
    contract["output_contract"]["v2_wiring_state"] = "ALLOWED"

    with pytest.raises(VALIDATOR.ValidationError, match="V2 wiring"):
        VALIDATOR.validate_contract_semantics(contract)

    contract = contract_copy()
    contract["authorization"]["deployment_authorized"] = True

    with pytest.raises(VALIDATOR.ValidationError, match="deployment_authorized"):
        VALIDATOR.validate_contract_semantics(contract)
