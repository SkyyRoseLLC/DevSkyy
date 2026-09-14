"""Offline regression checks for the current founder image-review contract."""

import importlib.util
import json
from pathlib import Path

import pytest

THEME = Path(__file__).resolve().parents[1] / "wordpress-theme/skyyrose-flagship-2"
spec = importlib.util.spec_from_file_location(
    "v2_image_preflight", THEME / "scripts/validate-image-generation-preflight.py"
)
preflight = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preflight)


def policy():
    return json.loads((THEME / "data/image-generation-tournament-policy-v1.json").read_text())


def test_current_founder_policy_passes_without_running_provider_calls():
    preflight.validate_tournament_policy(policy())


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("required_vision_judges", ["gpt-5.5-pro", "gemini-3.1-pro-preview"]),
        ("synthesis_model", "claude-opus-5"),
        ("minimum_each_vision_score", 94),
        ("minimum_final_score", 94),
        ("founder_approval_required", False),
        ("source_hashes_current", False),
        ("required_hallucination_veto_result", True),
        ("unverifiable_required_regions", 1),
        ("all_judges_available", False),
    ],
)
def test_stale_or_weakened_policy_is_rejected(field, value):
    altered = policy()
    altered[field] = value
    with pytest.raises(SystemExit, match="FAIL"):
        preflight.validate_tournament_policy(altered)
