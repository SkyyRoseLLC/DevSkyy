from __future__ import annotations

import pytest

from scripts.validate_dossier import canonical_technique, recognized_techniques
from skyyrose.core.dossier_schema import parse_branding_regions
from skyyrose.core.material_fidelity import (
    MaterialFidelityError,
    compile_material_lock_prompt,
    require_complete_material_locks,
)
from skyyrose.elite_studio.agents.vision_audit_agent import _build_audit_prompt
from skyyrose.elite_studio.synthesis.prompts.decoration_prompts import (
    build_decoration_prompt,
)
from skyyrose.elite_studio.synthesis.stages.decoration_inpaint import (
    _compose_decoration_prompt,
)
from skyyrose.elite_studio.synthesis.stages.mask_deriver import parse_branding_entries


LOCKED_BRANDING = """- **front-right-chest** (~2in): Exact rose cluster.
  **Technique:** silicone. **Color:** tonal grey.
  **Material:** molded silicone elastomer.
  **Construction:** one molded cut-out unit.
  **Surface:** smooth rubber-like raised relief with no stitch texture.
  **Attachment:** applied flush to right-chest fabric; chemistry unverified.
  **Verify:** crisp perimeter and coherent molded highlights.
  **Reject:** thread embroidery, flat print, woven patch, or wrong placement.
"""


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("embroidered-patch (sewn-on hardware)", "embroidered-patch"),
        ("sublimated/printed", "sublimated"),
        ("patch (sewn-on hardware) + stitched", "stitched"),
        ("invented decoration process", ""),
    ],
)
def test_dossier_gate_preserves_detail_but_requires_a_known_technique(
    value: str, expected: str
) -> None:
    assert canonical_technique(value) == expected


def test_dossier_gate_tracks_every_combined_process_for_contradiction_checks() -> None:
    assert recognized_techniques("sublimated/printed") == ("sublimated", "printed")
    assert recognized_techniques("patch (sewn-on hardware) + stitched") == (
        "stitched",
        "patch",
    )


def test_material_lock_compiler_emits_all_machine_gate_fields() -> None:
    block = compile_material_lock_prompt(parse_branding_regions(LOCKED_BRANDING))
    for required in (
        "MATERIAL:",
        "CONSTRUCTION:",
        "SURFACE / LIGHT RESPONSE:",
        "ATTACHMENT:",
        "VERIFY:",
        "REJECT:",
    ):
        assert required in block
    assert "technique names are physical manufacturing facts" in block


def test_pre_generation_gate_rejects_an_unlocked_region() -> None:
    regions = parse_branding_regions(
        "- **front-right-chest** (~2in): Exact rose. **Technique:** silicone. "
        "**Color:** grey."
    )
    with pytest.raises(MaterialFidelityError, match="front-right-chest"):
        require_complete_material_locks(
            version="v1",
            regions=regions,
            context="br-005:front",
        )


def test_flux_decoration_prompt_has_exact_silicone_physics() -> None:
    prompt = build_decoration_prompt(
        decoration_description="exact three-rose-cluster cut-out",
        technique="silicone",
        region="front-right-chest",
        color="tonal grey",
    )
    assert "molded silicone cut-out" in prompt
    assert "ZERO embroidery thread" in prompt
    assert "ZERO multicolor applique" in prompt


def test_stage3_and_independent_audit_receive_the_same_material_contract() -> None:
    entries = parse_branding_entries(LOCKED_BRANDING)
    stage3 = _compose_decoration_prompt(
        decoration_entries=entries,
        prior_violations=[],
        attempt=1,
        lora_trigger=None,
        negative_block="- NO sleeve placement",
    )
    dossier = {
        "name": "Signature Hoodie",
        "sku": "br-005",
        "garment_type_lock": "Black pullover hoodie. NOT a zip-up.",
        "branding_block": LOCKED_BRANDING,
        "negative_block": "- NO sleeve placement",
    }
    audit = _build_audit_prompt(dossier, view="front")
    for prompt in (stage3, audit):
        assert "MATERIAL LOCKS — PHYSICAL PRODUCT TRUTH" in prompt
        assert "smooth rubber-like raised relief" in prompt
        assert "thread embroidery, flat print, woven patch" in prompt
    assert "wrong surface physics, attachment, or technique" in audit
