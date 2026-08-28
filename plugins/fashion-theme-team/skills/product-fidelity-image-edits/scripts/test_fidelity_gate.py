#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import fidelity_gate
from PIL import Image, ImageDraw


def save_rgb(path: Path, color: tuple[int, int, int], size: tuple[int, int] = (64, 64)) -> None:
    Image.new("RGB", size, color).save(path)


def save_protected(path: Path, *, real_alpha: bool) -> None:
    image = Image.new("RGBA", (32, 32), (10, 20, 30, 255))
    if real_alpha:
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, 7, 31), fill=(0, 0, 0, 0))
    image.save(path)


class FidelityGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.workspace = Path(self.tempdir.name)
        self.target = self.workspace / "scene.png"
        self.reference = self.workspace / "br-007-side.jpg"
        save_rgb(self.target, (15, 15, 15))
        save_rgb(self.reference, (245, 245, 245), (48, 60))

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def localized_contract(self) -> dict:
        return {
            "schema": fidelity_gate.SCHEMA,
            "operation": "localized_product_patch",
            "target": {
                "path": self.target.name,
                "sha256": fidelity_gate.sha256_file(self.target),
            },
            "generator": {"route": "masked_edit", "supports_explicit_mask": True},
            "edit_region": {
                "type": "bbox",
                "coordinate_space": "pixels",
                "bbox": [16, 16, 48, 48],
                "feather_px": 0,
            },
            "references": [
                {
                    "sku": "br-007",
                    "view": "side",
                    "role": "physical_product_authority",
                    "path": self.reference.name,
                    "sha256": fidelity_gate.sha256_file(self.reference),
                }
            ],
            "verification": {
                "max_outside_changed_pixels": 0,
                "min_inside_changed_pixels": 1,
                "per_channel_tolerance": 0,
            },
            "semantic_review": {"required": True, "scope": "logo_and_construction"},
        }

    def native_contract(self) -> dict:
        matte = self.workspace / "br-007-garment-matte.png"
        save_protected(matte, real_alpha=True)
        return {
            "schema": fidelity_gate.SCHEMA,
            "operation": "native_collection_scene",
            "target": {
                "path": self.reference.name,
                "sha256": fidelity_gate.sha256_file(self.reference),
            },
            "generator": {
                "route": "environment_rebuild_around_source",
                "supports_reference_images": True,
                "product_redraw_allowed": False,
            },
            "pose_change": False,
            "collection_scene": {
                "collection_id": "black-rose",
                "scene_id": "BR-COMMERCE-2",
                "story_role": "bridge overlook",
                "scene_variation": "bridge-overlook",
                "products": [{"sku": "br-007", "cta": "Shop the Shorts"}],
            },
            "references": [
                {
                    "sku": "br-007",
                    "view": "on_model_side",
                    "role": "physical_product_authority",
                    "path": self.reference.name,
                    "sha256": fidelity_gate.sha256_file(self.reference),
                    "authority_state": "founder_approved",
                    "approved_pose_source": True,
                },
                {
                    "sku": "br-007",
                    "view": "on_model_side",
                    "role": "protected_garment_matte",
                    "path": matte.name,
                    "sha256": fidelity_gate.sha256_file(matte),
                    "authority_state": "founder_approved",
                    "approved_pose_source": True,
                },
            ],
            "protected_placements": [{"path": matte.name, "x": 16, "y": 16}],
            "optical_contract": {
                "output_dimensions": [64, 64],
                "subject_bbox": [12, 4, 52, 62],
                "horizon_y": 30,
                "foot_contact_points": [[20, 58], [44, 58]],
                "lens_class": "normal",
                "camera_height": "eye_level",
                "key_light": {
                    "direction": "camera_left",
                    "temperature_k": 3200,
                    "softness": "soft",
                },
                "shadow": {"direction": "camera_right", "softness": "soft"},
                "floor_response": "polished",
                "reflection_mode": "subtle",
                "contact_shadow_required": True,
                "edge_spill_required": True,
                "depth_occlusion_required": True,
                "atmospheric_depth_required": True,
                "scene_grain_match_required": True,
            },
            "review_gate": {
                "candidate_author": "scene-builder",
                "independent_reviewer_required": True,
                "minimum_scores": {
                    "product_fidelity": 95,
                    "optical_integration": 90,
                    "anatomy_pose": 90,
                    "collection_story": 90,
                    "commerce_readiness": 90,
                },
            },
            "promotion": {
                "founder_approval_required": True,
                "wiring_allowed": False,
                "deployment_allowed": False,
            },
        }

    def native_candidate(self, contract: dict) -> Path:
        candidate = self.workspace / "native-candidate.png"
        image = Image.new("RGBA", (64, 64), (15, 15, 15, 255))
        matte_path = self.workspace / contract["protected_placements"][0]["path"]
        with Image.open(matte_path) as opened:
            matte = opened.convert("RGBA")
        image.alpha_composite(matte, (16, 16))
        image.convert("RGB").save(candidate)
        return candidate

    def native_review(self, candidate: Path, **overrides: object) -> Path:
        review = {
            "schema": "product-fidelity-native-review.v1",
            "candidate_sha256": fidelity_gate.sha256_file(candidate),
            "reviewer": "independent-reviewer",
            "scores": {
                "product_fidelity": 98,
                "optical_integration": 95,
                "anatomy_pose": 96,
                "collection_story": 96,
                "commerce_readiness": 95,
            },
            "hard_fails": [],
            "founder_status": "PENDING",
        }
        review.update(overrides)
        path = self.workspace / "native-review.json"
        path.write_text(json.dumps(review), encoding="utf-8")
        return path

    def test_maskless_builtin_route_is_blocked(self) -> None:
        contract = self.localized_contract()
        contract["generator"] = {
            "route": "builtin-imagegen",
            "supports_explicit_mask": False,
        }
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        codes = {finding["code"] for finding in receipt["findings"]}
        self.assertIn("MASKLESS_GENERATOR_FOR_LOCAL_EDIT", codes)
        self.assertIn("EXPLICIT_MASK_CAPABILITY_REQUIRED", codes)

    def test_outside_mask_change_is_blocked(self) -> None:
        contract = self.localized_contract()
        candidate = self.workspace / "candidate.png"
        image = Image.open(self.target).convert("RGBA")
        image.putpixel((20, 20), (255, 0, 0, 255))
        image.putpixel((2, 2), (255, 0, 0, 255))
        image.save(candidate)
        receipt = fidelity_gate.verify_localized_output(contract, self.workspace, candidate)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual(1, receipt["outside_changed_pixels"])

    def test_inside_only_change_passes(self) -> None:
        contract = self.localized_contract()
        candidate = self.workspace / "candidate.png"
        image = Image.open(self.target).convert("RGBA")
        image.putpixel((20, 20), (255, 0, 0, 255))
        image.save(candidate)
        receipt = fidelity_gate.verify_localized_output(contract, self.workspace, candidate)
        self.assertEqual("PASS", receipt["status"])
        self.assertEqual(0, receipt["outside_changed_pixels"])
        self.assertEqual(1, receipt["inside_changed_pixels"])

    def test_dimension_drift_is_blocked(self) -> None:
        contract = self.localized_contract()
        candidate = self.workspace / "candidate.png"
        save_rgb(candidate, (15, 15, 15), (63, 64))
        receipt = fidelity_gate.verify_localized_output(contract, self.workspace, candidate)
        self.assertEqual("DIMENSION_DRIFT", receipt["code"])

    def test_noop_patch_is_blocked(self) -> None:
        contract = self.localized_contract()
        receipt = fidelity_gate.verify_localized_output(contract, self.workspace, self.target)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual("NO_INTENDED_CHANGE", receipt["code"])

    def test_path_escape_is_blocked(self) -> None:
        contract = self.localized_contract()
        contract["references"][0]["path"] = "../outside.jpg"
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn("PATH_OUTSIDE_WORKSPACE", {item["code"] for item in receipt["findings"]})

    def test_filename_view_conflict_is_reported_and_canonicalized(self) -> None:
        contract = self.localized_contract()
        conflicting = self.workspace / "br-007-back-source.jpg"
        self.reference.rename(conflicting)
        contract["references"][0]["path"] = conflicting.name
        contract["references"][0]["sha256"] = fidelity_gate.sha256_file(conflicting)
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("PASS", receipt["status"])
        self.assertIn("FILENAME_VIEW_CONFLICT", {item["code"] for item in receipt["findings"]})
        contract_path = self.workspace / "contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        pack = fidelity_gate.optimize_references(
            contract, contract_path, self.workspace, self.workspace / "pack"
        )
        canonical = Path(pack["references"][0]["canonical_path"])
        self.assertIn("br-007-side-physical-product-authority", canonical.name)
        self.assertNotIn("back", canonical.name)

    def test_fake_protected_alpha_is_blocked(self) -> None:
        protected = self.workspace / "protected.png"
        save_protected(protected, real_alpha=False)
        contract = {
            "schema": fidelity_gate.SCHEMA,
            "operation": "protected_scene_composite",
            "target": {
                "path": self.target.name,
                "sha256": fidelity_gate.sha256_file(self.target),
            },
            "generator": {
                "route": "background_only_then_composite",
                "supports_explicit_mask": False,
            },
            "pose_change": False,
            "references": [
                {
                    "sku": "sg-009",
                    "view": "on_model_front",
                    "role": "protected_product_layer",
                    "path": protected.name,
                    "sha256": fidelity_gate.sha256_file(protected),
                    "approved_pose_source": True,
                }
            ],
        }
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "PROTECTED_LAYER_REAL_ALPHA_REQUIRED",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_pose_change_without_approved_source_is_blocked(self) -> None:
        protected = self.workspace / "protected.png"
        save_protected(protected, real_alpha=True)
        contract = {
            "schema": fidelity_gate.SCHEMA,
            "operation": "protected_scene_composite",
            "target": {
                "path": self.target.name,
                "sha256": fidelity_gate.sha256_file(self.target),
            },
            "generator": {
                "route": "background_only_then_composite",
                "supports_explicit_mask": False,
            },
            "pose_change": True,
            "references": [
                {
                    "sku": "sg-009",
                    "view": "on_model_front",
                    "role": "protected_product_layer",
                    "path": protected.name,
                    "sha256": fidelity_gate.sha256_file(protected),
                    "approved_pose_source": False,
                }
            ],
        }
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "POSE_CHANGE_WITHOUT_APPROVED_SOURCE",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_native_scene_requires_optical_contract(self) -> None:
        contract = self.native_contract()
        del contract["optical_contract"]
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn("OPTICAL_CONTRACT_REQUIRED", {item["code"] for item in receipt["findings"]})

    def test_native_scene_requires_garment_matte_for_exact_route(self) -> None:
        contract = self.native_contract()
        contract["references"] = [contract["references"][0]]
        contract["protected_placements"] = []
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        codes = {item["code"] for item in receipt["findings"]}
        self.assertIn("GARMENT_MATTE_REQUIRED", codes)
        self.assertIn("GARMENT_PLACEMENTS_REQUIRED", codes)

    def test_native_scene_rejects_unapproved_product_reference(self) -> None:
        contract = self.native_contract()
        contract["references"][0]["authority_state"] = "rejected"
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "REFERENCE_AUTHORITY_STATE_REQUIRED",
            {item["code"] for item in receipt["findings"]},
        )

    def test_native_scene_requires_target_to_be_approved_on_model_authority(
        self,
    ) -> None:
        contract = self.native_contract()
        contract["target"] = {
            "path": self.target.name,
            "sha256": fidelity_gate.sha256_file(self.target),
        }
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "APPROVED_ON_MODEL_TARGET_REQUIRED",
            {item["code"] for item in receipt["findings"]},
        )

    def test_native_scene_logo_reference_cannot_replace_product_truth(self) -> None:
        contract = self.native_contract()
        contract["references"][0]["role"] = "exact_logo_or_patch_authority"
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        codes = {item["code"] for item in receipt["findings"]}
        self.assertIn("CAST_SKU_PRODUCT_TRUTH_MISSING", codes)
        self.assertIn("APPROVED_ON_MODEL_TARGET_REQUIRED", codes)

    def test_native_scene_verify_requires_independent_review(self) -> None:
        contract = self.native_contract()
        candidate = self.native_candidate(contract)
        receipt = fidelity_gate.verify_native_output(contract, self.workspace, candidate, None)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual("NATIVE_REVIEW_REQUIRED", receipt["code"])

    def test_native_scene_low_optical_score_is_blocked(self) -> None:
        contract = self.native_contract()
        candidate = self.native_candidate(contract)
        scores = {
            "product_fidelity": 98,
            "optical_integration": 70,
            "anatomy_pose": 96,
            "collection_story": 96,
            "commerce_readiness": 95,
        }
        review = self.native_review(candidate, scores=scores)
        receipt = fidelity_gate.verify_native_output(contract, self.workspace, candidate, review)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn("SCORE_BELOW_MINIMUM:optical_integration", receipt["findings"])

    def test_native_scene_builder_self_review_is_blocked(self) -> None:
        contract = self.native_contract()
        candidate = self.native_candidate(contract)
        review = self.native_review(candidate, reviewer="scene-builder")
        receipt = fidelity_gate.verify_native_output(contract, self.workspace, candidate, review)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn("BUILDER_SELF_REVIEW_FORBIDDEN", receipt["findings"])

    def test_native_scene_passes_as_unwired_review_candidate(self) -> None:
        contract = self.native_contract()
        candidate = self.native_candidate(contract)
        review = self.native_review(candidate)
        receipt = fidelity_gate.verify_native_output(contract, self.workspace, candidate, review)
        self.assertEqual("PASS", receipt["status"])
        self.assertEqual("PASS_REVIEW_CANDIDATE", receipt["code"])
        self.assertFalse(receipt["wiring_allowed"])
        self.assertEqual("FOUNDER_REVIEW_REQUIRED", receipt["promotion_state"])

    def test_native_scene_cli_verify_review_round_trip(self) -> None:
        contract = self.native_contract()
        contract_path = self.workspace / "native-contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        candidate = self.native_candidate(contract)
        review = self.native_review(candidate)
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(fidelity_gate.__file__)),
                "verify",
                "--contract",
                str(contract_path),
                "--workspace",
                str(self.workspace),
                "--output",
                str(candidate),
                "--review",
                str(review),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        self.assertEqual(0, completed.returncode, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual("PASS_REVIEW_CANDIDATE", receipt["code"])
        self.assertFalse(receipt["wiring_allowed"])


if __name__ == "__main__":
    unittest.main()
