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
        self.catalog = self.workspace / "skyyrose-catalog.csv"
        self.sot_manifest = self.workspace / "sot-images.json"
        self.dossier = self.workspace / "product-dossier.md"
        self.catalog.write_text("sku,name\nbr-007,Black Rose Shorts\n", encoding="utf-8")
        self.sot_manifest.write_text(
            json.dumps({"images": {"br-007": {"side": self.reference.name}}}),
            encoding="utf-8",
        )
        self.dossier.write_text("# Product dossier\n", encoding="utf-8")

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def run_gate_cli(self, *arguments: str) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(Path(fidelity_gate.__file__)), *arguments],
            check=False,
            capture_output=True,
            text=True,
        )

    def authority_binding(self, source: Path, *, sku: str, view: str, role: str) -> dict[str, str]:
        receipt = {
            "schema": fidelity_gate.SOURCE_AUTHORITY_SCHEMA,
            "status": "VERIFIED",
            "sku": sku,
            "view": view,
            "role": role,
            "source_sha256": fidelity_gate.sha256_file(source),
            "catalog": {
                "path": self.catalog.name,
                "sha256": fidelity_gate.sha256_file(self.catalog),
            },
            "sot_manifest": {
                "path": self.sot_manifest.name,
                "sha256": fidelity_gate.sha256_file(self.sot_manifest),
            },
            "dossier": {
                "path": self.dossier.name,
                "sha256": fidelity_gate.sha256_file(self.dossier),
            },
            "resolver": {
                "id": "devskyy-product-authority-resolver",
                "version": "1.0.0",
            },
        }
        receipt_path = self.workspace / f"{sku}-{view}-{role}-authority.json"
        receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
        return {
            "path": receipt_path.name,
            "sha256": fidelity_gate.sha256_file(receipt_path),
        }

    def test_cli_rejects_ambiguous_or_nonfinite_contract_json(self) -> None:
        for payload in (
            '{"schema":"first","schema":"second"}',
            '{"schema":"product-fidelity-edit.v1","score":NaN}',
        ):
            with self.subTest(payload=payload):
                contract_path = self.workspace / "malformed-contract.json"
                contract_path.write_text(payload, encoding="utf-8")

                completed = self.run_gate_cli(
                    "preflight",
                    "--contract",
                    str(contract_path),
                    "--workspace",
                    str(self.workspace),
                )

                self.assertEqual(2, completed.returncode, completed.stderr)
                receipt = json.loads(completed.stdout)
                self.assertEqual("BLOCKED", receipt["status"])
                self.assertEqual("INVALID_CONTRACT_JSON", receipt["code"])

    def test_cli_rejects_invalid_utf8_contract(self) -> None:
        contract_path = self.workspace / "invalid-utf8-contract.json"
        contract_path.write_bytes(b"\xff\xfe")

        completed = self.run_gate_cli(
            "preflight",
            "--contract",
            str(contract_path),
            "--workspace",
            str(self.workspace),
        )

        self.assertEqual(2, completed.returncode, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual("INVALID_CONTRACT_JSON", receipt["code"])

    def test_cli_rejects_contract_outside_workspace(self) -> None:
        with tempfile.TemporaryDirectory() as outside_dir:
            contract_path = Path(outside_dir) / "contract.json"
            contract_path.write_text("{}", encoding="utf-8")
            completed = self.run_gate_cli(
                "preflight",
                "--contract",
                str(contract_path),
                "--workspace",
                str(self.workspace),
            )

        self.assertEqual(2, completed.returncode, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual("CONTRACT_OUTSIDE_WORKSPACE", receipt["code"])

    def test_cli_rejects_receipt_outside_workspace(self) -> None:
        contract = self.localized_contract()
        contract_path = self.workspace / "contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        with tempfile.TemporaryDirectory() as outside_dir:
            completed = self.run_gate_cli(
                "preflight",
                "--contract",
                str(contract_path),
                "--workspace",
                str(self.workspace),
                "--receipt",
                str(Path(outside_dir) / "receipt.json"),
            )

        self.assertEqual(2, completed.returncode, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual("RECEIPT_OUTSIDE_WORKSPACE", receipt["code"])

    def test_cli_rejects_receipt_collision_with_bound_input(self) -> None:
        contract = self.localized_contract()
        contract_path = self.workspace / "contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")

        completed = self.run_gate_cli(
            "preflight",
            "--contract",
            str(contract_path),
            "--workspace",
            str(self.workspace),
            "--receipt",
            str(self.target),
        )

        self.assertEqual(2, completed.returncode, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual("RECEIPT_INPUT_COLLISION", receipt["code"])
        with Image.open(self.target) as preserved:
            self.assertEqual((64, 64), preserved.size)

    def test_cli_receipt_failure_is_structured_and_never_creates_parent(self) -> None:
        contract = self.localized_contract()
        contract_path = self.workspace / "contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        receipt_path = self.workspace / "missing-parent" / "receipt.json"

        completed = self.run_gate_cli(
            "preflight",
            "--contract",
            str(contract_path),
            "--workspace",
            str(self.workspace),
            "--receipt",
            str(receipt_path),
        )

        self.assertEqual(2, completed.returncode, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual("RECEIPT_PARENT_MISSING", receipt["code"])
        self.assertFalse(receipt_path.parent.exists())

    def localized_contract(self) -> dict:
        return {
            "schema": fidelity_gate.SCHEMA,
            "operation": "localized_product_patch",
            "target": {
                "path": self.target.name,
                "sha256": fidelity_gate.sha256_file(self.target),
            },
            "generator": {
                "route": "masked_edit",
                "supports_explicit_mask": True,
                "model_id": "openai-responses-image-mask",
            },
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
                    "verbatim_text": [],
                    "authority_receipt": self.authority_binding(
                        self.reference,
                        sku="br-007",
                        view="side",
                        role="physical_product_authority",
                    ),
                }
            ],
            "verification": {
                "max_outside_changed_pixels": 0,
                "min_inside_changed_pixels": 1,
                "per_channel_tolerance": 0,
            },
            "semantic_review": {
                "required": True,
                "scope": "logo_and_construction",
                "candidate_author": "image-builder",
                "required_checks": sorted(fidelity_gate.SEMANTIC_BASE_CHECKS),
            },
        }

    def localized_review(self, contract: dict, candidate: Path, **overrides: object) -> Path:
        contract_sha = fidelity_gate.sha256_json(contract)
        review = {
            "schema": fidelity_gate.SEMANTIC_REVIEW_SCHEMA,
            "contract_sha256": contract_sha,
            "candidate_sha256": fidelity_gate.sha256_file(candidate),
            "reviewer": "independent-reviewer",
            "scope": "logo_and_construction",
            "verdict": "PASS",
            "findings": [],
            "checks": dict.fromkeys(contract["semantic_review"]["required_checks"], "PASS"),
            "expected_verbatim_text": [],
        }
        review.update(overrides)
        path = self.workspace / "localized-review.json"
        path.write_text(json.dumps(review), encoding="utf-8")
        return path

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
                "model_id": "openai-responses-image-mask",
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
                    "verbatim_text": [],
                    "authority_receipt": self.authority_binding(
                        self.reference,
                        sku="br-007",
                        view="on_model_side",
                        role="physical_product_authority",
                    ),
                },
                {
                    "sku": "br-007",
                    "view": "on_model_side",
                    "role": "protected_garment_matte",
                    "path": matte.name,
                    "sha256": fidelity_gate.sha256_file(matte),
                    "authority_state": "founder_approved",
                    "approved_pose_source": True,
                    "verbatim_text": [],
                    "authority_receipt": self.authority_binding(
                        matte,
                        sku="br-007",
                        view="on_model_side",
                        role="protected_garment_matte",
                    ),
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

    def protected_contract(self) -> dict:
        protected = self.workspace / "protected.png"
        save_protected(protected, real_alpha=True)
        return {
            "schema": fidelity_gate.SCHEMA,
            "operation": "protected_scene_composite",
            "target": {
                "path": self.target.name,
                "sha256": fidelity_gate.sha256_file(self.target),
            },
            "generator": {
                "route": "background_only_then_composite",
                "supports_explicit_mask": False,
                "model_id": "adobe-precise-composite",
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
                    "verbatim_text": [],
                    "authority_receipt": self.authority_binding(
                        protected,
                        sku="sg-009",
                        view="on_model_front",
                        role="protected_product_layer",
                    ),
                }
            ],
            "protected_placements": [{"path": protected.name, "x": 16, "y": 16}],
            "verification": {"protected_rgb_must_match": True},
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

    def protected_candidate(self, contract: dict, candidate: Path) -> Path:
        image = Image.new("RGBA", (64, 64), (15, 15, 15, 255))
        placement = contract["protected_placements"][0]
        with Image.open(self.workspace / placement["path"]) as opened:
            layer = opened.convert("RGBA")
        image.alpha_composite(layer, (placement["x"], placement["y"]))
        image.convert("RGB").save(candidate)
        return candidate

    def native_review(self, contract: dict, candidate: Path, **overrides: object) -> Path:
        review = {
            "schema": fidelity_gate.NATIVE_REVIEW_SCHEMA,
            "contract_sha256": fidelity_gate.sha256_json(contract),
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

    def test_registered_maskless_model_cannot_self_assert_mask_capability(self) -> None:
        contract = self.localized_contract()
        contract["generator"] = {
            "route": "masked_edit",
            "supports_explicit_mask": True,
            "model_id": "codex-imagegen-wrapper",
        }

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "MODEL_CAPABILITY_BLOCKED",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_preflight_receipt_binds_contract_hash(self) -> None:
        contract = self.localized_contract()

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual(fidelity_gate.sha256_json(contract), receipt["contract_sha256"])

    def test_legacy_v1_contract_requires_explicit_migration(self) -> None:
        contract = self.localized_contract()
        contract["schema"] = "product-fidelity-edit.v1"

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "SCHEMA_MIGRATION_REQUIRED",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_localized_preflight_requires_candidate_author(self) -> None:
        contract = self.localized_contract()
        del contract["semantic_review"]["candidate_author"]

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "CANDIDATE_AUTHOR_REQUIRED",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_localized_semantic_checks_cannot_be_weakened_by_contract(self) -> None:
        contract = self.localized_contract()
        contract["semantic_review"]["required_checks"] = ["none"]

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "SEMANTIC_REQUIRED_CHECKS_INVALID",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_product_reference_must_explicitly_declare_verbatim_text(self) -> None:
        contract = self.localized_contract()
        del contract["references"][0]["verbatim_text"]

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "VERBATIM_TEXT_DECLARATION_REQUIRED",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_product_reference_requires_hash_bound_source_authority(self) -> None:
        contract = self.localized_contract()
        del contract["references"][0]["authority_receipt"]

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "AUTHORITY_BINDING_REQUIRED",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_source_authority_blocks_catalog_drift(self) -> None:
        contract = self.localized_contract()
        self.catalog.write_text("sku,name\nbr-007,Changed\n", encoding="utf-8")

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "AUTHORITY_HASH_DRIFT",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_localized_preflight_blocks_malformed_pixel_tolerance(self) -> None:
        contract = self.localized_contract()
        contract["verification"]["per_channel_tolerance"] = "zero"

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "PER_CHANNEL_TOLERANCE_INVALID",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_localized_preflight_forbids_contract_weakened_pixel_policy(self) -> None:
        for field, value, code in (
            ("max_outside_changed_pixels", 4096, "OUTSIDE_CHANGE_TOLERANCE_FORBIDDEN"),
            ("per_channel_tolerance", 255, "PER_CHANNEL_TOLERANCE_FORBIDDEN"),
        ):
            with self.subTest(field=field):
                contract = self.localized_contract()
                contract["verification"][field] = value

                receipt = fidelity_gate.validate_contract(contract, self.workspace)

                self.assertEqual("BLOCKED", receipt["status"])
                self.assertIn(code, {item["code"] for item in receipt["findings"]})

    def test_localized_verifier_refuses_unsafe_pixel_policy_directly(self) -> None:
        contract = self.localized_contract()
        contract["verification"]["max_outside_changed_pixels"] = 4096
        candidate = self.workspace / "candidate.png"
        save_rgb(candidate, (255, 0, 0))

        receipt = fidelity_gate.verify_localized_output(contract, self.workspace, candidate)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual("UNSAFE_VERIFICATION_POLICY", receipt["code"])

    def test_localized_preflight_blocks_malformed_change_limits(self) -> None:
        contract = self.localized_contract()
        contract["verification"]["min_inside_changed_pixels"] = "one"

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "MIN_INSIDE_CHANGED_PIXELS_INVALID",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_preflight_blocks_invalid_reference_canvas_size(self) -> None:
        contract = self.localized_contract()
        contract["reference_canvas_px"] = "1536"

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "REFERENCE_CANVAS_SIZE_INVALID",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_reference_pack_cannot_write_outside_workspace(self) -> None:
        contract = self.localized_contract()
        contract_path = self.workspace / "contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        with tempfile.TemporaryDirectory() as outside_dir:
            with self.assertRaises(fidelity_gate.GateError) as raised:
                fidelity_gate.optimize_references(
                    contract,
                    contract_path,
                    self.workspace,
                    Path(outside_dir) / "pack",
                )

        self.assertEqual("OUT_DIR_OUTSIDE_WORKSPACE", raised.exception.code)

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

    def test_bbox_right_and_bottom_edges_remain_locked(self) -> None:
        contract = self.localized_contract()
        candidate = self.workspace / "candidate.png"
        image = Image.open(self.target).convert("RGBA")
        image.putpixel((20, 20), (255, 0, 0, 255))
        image.putpixel((48, 48), (255, 0, 0, 255))
        image.save(candidate)

        receipt = fidelity_gate.verify_localized_output(contract, self.workspace, candidate)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual(1, receipt["outside_changed_pixels"])

    def test_localized_preflight_blocks_invalid_mask_feather(self) -> None:
        contract = self.localized_contract()
        contract["edit_region"]["feather_px"] = "soft"

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "EDIT_REGION_FEATHER_INVALID",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_feathered_provider_mask_does_not_expand_verification_authority(
        self,
    ) -> None:
        contract = self.localized_contract()
        contract["edit_region"]["feather_px"] = 2
        candidate = self.workspace / "candidate.png"
        image = Image.open(self.target).convert("RGBA")
        image.putpixel((20, 20), (255, 0, 0, 255))
        image.putpixel((15, 15), (255, 0, 0, 255))
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

    def test_localized_cli_verify_requires_semantic_review_receipt(self) -> None:
        contract = self.localized_contract()
        contract_path = self.workspace / "localized-contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        candidate = self.workspace / "candidate.png"
        image = Image.open(self.target).convert("RGBA")
        image.putpixel((20, 20), (255, 0, 0, 255))
        image.save(candidate)

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
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(2, completed.returncode, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual("SEMANTIC_REVIEW_REQUIRED", receipt["code"])

    def test_localized_cli_verify_blocks_stale_candidate_review(self) -> None:
        contract = self.localized_contract()
        contract_path = self.workspace / "localized-contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        candidate = self.workspace / "candidate.png"
        image = Image.open(self.target).convert("RGBA")
        image.putpixel((20, 20), (255, 0, 0, 255))
        image.save(candidate)
        review = self.localized_review(contract, candidate, candidate_sha256="0" * 64)

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

        self.assertEqual(2, completed.returncode, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn("REVIEW_CANDIDATE_HASH_MISMATCH", receipt["findings"])

    def test_localized_cli_verify_passes_with_bound_independent_review(self) -> None:
        contract = self.localized_contract()
        contract_path = self.workspace / "localized-contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        candidate = self.workspace / "candidate.png"
        image = Image.open(self.target).convert("RGBA")
        image.putpixel((20, 20), (255, 0, 0, 255))
        image.save(candidate)
        review = self.localized_review(contract, candidate)

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
        self.assertEqual("PASS_PIXEL_AND_SEMANTIC_REVIEW", receipt["code"])
        self.assertEqual("SEMANTIC_REVIEW_PASS", receipt["semantic_review"]["code"])
        self.assertFalse(receipt["wiring_allowed"])

    def test_localized_review_blocks_normalized_builder_identity(self) -> None:
        contract = self.localized_contract()
        candidate = self.workspace / "candidate.png"
        image = Image.open(self.target).convert("RGBA")
        image.putpixel((20, 20), (255, 0, 0, 255))
        image.save(candidate)
        review = self.localized_review(contract, candidate, reviewer=" Image-Builder ")

        receipt = fidelity_gate.verify_semantic_review(contract, self.workspace, candidate, review)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn("BUILDER_SELF_REVIEW_FORBIDDEN", receipt["findings"])

    def test_localized_review_requires_exact_passing_check_set(self) -> None:
        contract = self.localized_contract()
        candidate = self.workspace / "candidate.png"
        image = Image.open(self.target).convert("RGBA")
        image.putpixel((20, 20), (255, 0, 0, 255))
        image.save(candidate)
        valid_checks = dict.fromkeys(contract["semantic_review"]["required_checks"], "PASS")
        variants = {
            "missing": {key: value for key, value in valid_checks.items() if key != "placement"},
            "failed": {**valid_checks, "placement": "FAIL"},
            "extra": {**valid_checks, "uncontracted_check": "PASS"},
        }
        for label, checks in variants.items():
            with self.subTest(label=label):
                review = self.localized_review(contract, candidate, checks=checks)
                receipt = fidelity_gate.verify_semantic_review(
                    contract, self.workspace, candidate, review
                )
                self.assertEqual("BLOCKED", receipt["status"])
                self.assertIn("SEMANTIC_CHECKS_NOT_PASSING", receipt["findings"])

    def test_localized_review_requires_exact_expected_text(self) -> None:
        contract = self.localized_contract()
        contract["references"][0]["verbatim_text"] = ["BLACK IS BEAUTIFUL"]
        contract["semantic_review"]["required_checks"].append("verbatim_text")
        candidate = self.workspace / "candidate.png"
        image = Image.open(self.target).convert("RGBA")
        image.putpixel((20, 20), (255, 0, 0, 255))
        image.save(candidate)

        for label, expected_text in (("missing", []), ("wrong", ["BLACK IS GREAT"])):
            with self.subTest(label=label):
                review = self.localized_review(
                    contract, candidate, expected_verbatim_text=expected_text
                )
                receipt = fidelity_gate.verify_semantic_review(
                    contract, self.workspace, candidate, review
                )
                self.assertEqual("BLOCKED", receipt["status"])
                self.assertIn("VERBATIM_TEXT_REVIEW_MISMATCH", receipt["findings"])

    def test_localized_verifier_blocks_baseline_changed_after_contract(self) -> None:
        contract = self.localized_contract()
        candidate = self.workspace / "candidate.png"
        save_rgb(candidate, (255, 0, 0))
        save_rgb(self.target, (12, 12, 12))

        receipt = fidelity_gate.verify_localized_output(contract, self.workspace, candidate)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual("SOURCE_HASH_DRIFT", receipt["code"])

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

    def test_filename_view_conflict_is_blocked_until_source_is_renamed(self) -> None:
        contract = self.localized_contract()
        conflicting = self.workspace / "br-007-back-source.jpg"
        self.reference.rename(conflicting)
        contract["references"][0]["path"] = conflicting.name
        contract["references"][0]["sha256"] = fidelity_gate.sha256_file(conflicting)
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn("FILENAME_VIEW_CONFLICT", {item["code"] for item in receipt["findings"]})

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

    def test_protected_layer_without_opaque_pixels_is_blocked(self) -> None:
        contract = self.protected_contract()
        protected = self.workspace / contract["references"][0]["path"]
        Image.new("RGBA", (32, 32), (10, 20, 30, 128)).save(protected)
        contract["references"][0]["sha256"] = fidelity_gate.sha256_file(protected)

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "PROTECTED_LAYER_OPAQUE_PIXELS_REQUIRED",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_protected_verify_cannot_vacuously_pass_without_opaque_pixels(self) -> None:
        contract = self.protected_contract()
        protected = self.workspace / contract["references"][0]["path"]
        Image.new("RGBA", (32, 32), (10, 20, 30, 128)).save(protected)
        contract["references"][0]["sha256"] = fidelity_gate.sha256_file(protected)
        candidate = self.workspace / "candidate.png"
        save_rgb(candidate, (255, 0, 0))

        receipt = fidelity_gate.verify_protected_output(contract, self.workspace, candidate)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual("PROTECTED_LAYER_OPAQUE_PIXELS_REQUIRED", receipt["code"])

    def test_protected_preflight_rejects_unbound_placement_layer(self) -> None:
        contract = self.protected_contract()
        rogue = self.workspace / "rogue.png"
        save_protected(rogue, real_alpha=True)
        contract["protected_placements"] = [{"path": rogue.name, "x": 16, "y": 16}]

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "PROTECTED_PLACEMENT_REFERENCE_MISMATCH",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_cli_verify_rejects_output_outside_workspace(self) -> None:
        contract = self.protected_contract()
        contract_path = self.workspace / "protected-contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        with tempfile.TemporaryDirectory() as outside_dir:
            candidate = self.protected_candidate(
                contract, Path(outside_dir) / "outside-candidate.png"
            )
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
                ],
                check=False,
                capture_output=True,
                text=True,
            )

        self.assertEqual(2, completed.returncode, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual("OUTPUT_OUTSIDE_WORKSPACE", receipt["code"])

    def test_cli_verify_returns_blocked_receipt_for_unreadable_output(self) -> None:
        contract = self.protected_contract()
        contract_path = self.workspace / "protected-contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        candidate = self.workspace / "candidate.png"
        candidate.write_text("not an image", encoding="utf-8")

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
            ],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(2, completed.returncode, completed.stderr)
        receipt = json.loads(completed.stdout)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual("OUTPUT_UNREADABLE", receipt["code"])

    def test_protected_verify_blocks_output_dimension_drift(self) -> None:
        contract = self.protected_contract()
        candidate = self.workspace / "oversized-candidate.png"
        image = Image.new("RGBA", (80, 80), (15, 15, 15, 255))
        placement = contract["protected_placements"][0]
        with Image.open(self.workspace / placement["path"]) as opened:
            layer = opened.convert("RGBA")
        image.alpha_composite(layer, (placement["x"], placement["y"]))
        image.convert("RGB").save(candidate)

        receipt = fidelity_gate.verify_protected_output(contract, self.workspace, candidate)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual("PROTECTED_OUTPUT_DIMENSION_DRIFT", receipt["code"])

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
                    "verbatim_text": [],
                }
            ],
        }
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "POSE_CHANGE_WITHOUT_APPROVED_SOURCE",
            {finding["code"] for finding in receipt["findings"]},
        )

    def test_native_preflight_requires_boolean_pose_change(self) -> None:
        for invalid in ("yes", 1):
            with self.subTest(invalid=invalid):
                contract = self.native_contract()
                contract["pose_change"] = invalid

                receipt = fidelity_gate.validate_contract(contract, self.workspace)

                self.assertEqual("BLOCKED", receipt["status"])
                self.assertIn(
                    "POSE_CHANGE_FLAG_INVALID",
                    {finding["code"] for finding in receipt["findings"]},
                )

    def test_native_scene_requires_optical_contract(self) -> None:
        contract = self.native_contract()
        del contract["optical_contract"]
        receipt = fidelity_gate.validate_contract(contract, self.workspace)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn("OPTICAL_CONTRACT_REQUIRED", {item["code"] for item in receipt["findings"]})

    def test_native_review_thresholds_cannot_drop_below_policy_floors(self) -> None:
        contract = self.native_contract()
        contract["review_gate"]["minimum_scores"] = dict.fromkeys(
            fidelity_gate.NATIVE_SCORE_DIMENSIONS, 0
        )

        receipt = fidelity_gate.validate_contract(contract, self.workspace)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn(
            "MINIMUM_SCORE_BELOW_POLICY_FLOOR",
            {finding["code"] for finding in receipt["findings"]},
        )

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
        review = self.native_review(contract, candidate, scores=scores)
        receipt = fidelity_gate.verify_native_output(contract, self.workspace, candidate, review)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn("SCORE_BELOW_MINIMUM:optical_integration", receipt["findings"])

    def test_native_scene_review_cannot_replay_after_contract_change(self) -> None:
        contract = self.native_contract()
        candidate = self.native_candidate(contract)
        review = self.native_review(contract, candidate)
        contract["collection_scene"]["scene_variation"] = "changed-after-review"

        receipt = fidelity_gate.verify_native_output(contract, self.workspace, candidate, review)

        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn("REVIEW_CONTRACT_HASH_MISMATCH", receipt["findings"])

    def test_native_scene_builder_self_review_is_blocked(self) -> None:
        contract = self.native_contract()
        candidate = self.native_candidate(contract)
        review = self.native_review(contract, candidate, reviewer="scene-builder")
        receipt = fidelity_gate.verify_native_output(contract, self.workspace, candidate, review)
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertIn("BUILDER_SELF_REVIEW_FORBIDDEN", receipt["findings"])

    def test_native_scene_passes_as_unwired_review_candidate(self) -> None:
        contract = self.native_contract()
        candidate = self.native_candidate(contract)
        review = self.native_review(contract, candidate)
        receipt = fidelity_gate.verify_native_output(contract, self.workspace, candidate, review)
        self.assertEqual("PASS", receipt["status"])
        self.assertEqual("PASS_REVIEW_CANDIDATE", receipt["code"])
        self.assertFalse(receipt["wiring_allowed"])
        self.assertEqual("FOUNDER_REVIEW_REQUIRED", receipt["promotion_state"])
        self.assertFalse(receipt["review_identity_verified"])

    def test_self_declared_founder_status_never_becomes_verified_approval(self) -> None:
        contract = self.native_contract()
        candidate = self.native_candidate(contract)
        review = self.native_review(contract, candidate, founder_status="APPROVED")

        receipt = fidelity_gate.verify_native_output(contract, self.workspace, candidate, review)

        self.assertEqual("PASS", receipt["status"])
        self.assertEqual("PASS_REVIEW_CANDIDATE_WITH_UNVERIFIED_FOUNDER_CLAIM", receipt["code"])
        self.assertFalse(receipt["founder_approval_verified"])
        self.assertEqual("FOUNDER_APPROVAL_ATTESTATION_REQUIRED", receipt["promotion_state"])

    def test_native_scene_cli_verify_review_round_trip(self) -> None:
        contract = self.native_contract()
        contract_path = self.workspace / "native-contract.json"
        contract_path.write_text(json.dumps(contract), encoding="utf-8")
        candidate = self.native_candidate(contract)
        review = self.native_review(contract, candidate)
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
