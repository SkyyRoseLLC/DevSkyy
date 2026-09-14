from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import model_registry


class ModelRegistryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.registry = model_registry.load_registry(model_registry.DEFAULT_REGISTRY)

    def test_unknown_model_fails_closed(self) -> None:
        receipt = model_registry.validate_request(
            self.registry, "future-magic-model", "bounded_image_edit"
        )
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual("UNKNOWN_MODEL", receipt["code"])

    def test_codex_wrapper_cannot_run_bounded_edit(self) -> None:
        receipt = model_registry.validate_request(
            self.registry, "image_gen.imagegen", "bounded_image_edit"
        )
        self.assertEqual("BLOCKED", receipt["status"])

    def test_openai_image_api_can_enter_bounded_gate_with_pinned_contract(self) -> None:
        receipt = model_registry.validate_request(
            self.registry, "gpt-image-2-2026-04-21", "bounded_image_edit"
        )
        self.assertEqual("PASS", receipt["status"])
        self.assertTrue(receipt["post_verification_required"])
        self.assertEqual("/v1/images/edits", receipt["api_contract"]["endpoint"])
        self.assertIn("input_fidelity", receipt["api_contract"]["forbidden_parameters"])

    def test_openai_responses_alias_cannot_run_one_shot_bounded_edit(self) -> None:
        receipt = model_registry.validate_request(
            self.registry, "openai-responses-image-mask", "bounded_image_edit"
        )
        self.assertEqual("BLOCKED", receipt["status"])
        self.assertEqual("OPERATION_NOT_ALLOWED", receipt["code"])

    def test_invalid_openai_mask_byte_limit_fails_registry_load(self) -> None:
        registry = json.loads(json.dumps(self.registry))
        openai_edit = next(
            model for model in registry["models"] if model["id"] == "openai-gpt-image-2-edit"
        )
        openai_edit["api_contract"]["mask"]["max_input_bytes"] = True
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "registry.json"
            path.write_text(json.dumps(registry), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "max_input_bytes must be an integer"):
                model_registry.load_registry(path)

    def test_registry_object_and_audit_hash_share_one_byte_snapshot(self) -> None:
        payload = model_registry.DEFAULT_REGISTRY.read_bytes()

        registry, digest = model_registry.load_registry_snapshot(model_registry.DEFAULT_REGISTRY)

        self.assertEqual("product-fidelity-model-registry.v2", registry["schema"])
        self.assertEqual(hashlib.sha256(payload).hexdigest(), digest)

    def test_generative_video_cannot_claim_exact_product(self) -> None:
        receipt = model_registry.validate_request(
            self.registry, "sora-2-pro", "exact_product_video"
        )
        self.assertEqual("BLOCKED", receipt["status"])

    def test_deterministic_video_route_can_enter_exact_gate(self) -> None:
        receipt = model_registry.validate_request(
            self.registry, "ffmpeg-composite", "exact_product_video"
        )
        self.assertEqual("PASS", receipt["status"])

    def test_single_image_3d_cannot_claim_exact_replica(self) -> None:
        receipt = model_registry.validate_request(
            self.registry, "microsoft/TRELLIS.2-4B", "exact_3d_replica"
        )
        self.assertEqual("BLOCKED", receipt["status"])

    def test_deterministic_3d_can_enter_exact_gate(self) -> None:
        receipt = model_registry.validate_request(self.registry, "clo3d", "exact_3d_replica")
        self.assertEqual("PASS", receipt["status"])


if __name__ == "__main__":
    unittest.main()
