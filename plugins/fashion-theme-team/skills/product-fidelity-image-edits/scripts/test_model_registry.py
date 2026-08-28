from __future__ import annotations

import unittest

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

    def test_openai_mask_adapter_can_enter_bounded_gate(self) -> None:
        receipt = model_registry.validate_request(
            self.registry, "openai-responses-image-mask", "bounded_image_edit"
        )
        self.assertEqual("PASS", receipt["status"])
        self.assertTrue(receipt["post_verification_required"])

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
