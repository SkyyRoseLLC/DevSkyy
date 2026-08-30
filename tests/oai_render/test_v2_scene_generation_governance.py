"""Paid legacy scene generators must fail before they reach a provider."""

from __future__ import annotations

from scripts.oai_render import scene_gen, scene_preview


def test_scene_gen_refuses_unbound_environment_generation_before_spend() -> None:
    assert scene_gen.main(["generate", "--yes"]) == 4


def test_scene_preview_refuses_unbound_environment_generation_before_spend() -> None:
    assert scene_preview.main(["generate", "--collection", "love-hurts", "--yes"]) == 4
