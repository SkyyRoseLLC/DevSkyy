from __future__ import annotations

from skyyrose.core.product_sot_support import SUPPORT_AREAS, build_support_report, strict_blockers


def test_support_report_covers_every_product_and_domain() -> None:
    report = build_support_report()
    assert report["product_sot"]["sku_count"] == 33
    assert len(report["products"]) == 33
    assert set(report["support_areas"]) == set(SUPPORT_AREAS)
    for product in report["products"]:
        assert set(product["areas"]) == set(SUPPORT_AREAS), product["sku"]


def test_support_report_is_honest_about_open_requirements() -> None:
    report = build_support_report()
    blockers = strict_blockers(report)
    assert blockers
    assert any(blocker.endswith("reconstruction_capture:BLOCKED_CAPTURE_RECEIPT_REQUIRED") for blocker in blockers)
