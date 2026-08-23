from __future__ import annotations

from scripts.launch import woocommerce_product_contract as contract_builder

PRODUCT_FIELDS = {
    "name",
    "slug",
    "type",
    "status",
    "featured",
    "catalog_visibility",
    "description",
    "short_description",
    "sku",
    "regular_price",
    "sale_price",
    "date_on_sale_from",
    "date_on_sale_to",
    "virtual",
    "downloadable",
    "downloads",
    "download_limit",
    "download_expiry",
    "external_url",
    "button_text",
    "tax_status",
    "tax_class",
    "manage_stock",
    "stock_quantity",
    "stock_status",
    "backorders",
    "sold_individually",
    "weight",
    "dimensions",
    "shipping_class",
    "reviews_allowed",
    "upsell_ids",
    "cross_sell_ids",
    "purchase_note",
    "categories",
    "tags",
    "images",
    "attributes",
    "default_attributes",
    "menu_order",
    "meta_data",
}


def _by_sku() -> dict[str, dict]:
    return {item["sku"]: item for item in contract_builder.build_contract()["products"]}


def test_contract_has_all_products_and_writable_fields() -> None:
    products = contract_builder.build_contract()["products"]
    assert len(products) == 33
    for record in products:
        assert set(record["product"]) == PRODUCT_FIELDS, record["sku"]
        assert record["product_sot_hash"]
        assert record["image_roles"]


def test_sized_product_is_variable_with_one_variation_per_size() -> None:
    record = _by_sku()["br-009"]
    assert record["product"]["type"] == "variable"
    size_options = record["product"]["attributes"][0]["options"]
    assert len(record["variations"]) == len(size_options)
    assert all(item["regular_price"] == "115" for item in record["variations"])


def test_one_size_product_remains_simple() -> None:
    record = _by_sku()["lh-005"]
    assert record["product"]["type"] == "simple"
    assert record["variations"] == []


def test_preorder_meta_uses_theme_keys_and_preserves_available_count() -> None:
    record = _by_sku()["br-009"]
    meta = {item["key"]: item["value"] for item in record["product"]["meta_data"]}
    create_only = {item["key"]: item["value"] for item in record["create_only_meta_data"]}
    assert meta["_is_preorder"] == "1"
    assert meta["_preorder_edition_size"] == "80"
    assert "_preorder_available" not in meta
    assert create_only == {"_preorder_available": "80"}
    assert "_skyyrose_preorder" not in meta


def test_jersey_series_meta_is_projected_without_name_or_sku_inference() -> None:
    jersey = _by_sku()["br-009"]
    jersey_meta = {
        item["key"]: item["value"] for item in jersey["product"]["meta_data"]
    }
    assert jersey_meta["_skyyrose_series_slug"] == "jersey-series"
    assert jersey_meta["_skyyrose_series_region"] == "oakland"
    assert jersey_meta["_skyyrose_series_order"] == 30

    non_series = _by_sku()["br-006"]
    non_series_meta = {
        item["key"]: item["value"] for item in non_series["product"]["meta_data"]
    }
    assert non_series_meta["_skyyrose_series_slug"] == ""
    assert non_series_meta["_skyyrose_series_region"] == ""
    assert non_series_meta["_skyyrose_series_order"] == 0


def test_contract_file_is_current() -> None:
    assert contract_builder.OUTPUT_PATH.read_text(encoding="utf-8") == (
        contract_builder.serialize_contract()
    )
