#!/usr/bin/env python3
"""Preview or apply the canonical product SOT to WooCommerce.

Dry-run is the default and performs no network requests. ``--apply`` is an
explicit production-write gate. Product payloads come only from the generated
Woo contract, which itself is built from ``data/product-sot.json``.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import httpx

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.launch import woocommerce_product_contract  # noqa: E402

IMAGE_MAP_JSON = ROOT / "scripts" / "launch" / "sku_image_map.json"
DEFAULT_SITE_URL = "https://skyyrose.co"


def load_env() -> None:
    """Load simple KEY=VALUE entries from the repository .env file."""
    env_path = ROOT / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, value = stripped.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


class WooClient:
    def __init__(self, site_url: str, key: str, secret: str) -> None:
        self.base_url = f"{site_url.rstrip('/')}/wp-json/wc/v3"
        self.auth = (key, secret)
        self.client = httpx.Client(timeout=30.0)

    def close(self) -> None:
        self.client.close()

    def request(
        self,
        method: str,
        endpoint: str,
        *,
        json_data: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any] | list[dict[str, Any]]:
        response = self.client.request(
            method,
            f"{self.base_url}/{endpoint}",
            auth=self.auth,
            json=json_data,
            params=params,
        )
        response.raise_for_status()
        return response.json()

    def paged(self, endpoint: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        page = 1
        while True:
            batch = self.request("GET", endpoint, params={"per_page": 100, "page": page})
            assert isinstance(batch, list)
            if not batch:
                return rows
            rows.extend(batch)
            page += 1


def _load_image_map() -> dict[str, dict[str, Any]]:
    if not IMAGE_MAP_JSON.is_file():
        return {}
    return json.loads(IMAGE_MAP_JSON.read_text(encoding="utf-8"))


def _category_ids(client: WooClient, contract: dict[str, Any]) -> dict[str, int]:
    existing = {item["slug"]: item for item in client.paged("products/categories")}
    result: dict[str, int] = {}
    for slug, name in contract["category_names"].items():
        if slug not in existing:
            created = client.request(
                "POST",
                "products/categories",
                json_data={"name": name, "slug": slug},
            )
            assert isinstance(created, dict)
            existing[slug] = created
            print(f"CREATED category {slug} ({created['id']})")
        result[slug] = int(existing[slug]["id"])
    return result


def _product_payload(
    record: dict[str, Any],
    category_ids: dict[str, int],
    image_map: dict[str, dict[str, Any]],
    *,
    creating: bool,
) -> dict[str, Any]:
    payload = copy.deepcopy(record["product"])
    payload["categories"] = [{"id": category_ids[slug]} for slug in record["category_slugs"]]
    image = image_map.get(record["sku"])
    if image and image.get("image_id"):
        payload["images"] = [{"id": int(image["image_id"])}]
    if creating:
        payload["meta_data"].extend(copy.deepcopy(record["create_only_meta_data"]))
    return payload


def _sync_variations(
    client: WooClient,
    product_id: int,
    variations: list[dict[str, Any]],
    delay: float,
) -> tuple[int, int]:
    existing = {
        item.get("sku"): item
        for item in client.paged(f"products/{product_id}/variations")
        if item.get("sku")
    }
    created = 0
    updated = 0
    for payload in variations:
        sku = payload["sku"]
        if sku in existing:
            client.request(
                "PUT",
                f"products/{product_id}/variations/{existing[sku]['id']}",
                json_data=payload,
            )
            updated += 1
        else:
            client.request(
                "POST",
                f"products/{product_id}/variations",
                json_data=payload,
            )
            created += 1
        time.sleep(delay)
    return created, updated


def preview(contract: dict[str, Any], image_map: dict[str, dict[str, Any]]) -> int:
    print("DRY RUN — no WooCommerce network requests or writes")
    print(f"Products: {len(contract['products'])}")
    print(f"Image mappings: {len(image_map)}")
    for record in contract["products"]:
        product = record["product"]
        print(
            f"{record['sku']:8} {product['type']:8} {product['status']:7} "
            f"variations={len(record['variations']):2} media={len(record['image_roles']):2} "
            f"sot={record['product_sot_hash'][:12]}"
        )
    return 0


def apply(contract: dict[str, Any], image_map: dict[str, dict[str, Any]], args: argparse.Namespace) -> int:
    load_env()
    key = os.getenv("WOOCOMMERCE_KEY", "")
    secret = os.getenv("WOOCOMMERCE_SECRET", "")
    if not key or not secret:
        print("ERROR: WOOCOMMERCE_KEY and WOOCOMMERCE_SECRET are required", file=sys.stderr)
        return 2

    client = WooClient(args.site_url, key, secret)
    failures = 0
    try:
        category_ids = _category_ids(client, contract)
        existing = {
            item["sku"]: item for item in client.paged("products") if item.get("sku")
        }
        for record in contract["products"]:
            sku = record["sku"]
            creating = sku not in existing
            payload = _product_payload(record, category_ids, image_map, creating=creating)
            try:
                if creating:
                    result = client.request("POST", "products", json_data=payload)
                    action = "CREATED"
                else:
                    result = client.request(
                        "PUT",
                        f"products/{existing[sku]['id']}",
                        json_data=payload,
                    )
                    action = "UPDATED"
                assert isinstance(result, dict)
                variation_created, variation_updated = _sync_variations(
                    client,
                    int(result["id"]),
                    record["variations"],
                    args.delay,
                )
                print(
                    f"{action:7} {sku:8} product={result['id']} "
                    f"variations=+{variation_created}/~{variation_updated}"
                )
                time.sleep(args.delay)
            except httpx.HTTPError as exc:
                failures += 1
                print(f"FAILED  {sku:8} {exc}", file=sys.stderr)
    finally:
        client.close()
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true", help="Preview only; this is the default")
    mode.add_argument("--apply", action="store_true", help="Write to production WooCommerce")
    parser.add_argument(
        "--site-url",
        default=os.getenv("WORDPRESS_URL", DEFAULT_SITE_URL),
        help="WooCommerce site URL",
    )
    parser.add_argument("--delay", type=float, default=0.5, help="Delay between writes")
    args = parser.parse_args(argv)

    contract = woocommerce_product_contract.build_contract()
    expected = woocommerce_product_contract.serialize_contract()
    output = woocommerce_product_contract.OUTPUT_PATH
    if not output.is_file() or output.read_text(encoding="utf-8") != expected:
        print(f"ERROR: stale Woo contract; regenerate {output}", file=sys.stderr)
        return 2
    image_map = _load_image_map()
    if not args.apply:
        return preview(contract, image_map)
    return apply(contract, image_map, args)


if __name__ == "__main__":
    raise SystemExit(main())
