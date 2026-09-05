"""Read a pinned upstream artifact; never regenerate or author product truth here."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONTRACT = Path(__file__).with_name('build-inputs.json')
SOT = Path(__file__).with_name('inputs') / 'product-sot.json'


def load_product_sot():
    contract = json.loads(CONTRACT.read_text())
    raw = SOT.read_bytes()
    if hashlib.sha256(raw).hexdigest() != contract['product_sot_sha256']:
        raise ValueError('Pinned upstream product SOT artifact changed')
    manifest = json.loads(raw)
    if manifest['sources']['catalog_sha256'] != contract['catalog_sha256']:
        raise ValueError('Upstream catalog identity changed')
    for sku, product in manifest['products'].items():
        payload = {k: v for k, v in product.items() if k != 'product_hash'}
        encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
        if hashlib.sha256(encoded).hexdigest() != product['product_hash']:
            raise ValueError(f'Invalid upstream product hash: {sku}')
    return manifest, raw.decode('utf-8')


def garment_types(manifest):
    """CSV classification is explicit; empty accessory classifications stay empty.

    Check shared commerce fields before combining the current authoring CSV with
    a versioned editorial/media artifact. Updating either is a separate review.
    """
    contract = json.loads(CONTRACT.read_text())
    with (ROOT / contract['catalog_authority']).open(newline='', encoding='utf-8') as source:
        rows = list(csv.DictReader(source))
    by_sku = {r['sku'].strip(): r for r in rows}
    if len(by_sku) != len(rows) or set(by_sku) != set(manifest['products']):
        raise ValueError('Catalog SKU identity differs from upstream artifact')
    for sku, product in manifest['products'].items():
        row = by_sku[sku]
        commerce = product['commerce']
        expected = {
            'price': row['price'].strip(),
            'sizes': [s.strip() for s in row['sizes'].split('|') if s.strip()],
            'colors': [s.strip() for s in row['color'].split('|') if s.strip()],
            'edition_size': row['edition_size'].strip(),
            'published': row['published'].strip() == '1',
            'is_preorder': row['is_preorder'].strip() == '1',
        }
        if commerce != expected or product['identity']['collection'] != row['collection'].strip():
            raise ValueError(f'Catalog commerce/collection disagreement: {sku}')
    return {sku: row['garment_type_lock'].strip().lower() for sku, row in by_sku.items()}
