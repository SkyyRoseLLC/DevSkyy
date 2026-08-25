# Catalog-Wide Product SOT Support Contract

The canonical product SOT is deliberately stronger than a catalogue listing: it resolves each SKU from the source catalogue, dossier, logo registry, media paths, and SHA-256 hashes. This companion gate asks a separate question: **does every published product have the evidence needed for each claimed downstream use?**

It audits all 33 SKUs against the current `data/product-sot.json` hash and its V2 opening-media approval manifest. It does not create, approve, upload, wire, or deploy anything.

| Support area | Ready requires | Current promotion boundary |
| --- | --- | --- |
| Catalog identity | Exact SKU, identity, dossier and registered artifacts hash-verify. | A valid listing is not product-fidelity approval. |
| Commerce catalog | Price, sizes, colors, publication and preorder semantics exist. | WooCommerce remains live authority for price, stock, variations, and permalink. |
| Storefront media | A current, hash-bound approved on-model front is present. | Card approval does not provide reconstruction or native-scene authority. |
| Exact product authority | A verified physical photo/video or approved design reference is registered. | Ghost/mannequin, generated, placeholder, and rejected material never qualifies. |
| Reconstruction capture | Multi-view calibrated physical evidence and a hash-bound receipt exist. | A capture pass still needs independent reconstruction/product-fidelity review. |
| Measured fit / fulfillment | Measurements, fit/model reference, material/care, package data, and policy rule are canonical. | Do not infer these from a product image or generic policy page. |
| Rights / promotion | Operator/owner, rights restriction, and needed founder promotion authorization are recorded. | Founder, product-fidelity, native-integration, wiring, and deployment remain independent gates. |

Run:

```bash
python3 scripts/validate-product-sot-support.py
python3 scripts/validate-product-sot-support.py --strict
```

The normal command validates and summarizes the whole support matrix. `--strict` must remain red until every published SKU has evidence in every area. It is intentionally not weakened by existing packshots, ghost assets, candidates, legacy previews, or previous approval statuses.
