# V2 catalog, media, and Scroll World binding audit

**Scope:** P1 discovery snapshot. No catalog, WooCommerce, or production-theme mutation was performed by this report. The original observations remain useful, but the prototype path below is archival.

**Historical candidate boundary:** worktree `/Users/theceo/DevSkyy-skyyrose-flagship-v2-prototype`, branch `codex/skyyrose-flagship-v2-prototype`, Git baseline `0074fc6e6b161266f9181303296941cd474208af`. Its preserved capture is `1982efe66`; it is not an active worktree.

**Active candidate boundary:** worktree `/Users/theceo/DevSkyy`, branch `codex/skyyrose-v2-marketplace`, source snapshot `a620435d646d89d3f99840350d3151a25dcbec47`, payload `d249a687c4dd1f9f053aa8dd5f66e8717f72f8a4`, and attestation `8b4be49d0a2642212b088974e4eca66560006f29`. The active candidate is tracked and has package/source evidence; media rights, SKU bindings, and independent runtime review remain blocked. The branch is the live ref; documentation-only reconciliation commits are not source mutations.

**P1 verdict: BLOCKED.** P1 requires a SKU -> WooCommerce product -> variation -> approved product media -> rights/policy record mapping for every purchasable CTA and scenic product link. V2 has no such artifact, V2 cards resolve WooCommerce attachments rather than `sot-images.json`, and its scenes do not carry registered provenance or object-level product association. Unknown media, rights, product identity, product availability, and variation states must fail closed.

**P1 ledger implementation:** [catalog-binding.json](../../../../.fashion-theme/catalog-binding.json) and [shot-manifest.json](../../../../.fashion-theme/shot-manifest.json) now preserve the observed source facts and model every unresolved relationship as `BLOCKED` or `UNKNOWN`. They are evidence contracts, not approval or production-promotion artifacts.

## Authorities and observed evidence

| Concern | Canonical authority | Observed state | Status |
| --- | --- | --- | --- |
| SKU, product facts, structured availability flag, size/color | `wordpress-theme/skyyrose-flagship/data/skyyrose-catalog.csv` | 33 SKU records; `published`, `is_preorder`, sizes, color, and a dossier slug are present. | OBSERVED |
| Garment canon | `wordpress-theme/skyyrose-flagship/data/dossiers/<name>.md` | Catalog carries dossier slugs; this audit did not infer a scene/product relationship from prose. | OBSERVED |
| Product images | generated `data/sot-images.json` | Front-first contract has entries for all 33 catalog SKUs. V2 does not read it. | OBSERVED / UNBOUND |
| Non-product imagery identity | `wordpress-theme/skyyrose-flagship/data/visual-manifest.json` | Relevant source records are `status: "verified"`; that status does not bind an unregistered V2 copy, a rights record, or an intended product association. | OBSERVED / UNBOUND |
| Runtime price, stock, purchasability, variation ID | WooCommerce server state | V2 gets product cards through `wc_get_products()` and image/price/stock from each WC product; no current WooCommerce snapshot was supplied or queried. | UNKNOWN |
| V2 candidate media identity / rights / hash | candidate-local manifest (required, absent) | No `catalog-binding.json`, `shot-manifest.json`, rights record, candidate manifest, or V2 visual manifest exists. | MISSING |

### Source hashes captured for this audit

| Source | SHA-256 |
| --- | --- |
| `SOT.md` | `0654d496482b953f43c3d34c063f83863b54723466be2c593ccdc4b1dd8fba94` |
| `data/sot-images.json` | `829dd820fc570a7da80113ec4f68f0e5a30092e73f96bc83fb2e3d040256cb6d` |
| V1 `data/visual-manifest.json` | `c27517ffb06b86381481d257b15ff348466009effb56fc1ce41475e3e77e81f7` |
| canonical catalog CSV | `ded09dd3cc62309e28b8ab6861131099601aa7898c46b632397f9488956d3ccf` |
| V2 `functions.php` | `b14c3e9242ffc26135b6a5b71ce149ebc227b44371ee2b0107bce06cb783b290` |
| V2 `front-page.php` | `40f50316a8f838ecdfbaa3befe071673a9d416b81f1a317227aeb10baa8c313e` |
| V2 `page.php` | `9cd15bd716551e79ab14715ae2be5663e818fd963568373e1395e5258860b440` |
| V2 `template-collection.php` | `4a091c550d9a6d1258f540c3050aaa387b7601088d9f2e5ac7424bc0637b75ca` |
| V2 `woocommerce/single-product.php` | `65e680fb15c1bcc3340cdc272b45dd3f78b47f4b212bb6e2ecacf54c9ad375cd` |

## Existing canonical media that can support the required collection surfaces

This table is an evidence inventory, not authorization to use the assets. “Supports” means that a V1 visual-manifest record exists and is `verified`; V2 must still bind the exact candidate byte/hash, rights/policy record, intended route/role, and fallback before use.

| Collection | Canonical V1 visual-manifest records observed | V2 packaged reference | V2 route/surface | Binding assessment |
| --- | --- | --- | --- | --- |
| Signature | `branding/hero/signature-golden-gate-yacht` (cinematic backdrop); `images/immersive/scene-signature-golden-gate` (scene art); `images/lockups/signature-lockup`; `images/logos/rose-gold-rose` | `assets/sot/branding/hero/signature-golden-gate-yacht-1280w.webp` is present; rail uses `assets/scroll-world/scene-1-signature.webp` | collection hero and PDP world panel use the hero; collection story lists `scene-signature-golden-gate` and `scene-signature-oakland-atelier-gpt2`; home rail uses scene 1 | Hero source record is OBSERVED; candidate byte/rights binding is UNKNOWN. Scroll World scene 1 has no visual-manifest identity record. No SKU/variation association exists. |
| Black Rose | `branding/hero/forbidden-midnight` (cinematic backdrop); `images/lockups/black-rose-lockup` | hero path is referenced but **absent**: `assets/sot/branding/hero/forbidden-midnight-1280w.webp`; rail uses `scene-2-black-rose.webp`; V2 has `scene-black-rose-moon-court-gpt2.webp` and the unregistered pre-order salon | collection hero/PDP world panel; home and collection rail; pre-order salon | Canonical hero record is OBSERVED but its required V2 file is MISSING. Rail, moon-court, and salon files lack a matching manifest identity in the inspected records; no scene-object-to-SKU proof exists. |
| Love Hurts | `branding/hero/beauty-and-beast` (cinematic backdrop); `images/immersive/scene-love-hurts-cathedral` (scene art); `images/lockups/love-hurts-lockup`; `images/logos/heart-rose-composite` | `beauty-and-beast-1280w.webp` and `scene-love-hurts-cathedral.webp` are present; rail uses `scene-3-love-hurts.webp`; V2 also contains `scene-love-hurts-cracked-rose-gpt2.webp` | collection hero/PDP world panel; world rail; home collection rail | Hero/cathedral source records are OBSERVED; their candidate byte/rights binding is UNKNOWN. Scroll World scene 3 and cracked-rose candidate file have no observed V1 manifest identity. No SKU/variation association exists. |
| Kids Capsule | `images/immersive/scene-kids-capsule-playroom` and `images/immersive/scene-kids-capsule-runway` (scene art); `images/logos/sr-monogram-rose-gold` | V2 includes playroom/runway assets; rail uses `scene-4-kids-capsule.webp`; V2 additionally contains `scene-kids-capsule-heir-runway-gpt2.webp` | collection hero/PDP world panel; world rail; home collection rail | Playroom/runway source records are OBSERVED; candidate byte/rights binding is UNKNOWN. Scroll World scene 4 and heir-runway candidate file have no observed V1 manifest identity. No SKU/variation association exists. |
| Pre-order (cross-collection gateway) | `branding/hero/luxury-nighttime` (verified global hero); no inspected visual-manifest record for `images/preorder/black-rose-salon` | `luxury-nighttime-1280w.webp` is referenced but **absent**; `black-rose-salon.webp` is present | `/pre-order/` hero and salon; home pre-order section | Hero source is OBSERVED, packaged file is MISSING. Salon identity, rights, and approved role are UNKNOWN; it has no candidate binding or object-to-SKU evidence and cannot prove the garments placed in the scene. |

### Scroll World snapshot inventory

The five candidate-only snapshots have pixels and stable hashes but no inspected visual-manifest record, rights record, route role, or product relationship. They cannot yet support the required high-fidelity story/card/PDP/pre-order scenes as authoritative media.

| V2 snapshot | Dimensions | SHA-256 | Status |
| --- | --- | --- | --- |
| `assets/scroll-world/scene-1-signature.webp` | 1920x1275 | `c0267322fcef1533849d50a04dee499230f0dce0406e509cebaf47c5c199cfdb` | UNKNOWN identity/rights/product association |
| `assets/scroll-world/scene-2-black-rose.webp` | 1920x1275 | `b8fc17624d754136c60949b72de9a51df58f5d4c1fa2d4bc6264de3bba32452d` | UNKNOWN identity/rights/product association |
| `assets/scroll-world/scene-3-love-hurts.webp` | 1920x1275 | `7d926b7854237cdbb882108661a401415cedfd8922e9b9bfd6aacfef105bc70c` | UNKNOWN identity/rights/product association |
| `assets/scroll-world/scene-4-kids-capsule.webp` | 1920x1275 | `b4e25c19d34c6ea21f1c0d1c442c5422273a37aab390c8387dc0c68188b98d78` | UNKNOWN identity/rights/product association |
| `assets/scroll-world/scene-5-finale.webp` | 2048x1360 | `d4f1f5010aa98b173424cec0864d132f772353e29fd87bd2f08553cd9f33a7be` | UNKNOWN identity/rights/product association |

## V2 product-bridge audit

### General product cards and PDP

`skyyrose2_product_cards()` obtains published WC products by category and renders WC attachment images, WC price HTML, and WC stock status. The V2 single-product template delegates the product form/media to WooCommerce and uses a collection-level scenic hero as a PDP aside. This establishes no canonical SKU lookup, no SOT-image resolution, no accepted variation, and no media-rights association. Product card and PDP images are therefore **UNKNOWN** for P1 despite a current WC runtime projection being coded.

`sot-images.json` does contain the canonical product-image contract for all 33 catalog SKUs and is the only observed basis for an eventual SKU-specific card/PDP media selection. It must be resolved by exact SKU and selected variation, never by candidate filename or scene appearance.

### Black Rose pre-order salon

The only explicit V2 scene-product table is in `page.php`. It uses product post slugs that happen to equal five catalog SKUs, and falls back to the Black Rose collection if the WC post cannot be found. No WC state was queried for this audit.

| Scene hotspot code | Catalog SKU/name | Catalog facts relevant to the claim | V2 displayed label/note | P1 result |
| --- | --- | --- | --- | --- |
| `br-001` | `br-001` — BLACK Rose Crewneck | Black Rose; published; not pre-order | “Black Is Beautiful” / “The Oakland statement jersey.” | **CONFLICT / BLOCKED:** product label/type disagrees with catalog. No proof this garment appears at hotspot. |
| `br-003` | `br-003` — BLACK is Beautiful Jersey Series: 0. Baseball Classic (Black) | Black Rose; published; pre-order | “Number 30” / “A numbered piece from the house.” | **UNBOUND:** generic language does not identify a variant; no scene-object, WC product, or variation proof. |
| `br-004` | `br-004` — BLACK Rose Hoodie | Black Rose; published; not pre-order | “Number 32” / “Bay Area sports memory, recut.” | **CONFLICT / BLOCKED:** catalog hoodie is presented as a numbered sports piece. No scene-object proof. |
| `br-008` | `br-008` — BLACK is Beautiful Jersey Series: 1. SF Inspired (Football) | Black Rose; published; pre-order | “The Oakland Jacket” / “Black Rose armor for the night.” | **CONFLICT / BLOCKED:** catalog football jersey is presented as a jacket. No scene-object proof. |
| `br-009` | `br-009` — BLACK is Beautiful Jersey Series: 2. Last Oakland (Football) | Black Rose; published; pre-order | “The Bay Tee” / “The rose travels across the Bay.” | **CONFLICT / BLOCKED:** catalog football jersey is presented as a tee. No scene-object proof. |

The `#reserve` product rail selects WC products in category `pre-order` (with an unscoped fallback if empty), not catalog `is_preorder`. This is **UNKNOWN** until the WC category assignment and availability are captured against the same candidate snapshot.

## Required binding schema (concrete, fail-closed)

Create one immutable `catalog-binding.json` and one `shot-manifest.json` for the V2 candidate. Use UUIDs/IDs, not display filenames, as relationships. The minimum records below permit P1 to prove every product CTA and every media frame.

```json
{
  "candidate": {
    "id": "immutable-candidate-id",
    "git_baseline": "commit",
    "source_hashes": {"catalog_csv": "sha256", "sot_images": "sha256", "visual_manifest": "sha256"},
    "created_at": "ISO-8601",
    "policy_version": "required-policy-id"
  },
  "products": [{
    "sku": "canonical CSV SKU",
    "catalog_row_hash": "sha256",
    "dossier_ref": "canonical dossier name/slug",
    "woocommerce": {
      "product_id": "captured server ID",
      "status": "captured publish/draft",
      "price": "captured server value",
      "currency": "captured currency",
      "availability": "captured state",
      "captured_at": "ISO-8601"
    },
    "variations": [{
      "variation_id": "captured server ID or null for simple product",
      "attributes": {"Size": "value", "Color": "value"},
      "availability": "captured state",
      "price": "captured server value",
      "media_ids": ["shot-id"]
    }],
    "default_media_id": "shot-id",
    "purchasable": false
  }],
  "bridges": [{
    "bridge_id": "stable route/component/hotspot ID",
    "route": "/path/",
    "surface": "card|pdp|preorder|scroll-world|scene-hotspot",
    "sku": "required exact SKU or null",
    "variation_selector": "required before purchase when variable",
    "media_id": "shot-id or scene-id",
    "cta_target": "canonical WC product/variation route",
    "fallback": "text-equivalent canonical route",
    "state": "bound|blocked"
  }]
}
```

```json
{
  "media": [{
    "media_id": "stable ID",
    "source_authority": "sot-images.json|visual-manifest.json|hub-manifest.json",
    "source_record_path": "JSON pointer or canonical key",
    "candidate_path": "relative V2 path",
    "sha256": "candidate byte hash",
    "kind": "product-front|product-back|packshot|hero|scene|lockup|poster|video",
    "collection": "canonical collection or null",
    "sku": "exact SKU or null for non-product media",
    "variation_attributes": {"Size": "value", "Color": "value"},
    "approved_uses": ["route/surface role"],
    "rights": {
      "status": "approved|required",
      "record_id": "rights record ID",
      "creator": "recorded value",
      "location": "recorded value",
      "captured_or_expires_at": "ISO-8601"
    },
    "fallback_media_id": "required static/text fallback",
    "state": "bound|blocked"
  }]
}
```

**Validation rules:** reject a bridge when its SKU is absent from the catalog snapshot; its current WC product/variation cannot be captured; a product image is not selected through `sot-images.json`; a non-product scene lacks a visual-manifest/hub identity; its candidate hash differs from the registered byte; rights are absent/expired; the intended route/role is unapproved; or a hotspot tries to derive product identity from pixels or filename. Product associations for scenic frames are optional only when the frame has no product CTA; otherwise exact SKU and scene-object evidence are mandatory.

## Missing V2 links and blocking decisions

1. **Historical freeze instruction:** At discovery time the prototype V2 seed was untracked and lacked a candidate manifest/package hash. The active tracked candidate now has an immutable payload/package snapshot, but no final P1 binding can close until Woo variation, media, and rights evidence is complete.
2. **Register candidate media:** none of the five Scroll World snapshots has an observed manifest/rights/product record. Add evidence; do not infer from filenames or collection labels.
3. **Repair missing files or remove their references:** `forbidden-midnight-1280w.webp` and `luxury-nighttime-1280w.webp` are referenced but absent from V2 `assets/sot/`. Their V1 manifest records do not make a missing V2 URL valid.
4. **Resolve source identity for candidate-only additions:** V2’s `scene-*-gpt2.webp` files (Black Rose moon court, Signature Oakland atelier, Love Hurts cracked rose, Kids heir runway) are referenced but do not have an inspected matching V1 visual-manifest record. Rights and approved role are UNKNOWN.
5. **Bind all 33 product cards/PDPs:** V2 currently uses WC attachment IDs and category slugs. Add SKU -> WC product -> variation -> `sot-images.json` media mapping, then prove runtime parity. No product image association is approved by this audit.
6. **Correct or remove the five pre-order hotspot claims:** each current scene bridge lacks object evidence; four also conflict with catalog product identity. Bind to verified scene-object/product evidence or make the scene non-shoppable with a generic collection route.
7. **Provide collection-complete high-fidelity media contracts:** each Signature, Black Rose, Love Hurts, and Kids Capsule surface needs distinct story, card, PDP, and pre-order immersive assignments. Existing verified manifest records can seed several story/PDP assets, but no observed V2 record supplies the required four-surface, per-collection, rights-approved mapping. The global pre-order hero and Black Rose salon do not satisfy a per-collection pre-order scene requirement.
8. **Capture WooCommerce authority after source binding:** current product IDs, variations, stock, price, category/pre-order classification, and purchase state are UNKNOWN. This audit intentionally made no external WooCommerce read/write.

## Exact evidence files inspected

- [SOT registry](/Users/theceo/DevSkyy-skyyrose-flagship-v2-prototype/SOT.md)
- [Canonical catalog CSV](/Users/theceo/DevSkyy-skyyrose-flagship-v2-prototype/wordpress-theme/skyyrose-flagship/data/skyyrose-catalog.csv)
- [Generated product-image contract](/Users/theceo/DevSkyy-skyyrose-flagship-v2-prototype/data/sot-images.json)
- [Non-product visual-manifest](/Users/theceo/DevSkyy-skyyrose-flagship-v2-prototype/wordpress-theme/skyyrose-flagship/data/visual-manifest.json)
- [V2 collection/media and WC card helpers](/Users/theceo/DevSkyy-skyyrose-flagship-v2-prototype/wordpress-theme/skyyrose-flagship-2/functions.php)
- [V2 collection template](/Users/theceo/DevSkyy-skyyrose-flagship-v2-prototype/wordpress-theme/skyyrose-flagship-2/template-collection.php)
- [V2 pre-order scene bridge](/Users/theceo/DevSkyy-skyyrose-flagship-v2-prototype/wordpress-theme/skyyrose-flagship-2/page.php)
- [V2 home Scroll World/card references](/Users/theceo/DevSkyy-skyyrose-flagship-v2-prototype/wordpress-theme/skyyrose-flagship-2/front-page.php)
- [V2 PDP collection-world bridge](/Users/theceo/DevSkyy-skyyrose-flagship-v2-prototype/wordpress-theme/skyyrose-flagship-2/woocommerce/single-product.php)
- [V2 scroll-world snapshots](/Users/theceo/DevSkyy-skyyrose-flagship-v2-prototype/wordpress-theme/skyyrose-flagship-2/assets/scroll-world)
- [P1 procedure](/Users/theceo/plugins/fashion-theme-team/skills/fashion-theme-team/brain/showcase/v2-gap-closure.html)
