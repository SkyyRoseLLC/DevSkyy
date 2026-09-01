# V2 Scroll-World Commerce Scenes — Exact Creation Handoff

Date: 2026-08-25
Checkout: `/Users/theceo/DevSkyy-product-card-approved`
Branch: `codex/v2-approved-product-cards`
Baseline commit: `ca50e403e599079543df310b29b23781d2228bf8` (`fix(v2): block maskless localized product edits`)
Theme: `/Users/theceo/DevSkyy-product-card-approved/wordpress-theme/skyyrose-flagship-2`
Scope: create or correct the seven unfinished V2 commerce scenes, preserve the two founder-approved scenes, then present all nine as three collection-level review boards.
Authority boundary: local founder-review candidates only. No production deployment, staging sync, remote upload, manifest promotion, commercial approval, or product-truth replacement is authorized by this handoff.

## Mission in one paragraph

Finish the V2 collection Scroll World as nine product-first cinematic shopping scenes: three Black Rose chapters, three Love Hurts chapters, and three Signature chapters. Every collection must use three visibly different environments derived from a different aspect of its hero. Garments and accessories remain the principal subjects; the world supports the product and collection story. Build the environment around the approved product/model view so each result reads as one native photograph. Do not paste models onto unrelated plates, redraw product details from memory, or use a rejected candidate as product or scene authority.

## Immediate execution decision

There are **seven scenes to create/correct**, not nine:

| Scene | Current state | Required action |
| --- | --- | --- |
| `BR-COMMERCE-1` | Approved environment plate; product scene unfinished | Create a new native product scene |
| `BR-COMMERCE-2` | Approved environment plate; product contract changed | Create a new native product scene with corrected hoodie and shorts |
| `BR-COMMERCE-3` | Founder-approved product scene | Preserve exact bytes; do not regenerate |
| `LH-COMMERCE-1` | Founder-approved product scene with founder override | Preserve exact bytes; do not regenerate |
| `LH-COMMERCE-2` | Approved environment plate; product/logo pass unfinished | Create a new native shorts scene |
| `LH-COMMERCE-3` | Approved environment plate; product/logo pass unfinished | Create a new native Fannie scene |
| `SIG-COMMERCE-1` | Partial protected-model preview; founder recomposition requested | Recreate with the corrected left-statue/right-bridge composition |
| `SIG-COMMERCE-2` | Partial protected-model preview; founder rejected random rectangles | Recreate with no floating rectangular frames |
| `SIG-COMMERCE-3` | Product contract changed; model placement rejected | Recreate all three looks with natural physical relationships |

Do not show the founder one image at a time. Produce one contact sheet per collection:

1. Black Rose: two new candidates plus the preserved approved `BR-COMMERCE-3`.
2. Love Hurts: preserved approved `LH-COMMERCE-1` plus two new candidates.
3. Signature: three new candidates.

## Canonical authority and freshness lock

The following authority order is mandatory. A lower item cannot overrule a higher item:

1. The founder corrections recorded in this handoff.
2. Current product SOT: `/Users/theceo/DevSkyy-product-card-approved/data/product-sot.json`.
3. Founder-shot physical flatlay/source photography for the exact SKU and exact visible view.
4. Approved techflat for the exact SKU and view.
5. Exact logo, patch, embroidery, print, or lockup source.
6. Current dossier and catalog text.
7. Scene blueprint for story, cast, CTA, and plate state only.
8. Generated contact sheets and prior candidates as composition context only.

Current product SOT SHA-256:

```text
4ccfbe18aba2846c8406f1f0d34853158e563601051527a47f8e401a71beea12
```

That hash is also bound in both:

- `wordpress-theme/skyyrose-flagship-2/data/scene-narrative-blueprints.json`
- `wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/manifest.json`

Recompute the SOT hash before authoring every collection batch. Any change invalidates the prompt contracts, reference packs, review receipts, and previously generated unapproved outputs.

## Known stale or dangerous inputs

These are explicit stop rules:

- `preflight-v1/vision-authored-prompts/localized-product-patches-v3.json` records useful correction intent, but its declared route is `builtin-imagegen` / `gpt-image-2` / `precise-object-edit`. That is **not executable under the hardened gate** because the wrapper has no explicit mask and cannot prove zero outside-mask drift. Do not run it as written.
- A localized product correction must be re-authored as `operation: localized_product_patch`, with an inspected same-size explicit mask, a mask-capable adapter, a derived-mask hash, and outside-mask equality proof.
- For the seven unfinished scenes, prefer `operation: native_collection_scene` and `generator.route: environment_rebuild_around_source`. Build the scene from the approved on-model camera instead of trying to repair an already rejected-looking composite.
- Everything inside `founder-review-batch-v3/` is quarantined. `FOUNDER_REJECTED.md` says those protected composites read as pasted cutouts. They must not be wired, promoted, staged, or used as a generation source.
- The seven `*-natural-placement-v2.png` files are unapproved correction/layout candidates. They may explain what the founder disliked, but they are not product authority and must not silently become the next generation seed.
- Placeholder files listed in `data/founder-selected-theme-placeholders-v1.json` are placeholders, not final composite approval, generation authorization, or deployment authorization.
- `br-007-shorts-side.jpeg` is a **side view**, not a back view. Never relabel it as `back`.
- The founder-supplied `br-007-shorts-back-source.jpg` discussed earlier was also identified as a front/left-side view rather than a true rear view. Use the canonically renamed repo view metadata.
- Do not use a filename to infer garment view. Visually inspect every file and record `front`, `side`, `back`, `on_model_front`, `on_model_side`, or `on_model_back` in the JSON contract.
- Do not reuse invented lettering, approximate embroidery, a generic rose, a generic team patch, or professional-team marks.

## Current reviewer policy

The repository’s current tournament policy is:

```json
{
  "required_vision_judges": ["gpt-5.5-pro"],
  "synthesis_model": null,
  "minimum_each_vision_score": 95,
  "minimum_final_score": 95,
  "all_judges_available": true,
  "unverifiable_required_regions": 0,
  "source_hashes_current": true,
  "founder_approval_required": true
}
```

Gemini is not part of the current limited review policy. A GPT score is advisory evidence, not founder approval. The founder and GPT review the batch; the founder alone decides promotion. The independent reviewer must be different from the candidate author.

## Collection-wide visual rules

- Product hierarchy: exact garment/accessory first, model second, collection monument/story third, atmosphere last.
- The product must be large enough to inspect construction, material, trim, wording, logo, embroidery, patch, print, and color at full size.
- One camera and one light field per scene. Model, garment, floor, shadows, reflections, edge spill, grain, depth, and atmosphere must agree.
- Feet must have believable floor contact. Garments must show correct gravity and interaction with the body. No floating, pasted, sticker-like, or separately lit subjects.
- Altering a model pose is allowed only when the product is not altered. For a final exact-product scene, use an approved same-pose source, a verified protected-garment reconstruction, or a verified 3D garment route. A generative pose change that redraws the garment is candidate-only and cannot pass product truth.
- Each collection must have three distinct scenes. Do not reuse the same room, runway, camera, or decorative layout three times.
- Hero monuments must match the approved hero identity. Black Rose requires the exact font statue when the font chapter is active and the exact star/single-rose/stem/leaves identity when the graphic chapter is active; its base may adapt. Signature requires the exact cleaned `Skyy Rose` font sculpture or exact SR-and-rose monogram, depending on chapter. Love Hurts requires the protected enchanted rose to remain legible and in focus where specified.
- No text is added to the generated scene as UI. CTA labels and product links belong in the theme overlay, not inside image pixels.
- The CTA may represent a set/look/pre-order edit, but every item retains its own product link. Do not invent a bundle SKU.

## Black Rose batch

### `BR-COMMERCE-1` — The Type Foundry

**Exact cast**

- `br-001` — BLACK Rose Crewneck
- `br-002` — BLACK Rose Joggers

**Commerce behavior**

- Primary CTA: `Complete the Set`
- Both products must have separate links.
- The set is editorial only; there is no aggregate bundle product.

**Composition**

- One model wears the crewneck and joggers together as a natural complete set.
- The model/product occupies the principal focal zone and dominates the frame.
- Use the hero-derived silver Black Rose font-and-rose sculpture as architecture aligned inline with the wall, not floating in front of it.
- Preserve the legible font statue silhouette. Do not replace it with approximate typography or a generic rose sign.
- The environment can include restrained Oakland night/bridge atmosphere, but it cannot compete with the black set.
- Choose a camera and light direction that reveal the crewneck’s raised relief without turning it into white ink.

**Non-negotiable product truth**

- `br-001`: heavyweight black fleece crewneck; no hood, zipper, buttons, or pocket; white ribbed neckband, cuffs, and waist hem.
- `br-001`: centered approximately 10-inch black/white/grey **raised embossed** three-rose/cloud relief. It is not front embroidery, print, silicone, a white patch, or flat ink.
- `br-001`: back center remains blank except the small reduced-palette embroidered rose/cloud cluster below the collar.
- `br-002`: black fleece joggers; white ribbed waistband and white ankle cuffs; tapered jogger construction.
- `br-002`: exact small molded silicone appliqué on wearer-left thigh. It is not embroidery or a printed rose.

**Primary product authorities**

```text
assets/products/source-photos/black-rose/br-001-crewneck-front-authentic.png
c9cd0614f52f8c9a6f8bcfb098acd89bcca15ea612cb62a48bba5d48697123c2

assets/products/source-photos/black-rose/br-002-joggers-front-authentic.png
f7596bbcee6a189aac561b60261bb313ff063cc62722a8f13fe721f0ac7a9928

wordpress-theme/skyyrose-flagship/assets/images/products/br-001-onmodel.webp
d1afd357e6590de47fb7a2cd3512cce112eeee6dbf5aa2cfc1fb2f0232f78724
```

**Hero font authority**

```text
wordpress-theme/skyyrose-flagship-2/assets/sot/images/lockups/hero-derived/black-rose-font-statue-hero-exact-v1.png
a879be465973a5edf2aa8fe6ae2e0eb34d66e11cf3e5081b3f4beea4d967aa3c
```

**Hard reject**

- Embroidered or white printed chest logo.
- Black cuffs/neck/hem instead of white.
- Generic rose or altered Black Rose lettering.
- Pasted model edges or unmatched reflection/contact shadow.

### `BR-COMMERCE-2` — The Open Waterfront

**Exact cast**

- `br-005` — BLACK Rose Hoodie — Signature Edition
- `br-007` — BLACK Rose x Love Hurts Basketball Shorts

**Commerce behavior**

- Primary CTA: `Complete the Look`
- Both products receive separate links.
- No invented bundle product.

**Composition**

- One model wears the hoodie and shorts together in a natural waterfront look.
- Build an open Oakland waterfront at night with the Bay Bridge moved into supporting depth, water reflections, and a full moon.
- This is the bridge/landscape chapter. Do not insert completed Black Rose statues in this scene.
- The environment must feel open-air and physically connected to the model. Match wet-ground reflection, moon/bridge spill, edge light, and contact shadow to the source.

**Non-negotiable product truth**

- `br-005`: black pullover hoodie in lightweight polyester/jogger-feel fabric, white drawstrings, kangaroo pocket, tonal ribbing, and sublimated rose-print inner hood lining.
- `br-005`: small raised silicone cutout on the right chest.
- `br-005`: large rose/cloud artwork belongs on the **side-body panel of the hoodie, not the arm or forearm**. This founder correction overrides the stale “forearm art” sentence in the scene blueprint.
- `br-007`: black mesh basketball shorts with tonal rose/cloud repeat, narrow white side constructions, white waistband/drawstring, three zipper pockets, and black/white/gold binding.
- `br-007`: readable `OAKLAND` tackle twill across the front; no mark may cover the final `D`.
- `br-007`: `Love Hurts` script sits above the narrow side insert.
- `br-007`: the white side section remains narrow. Do not enlarge it into a broad front-facing white panel.
- `br-007`: small exact black rose with green stems and blue cloud stays inside the narrow white side panel.

**Primary product authorities**

```text
assets/products/source-photos/black-rose/br-005-signature-edition-hoodie.jpeg
e3e0bb25ec76bf07f806746810757e3b6a74b9092a12a180b021050afd4c5e4c

assets/products/source-photos/black-rose/br-007-shorts-front.jpeg
a176ba1ade26f1279e1d4847b204fe838d81951942686721e5e04203a1695314

assets/products/source-photos/black-rose/br-007-shorts-side.jpeg
850d29f480bcb8e282a21ffca427bd368a03d05ff5f61fad8a8ff58f497ec8f8

assets/products/source-photos/black-rose/br-007-shorts-back-hanger.jpeg
8ee61bc5c937b9caf20e3d33445c9eaee8de70bc48c13a3a782006bf192eb2cb

assets/products/source-photos/black-rose/br-007-shorts-back-detail.jpeg
34ad777e56b38bea0d86c87e9815c537b109ab7fcf39614711093fb55f2309d2
```

The side authority is valid only for a side-facing worn region. It is not a rear authority.

**Hard reject**

- Hoodie art on the sleeve or forearm.
- Embroidered chest mark instead of silicone.
- Oversized white shorts panel.
- Incomplete `OAKLAND`, broken `Love Hurts`, wrong rose/cloud colors, or generic collab art.
- A model standing in a decorative empty pocket with no optical relationship to the waterfront.

### `BR-COMMERCE-3` — The Town Line

**Status: preserve, do not generate.**

Approved asset:

```text
wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/black-rose/br-commerce-3-five-jersey-lounge-a-black-founder-approved-v1.png
90aa66aa432bc9176b393dfbb7ce15d507860e47c970650337f78c4b1a480683
```

Exact cast: `br-008`, `br-009`, `br-010`, `br-011`, `br-012`.
CTA: `Pre-Order the Jersey Series`; every jersey gets its own pre-order link.

Preserve the approved 79-pixel correction: only the interior face of the `A` in `BLACK` on `br-012` is black; the gold stitched edge and every other pixel remain unchanged. If the asset hash differs, stop. Do not “improve” or regenerate this scene.

## Love Hurts batch

### `LH-COMMERCE-1` — The Vow Aisle

**Status: preserve, do not generate.**

Approved asset:

```text
wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/love-hurts/lh-commerce-1-vow-aisle-founder-approved-v2.png
1eb217504536634047eb3a682915a4d8ea0218b9968c9d3ff96addb231578f38
```

The founder approval receipt explicitly says `wire this one as the approved scene`. Preserve exact bytes. It is approved for V2 scene wiring only; deployment and production writes remain unauthorized.

Exact cast:

- Male: `lh-004` bomber + `lh-002` black joggers.
- Female: `lh-004` bomber + `lh-006` white joggers.
- CTA: `Complete the Look`; separate links for bomber, black joggers, and white joggers.

The product pair and enchanted rose under glass are co-primary focal subjects. Do not crop the rose away on the collection review board.

### `LH-COMMERCE-2` — The Fractured Chamber

**Exact cast**

- `lh-003` — Love Hurts Basketball Shorts only.

**Commerce behavior**

- Primary CTA: `Shop the Shorts`
- Direct link to `lh-003`.

**Composition**

- One on-model shorts look; the shorts are the sole product protagonist.
- Use a closer, lower camera than the Vow Aisle.
- Create a distinct thorned runway/cracked-rose chamber. Do not repeat the wide two-model cathedral aisle.
- Keep the independently lit enchanted rose clearly in focus on the opposite side of the frame.
- Crop and pose must make the shorts large enough to prove the entire visible panel construction and lettering.

**Non-negotiable product truth**

- White mesh body with a complete repeating red-rose/green-stem/white-cloud sublimation.
- Black waistband with no visible exterior drawstring.
- Black vertical pocket welts, sculpted black mesh side panels, and black hem binding with white/red piping.
- Complete, readable red `Love Hurts` wording. Do not truncate, mutate, or invent letters.
- Exact side-panel embroidery/marks and physical panel geometry.
- The visible side of the worn garment must be matched to the authority view used. Do not use a rear source to repair a front/side view or vice versa.

**Primary product authorities**

```text
assets/products/source-photos/love-hurts/lh-003-shorts-front.jpeg
5d159098e2a965d16b56a6751de45e3e42f49f7dc2a4c2fc3d0556c1d048307f

assets/products/source-photos/love-hurts/lh-003-shorts-back.jpeg
65e36327b0dceaee34a66ccd4af0a0811047badcb7f6afe9eaa00980f4d4f53c

assets/products/source-photos/love-hurts/lh-003-shorts-techflat-allover-front.jpg
4add28274bf053d70365200fc72a3808b14aabe485184042d66d0255ae261993

assets/products/source-photos/love-hurts/lh-003-shorts-techflat-allover-back.jpg
41e30b9f8bc94e3cfe44d2f3748f16014be827323e8c4152a8dc57c1eb14f350
```

**Hard reject**

- Wrong or incomplete logo.
- Missing rose/cloud repeat.
- Random flowers, generic hearts, altered panel boundaries, or an exterior drawstring.
- Shorts too small to inspect.
- Repeating the exact camera/environment from `LH-COMMERCE-1`.

### `LH-COMMERCE-3` — The Rose Vitrine

**Exact cast**

- `lh-005` — The Fannie only.

**Commerce behavior**

- Primary CTA: `Pre-Order The Fannie`
- Direct pre-order link to `lh-005`.

**Composition**

- Close on-model cross-body composition at accessory scale.
- Use an intimate threshold/vitrine environment, not another wide cathedral scene.
- The bag and enchanted rose are both sharply legible; the bag remains the product protagonist.
- Show believable strap tension, body contact, cast shadow, leather highlights, and scene spill.

**Non-negotiable product truth**

- Small rectangular black pebbled PU/faux-leather fanny pack/cross-body sling.
- Horizontal front and rear zipper pockets, adjustable black nylon webbing, black buckle, and black hardware.
- Exact white embroidered `FANNIE` wordmark with the exact small red rose accent, correct scale, letter forms, and placement.
- Do not turn it into a purse, backpack, tote, or generic belt bag.

**Primary product authorities**

```text
wordpress-theme/skyyrose-flagship/assets/images/products/lh-005-fannie.jpeg
8cae4c6b10bfe9bcd4ba60b4a5529769064ef9b5d558527bb71126a115697f18

wordpress-theme/skyyrose-flagship/assets/images/products/lh-005-onmodel.webp
e3df4aa2781881b9364490c09611308d0c9c0a42fbab19fd114365eca8bda43c
```

**Hard reject**

- Misspelled `FANNIE`, wrong lettering, generic logo, extra rose, or changed bag geometry.
- Floating strap, no body contact, or inconsistent shadows.
- A distant full-body scene where the accessory cannot be inspected.

## Signature batch

### `SIG-COMMERCE-1` — The Oakland Atelier

**Exact cast**

- `sg-009` — The Sherpa Jacket
- `sg-007` — The Signature Beanie

**Commerce behavior**

- Primary CTA: `Complete the Look`
- Separate links for jacket and beanie.

**Founder-locked recomposition**

- Do not bury the font sculpture in the wall.
- Place the exact cleaned gold `Skyy Rose` font sculpture standing alone on the **left side of the frame**.
- Move the bridge landscape to the **right side of the frame**.
- The model wears the Sherpa and beanie and leans naturally back against the sculpture while looking over the shoulder toward the bridge.
- The contact between model and sculpture must be physically believable: correct shoulder/back contact, occlusion, cast shadow, and balance.
- The sculpture is supporting architecture but must remain fully legible; it cannot be cut into random fragments or replaced with approximate type.

**Non-negotiable product truth**

- `sg-009`: smooth black nylon/windbreaker-style exterior, cream/white sherpa lining, visible pile at placket/cuffs/hem/tall funnel collar, full gold-toned zipper, lower patch pockets.
- `sg-009`: exact small red rose/cloud embroidery at wearer-left chest and exact lower hem tab.
- `sg-009`: no hood; sherpa is the interior, not the exterior.
- `sg-007`: black rib-knit cuffed beanie with a small rectangular molded silicone rose-logo patch on the cuff, slightly off-center to wearer-left.
- `sg-007`: the patch is silicone, not direct embroidery or a generic woven square.

**Primary product authorities**

```text
wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/protected-model-layers/sg-009-onmodel-front-protected-v1.png
4626dade5bf9a3825b5d4a65afac8df214e7f7aae2773248871a45337a9d9a45

wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/protected-model-layers/sg-007-onmodel-protected-candidate-v1.png
45c6bca620ca0a6793e093eeb1d9e0373d4992f0447cf975863e1194da13fcbb

wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/preflight-v1/generator-conditioning-v2/sg-009-sg-007-detail-proof-v2.png
af47c17849e7b33354c82f7fc2da0ba16a9424a0e53e51538e1af97408b199ad
```

**Signature font authority**

```text
wordpress-theme/skyyrose-flagship-2/assets/sot/images/lockups/founder-supplied/derived/signature-skyyrose-font-sculpture-protected-v1.png
e259c246a320b8220044da915c3b6fa89ecbbe7ab4a74eb80bafebe668d0e274
```

**Hard reject**

- Font sculpture lost in the wall, cropped beyond recognition, or altered lettering.
- Bridge behind the model instead of opening to the right.
- Model merely placed beside the sculpture with no believable lean/contact.
- Exterior sherpa, hooded jacket, wrong chest rose, or embroidered beanie patch.

### `SIG-COMMERCE-2` — Fog Over the Bay

**Exact cast**

- Male: `sg-013` Mint & Lavender Crewneck + `sg-014` Mint & Lavender Sweatpants.
- Female: `sg-006` Mint & Lavender Hoodie.
- Female lower garment is plain and unbranded unless a separately approved exact SKU is added later. Do not invent a matching trouser product.

**Commerce behavior**

- Primary CTA: `Complete the Look`
- Separate links for `sg-013`, `sg-014`, and `sg-006`.

**Founder-locked recomposition**

- Remove the empty rectangular cutouts/frames that appeared randomly behind the models.
- Use fog, open Bay water, bridge light, dusk atmosphere, terrace architecture, and rose-gold reflected lines as spatial structure—not floating frames.
- Models must share one floor plane and light field while remaining visually separate enough to read each product.
- The scene must be materially different from the warm indoor Signature atelier and the sunrise departure terrace.

**Non-negotiable product truth**

- All three products use the same muted mint/seafoam base family.
- `sg-013`: solid mint crewneck, no hood, no zipper, no chevrons or color blocks; large centered lavender rose/cloud embroidery; small back-neck embroidery.
- `sg-014`: solid mint tapered sweatpants; small lavender embroidered rose/cloud mark on wearer-left thigh.
- `sg-006`: solid mint pullover hoodie with white drawstrings, mint kangaroo pocket, and large lavender rose composition centered on the chest; plain mint back.
- Maintain real stitch/print relief according to the current product SOT and visual proof. Never substitute generic flat purple roses.

**Primary product authorities**

```text
wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/preflight-v1/generator-conditioning-v2/sg-013-sg-014-sg-006-detail-proof-v2.png
0cbf21c27ca3553bfcc2114d44fb567e58799d88f26a2a4c76bf62546b88231c

wordpress-theme/skyyrose-flagship/assets/images/products/sg-013-onmodel.webp
cb58e694639ff02b4dd7c1d009516df237f0acb39a8c6f6948930dad5d3c157b

wordpress-theme/skyyrose-flagship/assets/images/products/sg-006-onmodel.webp
510cac2c051615d0713180d512c1f4f132e6d774dd989e2e536e608be62bfe2f
```

`sg-014` currently has a physical packshot and ghost-front source, not an approved native joint on-model pair. If the exact male crewneck+sweatpants pose cannot be sourced without redrawing product construction, stop and obtain a founder-approved same-pose source or use a verified garment reconstruction route.

**Hard reject**

- Random rectangular portals or empty cutouts behind models.
- Rainbow chevrons, a zip hoodie, a branded female trouser, wrong product pairing, or flattened/incorrect lavender art.
- Two separately lit cutouts with no shared floor contact or scene grain.

### `SIG-COMMERCE-3` — The Departure Terrace

**Exact cast and product groups**

1. Bay Bridge look: `sg-005` Bay Bridge Shirt + `sg-001` Bay Bridge Shorts.
2. Stay Golden look: `sg-002` Stay Golden Shirt + `sg-003` Stay Golden Shorts.
3. Windbreaker look: `sg-015` Windbreaker jacket and pants, sold together as one canonical set SKU.

This is five product identities and four separately reservable product groups because the windbreaker jacket and pants are one product.

**Commerce behavior**

- Primary CTA: `Explore Pre-Orders`
- Separate pre-order links for `sg-001`, `sg-005`, `sg-003`, `sg-002`, and `sg-015`.
- Do not split `sg-015` into invented jacket and pants SKUs.

**Founder-locked recomposition**

- Repose or reframe the models so they are naturally participating in the terrace—not dropped into three evenly spaced slots.
- The Bay Bridge pair, Stay Golden pair, and Windbreaker set must each have a physical relationship to the terrace: weight shift, walking/turning action, interaction with railing or sightline, shared reflection, and believable overlap/depth.
- Model poses may change only if product truth remains intact under the hardened pose rule.
- Preserve open sunrise Bay light and the bridge/skyline as supporting depth.
- Use the exact SR-and-rose monument as a separate environmental object. The graphic stands alone naturally; do not merge it into clothing, distort it, or replace it with approximate lettering.

**Non-negotiable product truth**

- `sg-001`: white mesh shorts with full daytime Bay Bridge/clear-blue-sky/water sublimation, blue waistband, white drawstring, and small blue rose/cloud mark at lower wearer-left leg.
- `sg-005`: white tee with the exact blue/cyan rose-cluster and Bay Bridge imagery stitched inside the rose.
- `sg-003`: white mesh shorts with a continuous purple-violet night Golden Gate/city-light sublimation, purple waistband, white drawstring, and purple embroidered lower-left rose.
- `sg-002`: white tee with the exact purple rose-cluster and Golden Gate imagery stitched inside the rose.
- `sg-015`: one white nylon jacket-and-pants set with pink hood, pastel pink/lavender/mint/yellow chevrons, pink-trim pockets/zippers, multicolor striped bands, small pink rose marks, and large SR monogram at upper jacket back.
- Do not mix the blue Bay look with the purple Stay Golden look.

**Primary product authorities**

```text
wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/preflight-v1/generator-conditioning-v2/sg-002-sg-003-detail-proof-v2.png
2b11e6f6aca4b35587c7c6672483e49bb07403fcde2646bae0eda0245f247209

wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/preflight-v1/generator-conditioning-v1/signature-windbreaker-proof-board-v1.png
a4a7f4d02ca77a2c1e88d793ebb03cc7b902bb49618cb09d2199f23679f6ac66

wordpress-theme/skyyrose-flagship/assets/images/products/sg-001-bay-bridge-shorts.jpeg
3bfe3a460f6bb425240f2c268977e6c11f248bfd9bf9a872d83d2d11c441a950

wordpress-theme/skyyrose-flagship/assets/images/products/sg-005-onmodel.webp
e75b7f78bc271338c378a2f71b449602b9b773d18afbe09f809120e50f70265a

wordpress-theme/skyyrose-flagship/assets/images/products/sg-015-onmodel.webp
3d2c32ea1ef392a234d25d487c524234faa3f3c05830673ce8ba387898de52a5
```

**Signature graphic authority**

```text
wordpress-theme/skyyrose-flagship-2/assets/sot/images/logos/sr-monogram-rose-gold.webp
a67a4efc3811828f6bcea872422a76d85fb9afd929266256c71fc4c0ce7b1d1d
```

**Hard reject**

- Models lined up like pasted catalog cutouts.
- Wrong bridge/color story on either pair.
- Missing Bay Bridge shirt or shorts.
- Generic windbreaker, wrong hood color, missing chevrons/bands, or splitting the set into fake SKUs.
- Altered SR monogram/rose graphic.

## Required workflow for every unfinished scene

The Fashion Theme Team owns the collection story and commerce handoff. Its embedded `product-fidelity-image-edits` skill owns media truth and native optical integration.

### 1. Preflight the worktree without changing user files

```bash
cd /Users/theceo/DevSkyy-product-card-approved
git branch --show-current
git rev-parse HEAD
git status --short
shasum -a 256 data/product-sot.json
```

The worktree is already dirty with approved/current V2 work. Do not reset, stash, clean, overwrite, or broadly stage it. Add only new batch contracts, candidates, receipts, contact sheets, and the explicitly requested wiring changes.

### 2. Create a current source ledger before prompting

For each scene, visually inspect every reference and write a JSON contract containing:

- `schema: product-fidelity-edit.v1`
- `operation: native_collection_scene`
- exact collection, scene ID, SKU cast, and CTA ownership
- one declared view and role for every reference
- current path and SHA-256 for every reference
- exact visible wording
- product technique locks: embossed, silicone, embroidery, tackle twill, sublimation, satin, nylon, sherpa, or other material
- `pose_change: false` unless a verified route supports it
- one measured optical contract
- independent review thresholds
- `founder_approval_required: true`
- `wiring_allowed: false`
- `deployment_allowed: false`

Every SKU must have a `physical_product_authority` or `approved_techflat`. A dossier, logo crop, candidate, protected full-model layer, filename, or contact sheet cannot satisfy product truth by itself.

### 3. Measure the optical contract before writing the prompt

Record integer pixel values for:

- output dimensions
- subject bounding box
- horizon Y
- both foot/contact points for each full-body model
- lens class and camera height
- key-light direction, temperature, and softness
- shadow direction and softness
- floor material and reflection mode
- contact shadow, edge spill, depth occlusion, atmospheric depth, and grain match

The environment must be designed from those source measurements. Do not generate an arbitrary environment first and search for somewhere to place the model afterward.

### 4. Validate the selected model/adapter

The model registry and gate live here:

```text
/Users/theceo/plugins/fashion-theme-team/skills/product-fidelity-image-edits/
```

Before a provider call:

```bash
python3 /Users/theceo/plugins/fashion-theme-team/skills/product-fidelity-image-edits/scripts/model_registry.py validate \
  --model <exact-registry-model-id> \
  --operation native_collection_scene
```

Do not assume a provider supports a capability because its API does. The actual adapter/wrapper must expose the required reference, mask, or identity control. Unknown, stale, unavailable, or mismatched routes fail closed.

### 5. Run the fidelity preflight and optimize the reference pack

```bash
python3 /Users/theceo/plugins/fashion-theme-team/skills/product-fidelity-image-edits/scripts/fidelity_gate.py preflight \
  --contract /absolute/path/to/scene-contract.json \
  --workspace /Users/theceo/DevSkyy-product-card-approved \
  --receipt /absolute/path/to/preflight-receipt.json

python3 /Users/theceo/plugins/fashion-theme-team/skills/product-fidelity-image-edits/scripts/fidelity_gate.py optimize \
  --contract /absolute/path/to/scene-contract.json \
  --workspace /Users/theceo/DevSkyy-product-card-approved \
  --out-dir /absolute/path/to/reference-pack
```

Visually inspect the optimized pack. Derived files never replace physical authorities. Recompute all input hashes after prompt authoring; hash drift invalidates the prompt.

### 6. Prompt only after the visual/source pass

Prompts must live in JSON or HTML, not only in chat. Each prompt must include:

- one camera, one lens behavior, and one light field
- exact product feature locks verbatim
- exact visible wording verbatim
- exact collection-story role
- exact hero aspect for this chapter
- natural physical interaction and contact requirements
- explicit anti-invention and anti-mixing constraints
- candidate-only promotion boundary

The visualizer/vision reviewer should first describe what is actually visible in the physical sources, then author the prompt from that evidence. Do not start from the dossier and ask the model to imagine the garment.

### 7. Generate only collection-sized batches

Recommended output root:

```text
wordpress-theme/skyyrose-flagship-2/assets/scroll-world/generated-candidates/founder-commerce-scenes-v1/native-scene-regeneration-v2/
```

Recommended filenames:

```text
black-rose/br-commerce-1-native-candidate-v1.png
black-rose/br-commerce-2-native-candidate-v1.png
love-hurts/lh-commerce-2-native-candidate-v1.png
love-hurts/lh-commerce-3-native-candidate-v1.png
signature/sig-commerce-1-native-candidate-v1.png
signature/sig-commerce-2-native-candidate-v1.png
signature/sig-commerce-3-native-candidate-v1.png
```

Generation produces review candidates only. It does not authorize wiring.

### 8. Verify product fidelity and native integration separately

Create a hash-bound independent review JSON for each candidate using `product-fidelity-native-review.v1`. Minimum scores:

```json
{
  "product_fidelity": 95,
  "optical_integration": 90,
  "anatomy_pose": 90,
  "collection_story": 90,
  "commerce_readiness": 90
}
```

Then run:

```bash
python3 /Users/theceo/plugins/fashion-theme-team/skills/product-fidelity-image-edits/scripts/fidelity_gate.py verify \
  --contract /absolute/path/to/scene-contract.json \
  --workspace /Users/theceo/DevSkyy-product-card-approved \
  --output /absolute/path/to/candidate.png \
  --review /absolute/path/to/native-review.json \
  --receipt /absolute/path/to/verification-receipt.json
```

Any hard fail blocks the scene regardless of average score. Inspect at original size, not only in a contact sheet.

### 9. Produce structured collection handoffs

Each collection batch must contain:

```text
preview.html
contract.json
evidence.json
contact-sheet.jpg
```

The HTML and JSON must use the same stable collection, scene, SKU, CTA, reference, candidate, and evidence IDs. Each contact sheet must show all three collection scenes together, including the preserved approved scene where applicable. Include close product-detail crops below the scene row so embroidery, silicone, embossing, tackle twill, sublimation, satin, sherpa, and lettering can be reviewed at useful scale.

### 10. Stop for founder review

Before founder approval, all new scenes remain:

```json
{
  "founder_status": "PENDING",
  "wiring_allowed": false,
  "deployment_allowed": false
}
```

Do not edit the runtime manifest, collection template, product registry, CSS/JS wiring, or deployment state merely because generation or GPT review passed. After the founder approves named hashes, prepare a separate wiring change and re-run theme/browser validation.

## Final acceptance checklist

A collection batch is ready for founder review only when every item below is true:

- [ ] Current SOT hash matches the contract and receipt.
- [ ] Every product source was visually classified by exact view.
- [ ] Every reference path exists and its hash matches.
- [ ] No rejected scene or placeholder was used as product authority.
- [ ] No stale prompt contract was executed.
- [ ] Exact SKU cast and CTA mapping match the scene blueprint.
- [ ] Every product technique and visible wording matches physical authority.
- [ ] No invented product, logo, panel, patch, letter, embroidery, or professional-team mark appears.
- [ ] Camera, horizon, feet, shadow, reflection, light, grain, edge spill, occlusion, and depth read as one photograph.
- [ ] The collection has three distinct chapters rather than three versions of the same room.
- [ ] Product remains the dominant commerce subject.
- [ ] Independent reviewer is not the candidate author.
- [ ] Candidate SHA is bound to the native review and verification receipts.
- [ ] Contact sheet and full-resolution detail crops exist.
- [ ] Founder status remains pending until the founder explicitly approves the named asset/hash.
- [ ] Wiring, commit, staging, deployment, and commercial authorization remain separate.

## Copy/paste opening instruction for the next task

```text
Continue the V2 Scroll-World commerce-scene creation from:
/Users/theceo/DevSkyy-product-card-approved/docs/design/v2-remodel/reports/v2-next-commerce-scene-creation-handoff-2026-08-25.md

Work only in /Users/theceo/DevSkyy-product-card-approved on branch codex/v2-approved-product-cards. Preserve all existing dirty-worktree changes. Use the Fashion Theme Team with its embedded product-fidelity-image-edits workflow. First verify the checkout, current data/product-sot.json hash, every reference path/hash/view, current judge policy, and the two founder-approved scene hashes. Do not regenerate BR-COMMERCE-3 or LH-COMMERCE-1. Create the seven unfinished scenes in three collection batches, using native_collection_scene contracts and environment_rebuild_around_source wherever an approved on-model view exists. Do not execute localized-product-patches-v3.json as written; built-in Imagegen is prohibited for maskless localized corrections. Do not use founder-review-batch-v3 or any rejected candidate as a source. Produce one complete review board per collection, with all three scenes and full-size product-detail crops. Run the independent product-fidelity and native-integration gates after every generated batch. Stop with founder_status PENDING and wiring_allowed false. Do not deploy, stage, upload, promote, or wire anything until I approve the exact candidate hashes.
```
