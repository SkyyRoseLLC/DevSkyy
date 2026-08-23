# V2 Product Dossier Visual Audit — 2026-08-22

## Outcome

All 33 active catalog SKUs were compared against the available founder flatlays,
physical-product photos, and techflats before the canonical product SOT was
regenerated. Physical/founder imagery controls visible color, construction,
placement, and decoration. A matching techflat controls unseen angles and exact
panel continuity. Existing filenames, generated media, and older prose do not
overrule the pixels.

Product-card image generation remains fail-closed: any approved media attached
to a product whose visual contract changed was cleared and marked
`STALE_PRODUCT_HASH`. No old approval was silently rebound to a changed dossier.

## Collection audit evidence

- Black Rose: `.artifacts/dossier-visual-audit/black-rose/page-01.jpg` through
  `page-06.jpg`
- Love Hurts: `.artifacts/dossier-visual-audit/love-hurts/page-01.jpg` through
  `page-05.jpg`
- Signature: `.artifacts/dossier-visual-audit/signature/page-01.jpg` through
  `page-08.jpg`
- Kids Capsule: `.artifacts/dossier-visual-audit/kids-capsule/page-01.jpg`
  through `page-02.jpg`

Each current product record and its hash-bound primary/secondary evidence paths
are serialized in `data/product-sot.json`.

## Per-SKU result

| SKU | Product | Visual result |
| --- | --- | --- |
| `br-001` | BLACK Rose Crewneck | Corrected to the raised embossed black/white/grey front rose; small Black Rose cluster at back neck; plain back center; no SR monogram. |
| `br-002` | BLACK Rose Joggers | Corrected to black drawstring, white side panels, and plain back; removed invented rear gold SR/extra logo. |
| `br-003` | Baseball Classic — Black | Confirmed black button-front jersey, white tackle-twill wordmark spanning approximately 80% of visible torso width, exact 3 × 4in lower-left patch, large back crest, white collar/placket/cuffs, and plain black bottom hem. |
| `br-004` | BLACK Rose Hoodie | Corrected small wearer-left chest mark, large wearer-left forearm artwork, white drawstrings, and floral hood lining; no circular arm patch. |
| `br-005` | BLACK Rose Hoodie — Signature Edition | Corrected to small wearer-left chest mark plus large wearer-left forearm artwork; removed invented hip/side-body placement. |
| `br-006` | The Bomber Sherpa | Confirmed black satin shell, black sherpa hood/exterior elements, snap storm flap over zipper, small front rose, and large back rose. |
| `br-007` | BLACK Rose × Love Hurts Basketball Shorts | Confirmed existing front/back mesh construction, Oakland tackle-twill, and mixed Black Rose/Love Hurts decoration map. |
| `br-008` | SF Inspired Football | Corrected numeral artwork: front rose fill in `8` only with plain white `0`; back plain white `8` with rose fill in `0`. |
| `br-009` | Last Oakland Football | Confirmed actual black/silver Oakland football colorway, tackle-twill lettering/numerals, patch placement, and front/back layout. |
| `br-010` | The Bay Basketball | Confirmed actual basketball colorway, lettering/numerals, patch placement, and front/back layout. |
| `br-011` | The Rose Hockey | Removed invented league/NHL shield; corrected collar to the small Black Rose floral crest from the techflat. |
| `br-012` | Last Oakland Baseball | Confirmed physical front/back layout and standardized the tackle-twill wordmark to approximately 80% of visible torso width; normalized the front-chest dossier structure so technique parsing cannot invent a bogus region. |
| `br-014` | Baseball Classic — Giants | Confirmed black/orange colorway, approximately 80%-torso-width front wordmark, and exact patch/back crest; corrected catalog trim summary to a plain black bottom hem. |
| `br-015` | Baseball Classic — White | Confirmed white/black colorway, approximately 80%-torso-width front wordmark, and exact patch/back crest; corrected catalog trim summary to a plain white bottom hem. |
| `lh-002` | Love Hurts Joggers — Black | Founder correction 2026-08-23: black drawstring and small embroidered Heart-and-Roses Composite on wearer-left thigh, as shown by the techflat; no cloud-cluster substitution; plain back. |
| `lh-003` | Love Hurts Basketball Shorts | Corrected the large front script to wearer-left, added both distinct side-panel marks, two rear welt pockets, and the large rear-right red/white heart composition; removed the invented exterior drawstring. |
| `lh-004` | Love Hurts Bomber Jacket | Confirmed split front Love/Hurts lettering, white/black satin construction, rose hood lining, and large back heart/rose graphic. |
| `lh-005` | The Fannie | Corrected to black pebbled faux leather with separate front and rear horizontal zipper pockets, white FANNIE embroidery, and red rose accent. |
| `lh-006` | Love Hurts Joggers — White | Founder correction 2026-08-23: black drawstring and small embroidered Heart-and-Roses Composite on wearer-left thigh, as shown by the techflat; no cloud-cluster substitution; plain back. |
| `sg-001` | Bay Bridge Shorts | Confirmed daytime Bay Bridge sublimation, blue waistband, white drawstring, and lower wearer-left blue rose/cloud mark. |
| `sg-002` | Stay Golden Shirt | Corrected the artwork from a small left-chest mark to the large centered embroidered violet rose with Golden Gate imagery inside; small rear-neck SR retained. |
| `sg-003` | Stay Golden Shorts | Corrected waistband from black to deep purple; retained white drawstring, night Golden Gate sublimation, and lower wearer-left purple rose. |
| `sg-005` | Bay Bridge Shirt | Corrected the artwork from a small left-chest mark to the large centered embroidered blue rose with Bay Bridge imagery inside; small rear-neck SR retained. |
| `sg-006` | Mint & Lavender Hoodie | Corrected hood drawstrings to white; retained solid mint pullover, large centered lavender rose, kangaroo pocket, and plain back. |
| `sg-007` | Signature Beanie | Corrected decoration technique to a small rectangular silicone patch, slightly wearer-left of center; no direct knit embroidery. |
| `sg-009` | Sherpa Jacket | Confirmed black shell, white/cream sherpa lining, stand collar, small front red rose/cloud embroidery, and plain exterior back. |
| `sg-011` | Original Label Tee — White | Founder-confirmed blank white exterior retained; apparent sleeve piece remains classified as a retail hang tag, not garment branding. Parser format normalized. |
| `sg-012` | Original Label Tee — Orchid | Founder-confirmed blank orchid exterior retained; apparent sleeve piece remains classified as a retail hang tag, not garment branding. Parser format normalized. |
| `sg-013` | Mint & Lavender Crewneck | Confirmed solid mint crewneck, large centered front lavender embroidery, and small back-neck lavender embroidery. |
| `sg-014` | Mint & Lavender Sweatpants | Corrected drawstring to white; retained solid mint body, wearer-left thigh lavender embroidery, and plain back. |
| `sg-015` | Windbreaker Set | Corrected chevrons to sewn color-block panels and striped rib-knit bands, moved jacket rose to wearer-left chest, added matching wearer-left pants-thigh rose, and retained large upper-back SR. |
| `kids-001` | Kids Colorblock Set — Red/Black | Corrected anatomical sleeve map: black wearer-right sleeve with circular Skyy Rose Collection patch, plain white wearer-left sleeve; locked wearer-left chest/thigh roses and exact front/back panel seams. |
| `kids-002` | Kids Colorblock Set — Purple/Black | Corrected hood to deep purple; locked dusty-pink wearer-right patch sleeve, medium-purple wearer-left sleeve, exact asymmetric panels, and wearer-left chest/thigh roses. |

## Fidelity gate state

- Dossier validator: 35/35 files pass.
- Active dossier coverage: 33/33 SKUs.
- Canonical catalog consistency: 29/29 checks pass.
- Product SOT, SOT-image manifest, collection SOT, V7 cards, WooCommerce sync
  contract, lookbook SOT, and V2 presentation registry were regenerated.
- Changed approved media records were invalidated rather than rebound.
- Product imagery generation may resume only against the current per-SKU
  `product_hash` and the ordered reference evidence in `data/product-sot.json`.
