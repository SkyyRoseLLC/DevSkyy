# SkyyRose Flagship 2

SkyyRose Flagship 2 is a WooCommerce-first, Oakland-rooted luxury streetwear
theme. The presentation layer combines collection-specific editorial worlds,
accessible commerce controls, and progressive cinematic enhancements while
WooCommerce remains the sole authority for products, variations, price,
inventory, cart, checkout, customer, order, and payment state.

## Install

1. Upload the `skyyrose-flagship-2` folder or packaged ZIP in **Appearance → Themes**.
2. Install and activate WooCommerce before importing the store structure.
3. Activate SkyyRose Flagship 2.
4. Open **Appearance → SkyyRose V2 Setup**.
5. Select **Import SkyyRose V2 demo structure** once.
6. Import or synchronize the authorized product catalog through your separate
   WooCommerce catalog workflow. The theme importer deliberately creates no
   products and invents no inventory.
7. Assign payment, tax, shipping, email, privacy, and policy settings for the
   merchant and jurisdiction before accepting orders.

The importer is idempotent. It reuses exact existing page paths, never deletes
content, never overwrites merchant-authored pages, and never replaces a menu
that already contains items. If WooCommerce is inactive, commerce pages are
skipped with a warning and can be provisioned by running the importer again.

## Imported route structure

- `/` — House homepage
- `/collections/` — collection index
- `/collections/signature/`
- `/collections/black-rose/`
- `/collections/love-hurts/`
- `/collections/kids-capsule/`
- `/pre-order/`, `/about/`, `/contact/`, `/journal/`, `/wishlist/`
- `/faq/`, `/shipping-returns/`, `/size-guide/`
- `/privacy-policy/`, `/terms-of-service/`, `/accessibility/`
- WooCommerce shop, bag, checkout, account, and order-tracking pages

## Product and asset authority

- Product facts and product media resolve from WooCommerce and the authorized
  SkyyRose SOT pipeline. The theme never manufactures a product fallback.
- `data/product-presentation-registry.json` is a generated, non-commercial
  adapter. It may classify a verified SKU into a collection or Jersey Series
  presentation, but contains no prices, stock, WooCommerce IDs, or product
  media.
- Theme-local editorial assets require candidate-bound provenance. A file path
  alone is not proof of product identity or usage rights.

## Build contract

Production uses generated `.min.css` and `.min.js` siblings. Edit source files,
then rebuild and verify byte parity:

```bash
npm install
npm run build
npm run verify
npm run package:theme
```

`npm run verify` checks PHP syntax, JSON, presentation-registry freshness,
minified-asset parity, required marketplace artifacts, placeholder markers,
and retired fonts. Generated minified artifacts must be force-tracked by the
release integrator because the repository’s global ignore policy excludes
`*.min.css` and `*.min.js`; a clean checkout is not release-ready without them.

## Customization

`theme.json` exposes the canonical SkyyRose palette, spacing, and type roles in
the editor. Archivo is the text display face, Hanken Grotesk is body, Anton is
utility, and Cinzel is restricted to ceremonial metadata. Collection names in
cinematic heroes remain approved lockup images rather than type-rendered
wordmarks.

The editor loads `editor-style.css`. WordPress loads `rtl.css` for right-to-left
locales. All front-end motion must preserve native scrolling, keyboard access,
reduced-motion equivalence, and a static failure path.

## Marketplace handoff gates

Before distribution, verify the same frozen candidate at 390, 768, and 1440
pixels; run keyboard and assistive-technology journeys; validate WooCommerce
simple and variable products, cart mutations, checkout failures/recovery,
account ownership, and empty/error states; run Lighthouse; confirm every
product image against its SKU pixels; and review starter policy copy with the
merchant’s legal adviser.

No deployment, catalog write, media upload, or payment configuration is part of
the theme package.
