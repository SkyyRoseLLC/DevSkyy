#!/usr/bin/env node
/** Build a file://-safe gallery manifest from the immutable product SOT. */

import { readFile, writeFile } from 'node:fs/promises';
import { resolve, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';

const galleryDir = resolve(fileURLToPath(new URL('.', import.meta.url)));
const root = resolve(galleryDir, '../..');
const sourcePath = resolve(root, 'data/product-sot.json');
const outputPath = resolve(galleryDir, 'product-asset-manifest.js');
const labels = {
  packshot_front: 'Physical front',
  packshot_back: 'Physical back',
  on_model_front: 'On-model front',
  on_model_back: 'On-model back',
};

const source = JSON.parse(await readFile(sourcePath, 'utf8'));
const products = [];
const images = {};

for (const [sku, product] of Object.entries(source.products)) {
  const identity = product.identity ?? {};
  const collection = identity.collection ?? 'unclassified';
  products.push({
    id: sku,
    name: identity.name ?? sku,
    col: collection,
    desc: identity.description ?? '',
  });
  images[sku] = Object.entries(product.media ?? {})
    .filter(([, media]) => media && typeof media.path === 'string')
    .map(([view, media]) => {
      const asset = resolve(root, media.path);
      const assetPath = relative(galleryDir, asset).split(sep).join('/');
      const repositoryPath = relative(root, asset);
      if (repositoryPath.startsWith(`..${sep}`) || repositoryPath === '..') {
        throw new Error(`SOT media path escapes repository: ${sku}/${view}`);
      }
      return { src: `./${assetPath}`, label: labels[view] ?? view };
    });
}

const manifest = `/* Generated from data/product-sot.json. Do not edit by hand. */\n` +
  `window.SKYROSE_PRODUCT_GALLERY = Object.freeze(${JSON.stringify({ products, images }, null, 2)});\n`;
await writeFile(outputPath, manifest, 'utf8');
console.log(`Wrote ${products.length} products to ${outputPath}`);
