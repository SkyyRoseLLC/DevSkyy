# Verified Garment Reconstruction / Photography Intake

**Status:** active candidate-only acquisition workstream  
**Branch:** `codex/garment-reconstruction-photography-prep`  
**Founder authorization:** 2026-08-25 — verified garment reconstruction and controlled physical-product photography

This package prepares the exact physical evidence required to resolve the remaining product-source blockers for the seven V2 native collection scenes. It does **not** create a scene candidate, a garment model, a product matte, a composited image, or a deployable asset.

## What the authorization permits

Two verification-first acquisition routes are allowed:

1. **Controlled physical-product photography / photogrammetry.** Capture the exact sellable SKU, configuration, and size with a calibrated colour target and physical scale reference.
2. **Measured 3D reconstruction.** Build an exact garment only from a manufacturer pattern/CAD, verified scan/photogrammetry set, CLO/Marvelous garment file, or founder-approved measured mesh. Reconstruction must preserve the complete evidence chain below.

The routes may produce reference packs and internal reconstruction candidates only. A reconstructed garment does not itself become an approved on-model source or scene asset.

## Non-negotiable source and promotion boundary

- `data/product-sot.json` is pinned by SHA-256 in [`reconstruction-program.json`](./reconstruction-program.json). Any SOT drift invalidates the intake.
- A ghost/mannequin image, a single flat lay, a previous generated image, a semantic redraw, a placeholder, or a rejected preview cannot be promoted to reconstruction authority.
- The previously rejected V3 protected composites and previews remain quarantined. They are neither input nor visual reference.
- No GPT Image 2 or other image generation request, paid external call, Gemini use, scene wiring, deployment, asset upload, or manifest promotion is authorized by this package.
- Every captured/reconstructed file must be SHA-256-bound in its per-SKU receipt. The validator recomputes those hashes from local files.

## Evidence required before a reconstruction candidate can exist

For every SKU, supply one receipt at `evidence/<sku>/capture-receipt.json` and place the hash-bound files below that directory. Each receipt must identify the exact SKU, colorway/configuration, physical size, capture operator, source method, and files.

Start from [`capture-receipt.template.json`](./capture-receipt.template.json), replace every example value, and calculate each digest from the locally retained file. Do not copy the template into `evidence/` unchanged: its example source names and hashes are intentionally invalid.

| Evidence group | Required proof |
| --- | --- |
| Geometry | Calibrated front, left, right, back, top, and bottom views; ruler/known scale in each geometry view; all views of the same physical garment and configuration. |
| Construction | Inside/outside neck or waistband, labels, seams, cuffs/hem, pocket construction, closures, and all hardware. |
| Surface | Neutral-light colour-card frame plus close-ups for print, embroidery, patches, texture/knit/pile, and material transitions. |
| Reconstruction | Source tier; pattern/CAD/scan provenance; mesh, UV, PBR texture, and render review receipts. A source tier below verified scan/pattern/CAD/founder-measured mesh blocks 3D release. |
| Matte / photography | Full-resolution alpha matte generated from the verified source, alpha statistics, clean silhouette inspection, and source/camera/light/floor receipt. |

The detail checklist is garment-specific in `reconstruction-program.json`; it covers the visible logo, text, trim, zipper/pocket, and silhouette risks of each item. The capture set has to show the real garment—not infer unobserved construction from a front product image.

## Seven-scene dependency map

See [`scene-coverage.json`](./scene-coverage.json). It maps each exact SKU to the scene blocker it can resolve after independent verification. It deliberately does not promise a combined look, pose, floor contact, or lighting field from a garment reconstruction alone.

The final native scene path remains:

```text
verified physical evidence
  -> independent garment reconstruction / photography fidelity review
  -> approved same-pose on-model production source
  -> hash-bound garment matte and physical integration proof
  -> candidate-only native scene request
  -> product-fidelity review + native-integration review + founder approval
  -> separate wiring authorization
```

## How to validate

```bash
python3 docs/design/v2-remodel/garment-reconstruction-photography-v1/validate-reconstruction-evidence.py
python3 docs/design/v2-remodel/garment-reconstruction-photography-v1/validate-reconstruction-evidence.py --strict
```

The first command validates the program and reports every missing receipt. `--strict` fails until all 16 SKU receipts and their hash-bound source files have passed. A pass means only that the reconstruction/photography evidence package is complete; it never clears the separate scene, native-integration, founder, wiring, or deployment gates.
