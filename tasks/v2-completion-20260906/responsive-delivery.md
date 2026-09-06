# Responsive delivery without replacing the approved design

Local V2 candidate, 2026-09-06. Original artwork, typography identity, commerce and motion are preserved. This report describes delivery changes, not deployment approval.

## Archivo default-width delivery

The original Archivo variable font is 90,096 bytes and contains weight100–900 plus width62–125. The theme uses its normal width100. A separate derivative freezes only that default width, retaining all weight values, every Unicode cmap subtable, glyph order, family/license identity and line metrics. It is34,976 bytes, saving55,120 bytes (61.18%). Both frontend CSS and editor font-face declarations offer the derivative first and the original file second as the loading fallback. No original font or its original provenance record is replaced, and no upstream revision is invented. Derivative lineage and recipe live beside the derived font; the existing OFL license and copyright attribution remain packaged.

Independent review: Home/Shop390/1440, five weights100/400/600/800/900 and404fallback. All25 measured text font/glyph/geometry records match. All five weight samples are pixel-identical. Five full screenshot pairs are exact; Shop1440 has eight pixels changing by one RGB level in repeated labels, with no visible distinction. This is a bounded rendering PASS, not universal cross-platform pixel identity.

The generator pins fontTools4.59.2 and the actually selected Brotli1.2.0 backend. Six tests cover determinism, nonwriting checks, stale outputs/manifests, source/axis/version mismatch and output/ancestor symlinks. Independent Python review closed the encoder-backend and complete-cmap proof findings.

## Small native archive frames

Four384w WebP derivatives retain the entire approved frame and exact alpha of the proportionally resized source. Source640w files and all higher-resolution authoring assets remain intact.

| Collection | Original640w bytes | Derived384w bytes | Saving |
|---|---:|---:|---:|
| Signature |122,698|70,954|42.17%|
| Black Rose |123,928|64,854|47.67%|
| Love Hurts |137,998|74,912|45.72%|
| Kids Capsule |116,604|68,282|41.44%|

Independent visual review accepted all four at358CSS pixels/DPR1 on dark and light backgrounds. Fine texture softening is most noticeable in Love Hurts and Signature; these are smaller delivery images, not pixel-identical replacements or approval to reduce higher-DPR fidelity. The frame openings, silhouettes, colors and decorative identity remain recognizable and intact.

Only native archive cards offer the new responsive source. Their existing640w `src` remains. Shared `srcset`/`sizes` in the preload and card select384 where it fits the small single-column layout; DPR2 and viewports480px and wider retain640. This works with JavaScript disabled. Editorial and feature cards retain their original delivery.

The PHP resolver requires exact source/output hashes, local paths, dimensions, WebP format and no symlink components below the theme root. Missing or invalid derivatives omit the responsive attributes. Ten request-isolated PHP tests cover valid delivery, missing/altered files, malformed/schema/path/dimension errors and direct/ancestor links. Eight generator tests cover full-frame deterministic encoding, exact resized alpha, source/codec drift, stale outputs and filesystem failure modes. Both source reviews approved after the ancestor-link omission was repaired.

## Verification and limits

Full pinned build and verify passed:86Node regressions, six font generator tests, eight frame generator tests, ten PHP derivative modes, plus all existing PHP/native commerce, media, source, token, translation and generated-asset contracts. Original402 protected assets remain hash-bound by `preservation.json`. Build dependencies and generated/package boundaries include every new delivery file.

Evidence: `.artifacts/v2-completion-20260906/archivo-coverage/`, `frame-384-study/`, `verify-responsive-delivery.log`. Actual integrated browser selection and final repeated Lighthouse results are recorded separately. No performance PASS is inferred from byte savings alone. No deployment, live commerce mutation or paid provider call occurred.

## Reviewed q80 mobile delivery follow-up

The later full-frame WebP quality80 recipe was independently reviewed at358CSS pixels/DPR1 on both dark and light backgrounds for all four collections. All16 q85/q80 comparison sheets were inspected. q80 retains visible apertures, silhouette, ornament structure and material/color identity with slight fine-grain softening most noticeable in Love Hurts. Larger/DPR2 selections remain the approved640w originals. The q90 table above records the previous checkpoint, not current derivative sizes.

Current q80 derivative sizes: Black Rose47,630B; Kids Capsule49,014B; Love Hurts51,128B; Signature49,150B. Total196,922B versus279,002B at q90, a29.42% reduction. The generator retains source hashes, exact resized alpha, complete frame, metadata and pinned Pillow/libwebp; only its reviewed quality constant changes. Runtime selection, decoded dimensions, wire hashes and performance require their separate receipts.

The final optional360w supplement serves the358px mobile slot without changing384w or640w choices. Its four files total176,088B (Black Rose43,492; Kids43,088; Love Hurts46,122; Signature43,386),10.58% below the384w q80 set. All eight dark/light comparison contacts were independently accepted at358CSS pixels. Runtime selection/fallback passed14browser cases with13axe checks, no violations or overflow, and one matching preload/image request. Five negative PHP modes preserve384w/640w when narrow output is invalid.
