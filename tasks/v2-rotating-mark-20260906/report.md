# Rotating header/footer restoration and parity correction

**STAGING FEATURE PARITY: FAIL. LOCAL RESTORATION: BOUNDED CHECKS PASS, NOT DEPLOYED.**

The founder clarified mandatory rotating identity after the prior exact staging deployment. Fresh read-only Home browser checks at320/390/768/1024/1440 show both staging instances render the static WebP. The deployed candidate matches its accepted local source; the source itself omitted rotation. Therefore the prior overall browser/feature PASS is superseded under this contract. Filesystem identity evidence is unaffected.

## Cause and preserved identity

Historical commit f2f01bbcf used data-brand-animation with the unchanged384w animation in both locations and viewport mode in the footer. The global-shell migration in69249828b omitted these hooks. Existing animations were not deleted by this deployment. Independent provenance review confirms the chain, but the original GIF and authoring project remain unlocated.

| Preserved asset | Encoded bytes | Actual canvas | Frames | Cadence | Loop |
|---|---:|---|---:|---|---|
| Historical384 WebP | 1,040,770 |384×216|326|40ms,25fps;13.040s|Infinite|
| Smaller256 WebP |221,526|256×144|109|120ms,8.33fps;13.080s|Infinite|
| Static fallback |5,320|179×134|1|Static|N/A|

Both animations have alpha. They are WebP rather than indexed-palette GIF. The256 derivative is preserved but not substituted: its temporal fidelity is materially different. No new lossy optimization was performed. One decoded384 RGBA canvas is331,776bytes; browser buffering cost is not inferred by multiplying every frame. A local desktop Chromium WebCodecs benchmark decoded the first frame in5.1ms and all326 sequentially in266.2ms. This is a codec diagnostic, not mobile/page animation CPU certification.

## Local repair

Only the shell implementation and its generated assets changed:
- inc/global-shell.php restores both historical384 animation hooks, low fetch priority, footer viewport mode, and existing accessible home-link naming.
- assets/js/theme.js activates after window load, retains reduced-motion/Save-Data static states, reacts to preference changes, defers footer activation near viewport, and restores the fallback after image error.
- assets/css/global-shell.css reserves the existing footer image box with179/134 aspect ratio and contain fitting. The still and animation canvases differ, so the reservation prevents a height jump. Header dimensions remain reserved by existing CSS.
- Corresponding minified JS/CSS rebuilt; all19CSS/13JS generated parity checks pass. PHP lint and10 existing shell-overlay tests pass.

No staging, CDN, database, media-authority, model, original image or approved ZIP changes were made. Local source is now a new unshipped candidate; the previous608-source-file preservation receipt remains historical to deployment completion.

## Browser evidence

Five normal-motion local viewport cases plus reduced-motion, Save-Data and animation-error cases pass. Tests sample actual rendered element screenshots for frame changes, rather than trusting assigned URLs. Both instances rotate in every normal case. Static cases remain static; reduced-motion/Save-Data request no animation. Footer remains still until proximity; its existing box stays95.8125px high. Image boxes are80×48 on small mobile,112×60 at768/1024,144×60 at1440. Full screenshot review found no clipping, distortion, navigation overlap or horizontal overflow. Artwork uses the historical animated canvas and appears smaller than the tightly cropped still; no redesigned scale was introduced. Exact founder scale acceptance remains a visual-review boundary.

Startup local page shifts of approximately0.007–0.063 remain in hero content; these tests do NOT establish pagewide zero CLS or clear the separate staging Home CLS failure. A dedicated controlled settled-page still→animation swap test is recorded separately in swap-cls.json; all10 swaps passed: identical x/y/width/height before and after, with zero layout-shift entries in the controlled swap window. This does not clear page startup CLS.

Animation requests started about593–798ms after local navigation. These are local fixture timings, not new staging performance results. Footer uses the same URL; the local noncacheable fixture recorded another request after proximity. Actual post-restoration CDN/cache behavior must be reverified only after an authorized new deployment.

## Current staging gate

| Location | Rotating present | Working | Correct animated asset selected | Responsive | Required delivery/stability |
|---|---|---|---|---|---|
| Header |FAIL|FAIL|FAIL — still selected|Static shell has no overflow at5widths; rotation unverified|Rotating NO CLS unverified; Home startup CLS remains failed|
| Footer |FAIL|FAIL|FAIL — still selected|Static shell has no overflow at5widths; rotation unverified|Static img lazy; animated near-viewport delivery absent|

Evidence: .artifacts/v2-rotating-mark-20260906/staging.json, local-results.json, asset-metadata.json, decode.json, swap-cls.json and screenshot sets. Initial canvas-based animation test was rejected because canvas image drawing did not demonstrate presentation-frame changes; screenshot sampling is the passing method. Browser scripts are preserved alongside output.

Independent reviews: rotating_mark_review inspected provenance and PHP/CSS/JS; rotating_mark_js_review reviewed the JavaScript lifecycle and harness. No blocking runtime issue found. The JS reviewer correctly identified that initial after-load height checks did not prove zero CLS; dedicated pre/post swap checks address this narrower geometry question. No physical-device or complete release certification is claimed.

Staging remains FAIL until the restored source is packaged, separately authorized, deployed and reverified. No automatic rollback was performed: rolling back this artifact is not proven to restore the full mandatory system and would replace the accepted V2 candidate. The exact prior ZIP, plan and rollback remain preserved.
