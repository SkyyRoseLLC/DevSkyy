# Phase 3B independent visual review

Reviewer: independent `/root/visual_review`; author/builder separation retained.
Ownership: this review record only. No runtime, catalog, media, database, deployment,
or payment changes. Review applies the founder attachment and active design contract.

## Checkpoint 0 — accepted Phase 3A page baseline

Starting runtime: `75ce80b90380d0e5915be79dab33484eda6d3fca`.
Evidence/report baseline: `06f68bd11a08aeab608b6519f2f80a5709476f9b`.

Eyes-on evidence: `.artifacts/v2-phase3b-20260905/baseline-{shop,pdp,signature,home}-{390,1440}.png`.
These are initial viewport captures, not complete page/card/state reviews. Pixel positions
below are approximate visual readings, not browser geometry measurements. No candidate
score, approval, performance result, or media-authority change is implied by this baseline.

### Concrete findings and implementation direction

1. **Shop delays the actual merchandise.** At390, the first frame starts around y698,
   after the title, generous empty space, collection strip, breadcrumb, result count,
   and sorting. No product name or price is visible in the844px opening. At1440, the
   first row begins around y650; the opening sells four repeated arches before visible
   garments. Compress the introduction and consolidate the collection/filter/result/sort
   region. The initial mobile frame should show a meaningful garment and its commerce
   identity, not merely the top of a decorative border. Keep native sorting/filter URLs.

2. **Card recognition must survive the photographic correction.** The collection arch
   silhouette, material treatment and inscription/nameplate belong to the existing
   accepted visual language. Their repeated ornamental mass currently dominates the
   visible Shop strip. Give the approved garment image most of the useful card area and
   retain a subordinate, readable archive/frame/nameplate cue. Do not solve this with
   interchangeable bare photo rectangles, generic hover zoom, or an unverified source
   image. Keep the same authored rule on Shop, collections, related products and Home.
   These opening captures do not show complete cards, so detailed crop fidelity and
   final nameplate legibility remain UNVERIFIED until complete candidate cards are shown.

3. **PDP purchase hierarchy is the strongest immediate composition defect.** At390,
   pre-order explanation and availability copy precede a gallery beginning around y325;
   the entire opening lacks product title, price, size selector and purchase action.
   At1440, the purchase column spends its top on collection identity, emblem, house
   note and fit link; the product title only starts near y780. Move name, collection,
   actual price and native purchase controls together near the opening. Place fit help
   beside that cluster and supported story after it. Preserve truthful pre-order text
   near purchasing, but stop presenting it as the page's main arrival. A bounded mobile
   primary image should allow identity/price and the native buying path to appear near
   the first viewport; do not replace native semantics with a decorative duplicate CTA.

4. **Signature loses its strongest identity on mobile.** The1440 capture has recognizable
   monumental lockups and bridge geography, but nearly the entire initial viewport is
   scenic media without a visible collection title or Shop entry. At390 the crop removes
   both major lockups, leaving mostly clouds, city and water; collection text only begins
   near the bottom. Use a deliberate mobile image composition with visible context plus
   adjacent collection identity and direct commerce access. Preserve verified landmark
   and lockup authority; do not fabricate a mobile alternate or type-render the script.

5. **Home has a recognizable premise but incomplete opening orientation.** The concrete
   line, bridge/rose monument and family copy communicate identity. The collection index
   visible at the bottom presents only three worlds, with Kids absent from that initial
   orientation. On mobile the crop removes much of the distinctive rose monument while
   copy remains legible. The later Home composition should expose all four worlds as
   peers in the house, retain provenance through existing content, and use mature card
   and collection modules. Do not compensate for weak mobile crops with heavier motion.

### First card / Shop / PDP candidate evidence required

- Complete cards at390 and1440, including image, full name, collection/nameplate,
  price, real status and action;768 for tablet transformation. Include both a dense
  multi-card view and a full individual card.
- Show actual image decode/readiness and source context; preserve the distinct accepted
  card-front contract and PDP editorial-media states. Filenames alone do not prove pixels.
- Shop opening plus complete first product row, filter drawer, active filters/sort,
  empty result and reset. Demonstrate GET/refresh/history behavior separately.
- PDP opening at390 and1440 with native identity/price/size/purchase visible or immediately
  adjacent, followed by the supporting story/media. Include approved/stale/missing/rejected
  media behavior without promoting authority.
- Compare motion-disabled and delayed/no-JS presentation, keyboard/focus, controls,
  overflow, contrast and mobile media payload. Missing evidence remains UNVERIFIED.

### Scoring contract for subsequent candidate checkpoints

Use exactly: brand specificity20; composition15; typography15; imagery15;
commerce clarity15; responsive quality10; motion restraint5; technical finish5.
The Phase3A88/100 score is a reference, not an automatic entitlement to90+.
Review each flagship surface independently; no candidate PASS before required evidence.
No final Phase3B or launch verdict is issued at this checkpoint.

## Checkpoint 1 — canonical card candidate

Evidence inspected: `card-{390,768,1440}.png`, `card-grid-{390,768,1440}.png`,
and `card-browser.json` in `.artifacts/v2-phase3b-20260905/`. The surrounding Shop
introduction, sorting region and legacy loop geometry are not a new PLP candidate.
The candidate is uncommitted at review time; Git HEAD is the accepted evidence baseline
`06f68bd11a08aeab608b6519f2f80a5709476f9b`, not the card implementation identity.
Reviewed artifact SHA-256: card390
`715825ad86dee20ab855120aee7571cf232ae972a58fb5c1abb720b37134ec3c`;
card1440 `3d80def24284e15349e69e9bba65c6afdb4fc15a03a292b74970e897904fa7b0`;
browser receipt `fb370db8c7da9c796cf12fc10c22f0a081762315713d67c8c3a47ae146cd8c55`.

**Decision: proceed to PLP integration with this card direction. Final responsive-grid
acceptance remains pending.** No flagship-wide numerical score or final PASS is issued.

### What improved

- Actual garments now dominate the image instead of appearing behind repeated oversized
  ornamental arches. The approved photographic scene still contains architectural,
  bridge and rose context; this is not an interchangeable blank-background retail tile.
- The horizontal rules, ordinal marker and bold single-line series band retain a clear
  archive/nameplate cue without another competing ornamental image. The scene and band
  together carry recognition; neither the logo nor new decorative animation is required.
- Product names, real prices, status and native selection action have a clear reading
  sequence. At1440 the four-card comparison is materially more useful than the baseline:
  silhouettes and colorway differences are visible, with price and action immediately below.
- The full 390 card shows the complete garment and legible commerce facts. Quick view is
  secondary to selection, as it should be for variable products.
- Browser evidence records zero card/loop axe violations and no listed errors at each
  of390/768/1440, successful Quick View focus behavior, exactly two eager images and one
  high-priority image per tested placement. Selected image sources are responsive derived
  assets, not full originals. Offscreen undecoded lazy images are not counted as failures
  or as visually verified content.

### Criticism / required PLP integration check

1. **Resolve the legacy loop-width collision before accepting the PLP.** The390 grid
   capture displays one approximately172px-wide card while the right half of the page
   is empty. At768, approximately165px-wide cards occupy tracks separated by very large
   gaps. The standalone card is readable, but this geometry wastes available comparison
   space and makes tablet/mobile merchandise unnecessarily small. Set one owner for
   column width/gap; do not retain a legacy percentage width inside new grid tracks.
   At390, either use two fully occupied readable tracks or a genuinely full-width card;
   at narrower widths choose an intentional transformation, not accidental empty columns.

2. **Tighten the mobile facts-to-action rhythm.** In the390/768 individual card captures,
   roughly70–80px separates price/status from the selection action. Once the final PLP
   determines row alignment, remove excess minimum-height/padding for mobile stacks.
   Equal action baselines are useful in a comparison row, but should not become large
   empty blocks in a narrow isolated card.

3. **Check the full collection range next.** Current eyes-on evidence concentrates on
   Black Rose jersey imagery. Signature, Love Hurts and Kids nameplate lengths, accent
   treatment, child/adult composition and unavailable/missing states remain unreviewed.
   Do not generalize this checkpoint into approval of unseen cards or new media authority.

The component is sufficiently improved to be consumed by the next local PLP iteration.
Final acceptance still requires candidate-bound responsive composition, remaining card
states, source/authority tests and broader performance verification. No runtime changes
were made by the reviewer.
