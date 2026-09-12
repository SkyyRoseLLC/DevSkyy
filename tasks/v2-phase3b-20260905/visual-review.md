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

## Checkpoint 2 — first Shop / PLP candidate

Evidence: all ten `shop-{320,360,375,390,414,768,1024,1280,1440,1728}.png`
captures, `shop-full-{390,1440}.png`, `shop-filter-390.png`, and
`shop-responsive.json`, compared with the baseline Shop captures. Full-page images
were inspected for sequencing/rhythm; their downscaled text was cross-checked against
the corresponding viewport captures. SHA-256: shop390
`3b9b5e4a2a1d9ebe635d4d6039fb0627ed5b0985f9624e1a9918fa2a0b7c3bca`;
full1440 `f8d7caeb8e8389e7f9700c93f058d382706f5b2208f11344d617296537c4f643`;
responsive receipt `96cf27f71a42efad0a0fc3de02535fa190dde13308daa7f6b9615970a519f1af`.

**Provisional visual score:80/100. Verdict: REJECT for final Shop visual acceptance;
continue refinement.** This is the Shop snapshot score, not an overall Phase3B verdict.
Native-query security remediation is outside this visual approval and remains a
separate required gate. Dynamic motion/performance acceptance is not established by
still screenshots; the provisional motion/technical points below cannot be converted
into certification without their respective evidence.

| Founder dimension | Score | Reason |
| --- | ---: | --- |
| Brand specificity | 16/20 | Approved garment scenes and archive/nameplate cues are specific; the surrounding layout still behaves like a reskinnable retail grid. Generic `THE HOUSE / 01` adds little provenance. |
| Composition | 10/15 | Excellent reduction in arrival overhead, but sixteen equally treated stacks produce unbroken repetition with no merchandising hierarchy or purposeful narrative pause. Below the70% category gate. |
| Typography | 13/15 | Clear utility/commerce hierarchy and readable full names. Long names produce inconsistent neighboring price/action baselines at narrow widths. |
| Imagery | 13/15 | Garments are fully legible in available space and backgrounds preserve collection identity. Repeated first-eight scene treatments need composition to supply rhythm; do not alter approved imagery merely for variety. |
| Commerce clarity | 12/15 | Filters, native sorting, counts, product identity and prices are much earlier. At320, the full-width tall photograph still pushes product name/price beyond the first844px. |
| Responsive quality | 8/10 | Legacy half-track width collision is fixed; sensible1/2/3/4-column behavior with no observed overflow at all ten widths. The320 first-card height needs refinement. |
| Motion restraint | 4/5 provisional | Static composition does not depend on cinematic ornament; complete reduced/delayed-JS behavior remains a separate evidence gate. |
| Technical finish | 4/5 provisional | Three core widths have zero recorded axe violations, filters are visibly labeled, and all ten widths have zero horizontal overflow. Query/security, complete behavior and final performance are not approved here. |
| **Total** | **80/100 provisional** | Material usability progress, not yet the requested uplift beyond the accepted3A88. |

### What is now demonstrably better

- At390, the first product image starts at measured y363.8 versus approximately y698
  in the baseline. Both first-row garments and their names/prices now appear by the
  bottom of the opening viewport. The artwork no longer arrives as empty ornamental arches.
- The grid uses its available width:171px cards at390,234.7px at768,340px at1440,
  and a bounded363px at1728. The earlier half-empty mobile/tablet layout is resolved.
- All five category choices, including Kids Capsule, remain visible through wrapping;
  no required horizontal navigation is imposed. Native sort controls and explicit
  submit button are understandable. The open filter state has visible labels, active
  count, clear/reset access and strong focus treatment.

### Required refinement

1. **Resolve the monotonous-grid hard-fail concern.** The full-page view contains sixteen
   repeated image/nameplate/name/price/action stacks; the first eight also share the same
   scene treatment. Product comparison is necessary, but the page still lacks an authored
   merchandising sequence. Add one purposeful source-backed editorial pause or hierarchy
   without changing native product order or degrading comparable cards. A clearly labeled
   `From the house` aside using an actual visible product's canonical collection story
   and real world CTA is semantically safer under arbitrary sorting than inventing groups
   or collection counts. It must not imply that following products belong to that world.
   No additional imagery or animation is necessary. Score only after it is rendered.

2. **Recover320 first-view identity.** The288px-wide full portrait occupies roughly432px
   height before nameplate and product metadata; name and price remain beyond the opening.
   A bounded approximately280px `contain` treatment is a reasonable candidate if it keeps
   the entire approved image visible and does not create a false crop/authority claim.
   Verify the actual result rather than treating the measurement as a final design rule.

3. **Keep any new hierarchy from undoing the compact opening.** At1440, prices are already
   near the bottom of the1000px capture. New chapter/provenance content must replace
   overhead or occur later; do not simply append another introductory panel.

4. **Give provenance real meaning.** Existing Oakland/Living Archive language is more
   specific than generic house labeling. Use only verified language and canonical
   collection ordinals; do not invent a new initial/lockup or artificial stock/group totals.

The next capture should show the complete refined page,320 opening,390 comparison row,
and1440 merchandising rhythm. Retain separate native-query/security and performance
gates; this checkpoint authorizes no broader Phase3B PASS or deployment.

## Checkpoint 3 — refined Shop / PLP

**Shop-only visual verdict: PASS,87/100. Proceed to the next local PDP stage.**
The revised Shop clears the85 floor and70% category minimums with no remaining observed
Shop composition blocker. It does **not** materially exceed the accepted3A88-point target;
do not present this as90+ work or as final flagship/Phase3B certification. The complete
phase still requires the remaining surfaces and final comparable performance evidence.

Reviewed evidence: refreshed `shop-refined-{all ten widths}.png`, full390/1440 page
captures, `shop-refined-responsive.json`, `shop-behavior.json`, `shop-static.json`,
`shop-delayed-{390,1440}.png`, `shop-nojs-{390,1440}.png`, and the baseline/candidate
Shop mobile Lighthouse JSON. The fresh320/390/768 screenshots were inspected after the
heading repair. The initial refined capture's split `Sh/op` or `Sho/p` is retained as
a first-seen regression; it is corrected in the refreshed evidence, not erased from history.
Refreshed artifact SHA-256: shop390
`0511d2eb20a5f530e110d69a4a740213da45812b2ab9b6040b83cdac7239b064`;
shop320 `563b53adae4ed0101c2f3d270021e600c1aa561fd1069a05274e6c8edcdcd4fc`;
full1440 `9dc1514232af8aaec9bed8d5dfec933b68cda88182abb81ed56965170306df51`;
responsive receipt `d74847385febec3554f73392693e83a8efe22def06243a0bbc998c6d03ebae38`.

| Founder dimension | Score | Independent assessment |
| --- | ---: | --- |
| Brand specificity | 17/20 | Oakland/Living Archive provenance and a real collection story now support the source-specific imagery and ordinal/nameplate system. It is more authored, though much recognition still comes from the garments/scenes. |
| Composition | 12/15 | A single source-backed aside interrupts the formerly unbroken sequence while preserving product comparison and order. Strong utility architecture; still deliberately restrained rather than exceptional editorial variety. |
| Typography | 13/15 | Heading is intact at320/390; archive/commerce/utility roles are clear. Long native names still create uneven neighboring metadata/action baselines. |
| Imagery | 14/15 | Full garments remain visible; the320 contained image balances identity and purchasing information without cropping the approved front. Responsive derivatives reduce waste. |
| Commerce clarity | 13/15 | At320 the name and actual price now fit near the bottom of the opening;390 shows two clear comparison products. Explicit filter/sort controls and real status remain understandable. |
| Responsive quality | 9/10 | Ten-width geometry has no recorded overflow; all columns use available tracks. Header regression was caught and fixed. |
| Motion restraint | 5/5 | Composition is already meaningful with theme initialization delayed; measured main heights remain unchanged before/after initialization. No cinematic dependency is needed for this Shop. |
| Technical finish | 4/5 | Zero recorded axe violations at390/768/1440 and JS-on/off behavior receipts are positive. Mobile loading is substantially improved but still weak, so no full technical/performance score. |
| **Total** | **87/100** | Local Shop visual gate passed; no phase-wide or launch approval. |

### Why the earlier rejection is resolved

- The `From the house` aside introduces a canonical world ordinal, actual collection
  heading, existing manifesto and real story destination after eight products. It is
  visibly editorial, not a fabricated grouping claim over following merchandise. The
  predictable comparison tracks remain intact and no extra image request is needed.
- The320 photograph now uses a bounded contain presentation. The complete approved
  frame remains visible; product name and price appear by the lower opening viewport.
- The provenance line has its own controlled wrap while `Shop` stays whole. The first
  refined version broke the word; the refreshed capture explicitly closes that defect.
- The unchanged static-main heights recorded for delayed initialization are5719px at390
  and3950px at1440. The no-JS capture visibly uses the accepted in-flow house directory;
  this is a fallback navigation presentation, not a claim that the no-JS first fold has
  the same merchandise placement as the scripted first fold.

### Remaining criticism and boundaries

The first eight products still form a long repeated sequence on mobile before the
editorial pause. This is now acceptable for a comparison-first Shop, but it is not a
reason to reuse the same rhythm for the flagship homepage or all collection pages.
Long names make adjacent prices/actions less precisely aligned. At390, the second
price and selection actions remain near/below the lower viewport edge. These are
nonblocking refinements, not evidence of exceptional commerce composition.

Read directly from local Lighthouse reports: performance65→71, LCP9236.6→5393.5ms,
CLS0.00550→0.00112, accessibility96→100. This is meaningful local recovery, but5.39s
LCP remains poor and is **not** a performance PASS or field-CWV claim. Final source,
build, security/query, broader state and performance certification belong to the lead's
integrated release evidence; their existence must not be inferred from this visual score.

No runtime files were changed by this reviewer. No Phase3B overall approval, payment,
deployment, media promotion or Phase3C permission is granted here.
