# Independent hero and mobile art-direction review

Status: **MATCHED HERO STILL REVIEW COMPLETE — all five heroes EQUIVALENT at the inspected 390/1440 widths; current 320/414/768 compositions also inspected.** The earlier Love Hurts card image finding was withdrawn after original-detail verification.

No browser or heavy task was launched by this reviewer. Root implemented and executed the capture runner; this reviewer independently opened all 10 baseline images and all 25 current images at original detail. The JSON reports record 10 baseline and 25 current successful capture cases. Root's runner now samples actual presented frames at 1.00–1.06 seconds and pauses without seeking; this supersedes the original proposed seek-based capture contract below.

## Required matched hero capture contract

Routes: Home `/`, Signature `/collections/signature/`, Black Rose `/collections/black-rose/`, Love Hurts `/collections/love-hurts/`, Kids/The Heir `/collections/kids-capsule/`.

Current mobile widths: 320, 390, 414, 768. Matched immutable accepted baseline/current widths: 390 and 1440. Use the same viewport height and fixture content for each pair and record them.

Inspect `[data-recovery-hero]`, `[data-recovery-hero-video]`, and `[data-recovery-motion-toggle]`. Wait for the real poster decode and font readiness. Confirm actual native video playback began before using its pause control, then seek to exactly 1 second if duration permits and wait for a decoded/composited frame. Preserve the approved video and poster sources. Record `currentSrc`, `currentTime`, duration, `readyState`, hero rectangle, image/video object-fit/object-position, heading/CTA rectangles, source variant and viewport. Capture both the full viewport and the hero.

Fixed-frame comparisons establish composition at a known point, not motion smoothness. Do not suppress Skyy or mandatory content in shipped source. If diagnostic test freezing is used, record it. Record reduced-motion/no-JS fallback separately if not already covered by current evidence.

## Individual hero assessment

| Hero | Matched normal-motion assessment | Independent observation at 390 and 1440 |
|---|---|---|
| Home | **EQUIVALENT** | Approved moon/bridge/rose-star film, crop, dark text scrim, headline, body copy and purchase/discovery actions retained. The current same-model static guide replaces the baseline capture's original-portrait motion-failure presentation. That is a changed renderer state, not proof that a 3D animation improved. No hero composition degradation seen. |
| Signature | **EQUIVALENT** | Golden Gate horizon, copper monuments, lit platforms and water remain in the same composition. Mobile keeps the center view with monument edges cropped as in baseline; separate collection title/copy and both native CTAs remain legible. Desktop retains the full two-monument setting. |
| Black Rose | **EQUIVALENT** | Silver metal lettering, bridge cables, moon, rose/star relief and dark reflections remain readable at desktop. Mobile retains the bridge/star emphasis and the same partially cropped left monument, with complete collection heading and clear commerce/story actions below. |
| Love Hurts | **EQUIVALENT** | Central aisle, back-turned figure, bell jar, crimson wordmark/star and reflective floor retain their approved positions. Desktop preserves the full environmental composition; mobile retains the same center-weighted crop with readable separate title/copy and CTAs. |
| Kids / The Heir | **EQUIVALENT** | Character face/hair/clothing, throne, warm lighting and The Heir lettering retain the same framing. Mobile and desktop lower-body crop is unchanged from baseline. Separate collection identity and commerce continuation remain readable where within the captured viewport. |

Evidence paths: `.artifacts/v2-cinematic-finalization-20260906/heroes-mobile/{baseline,current}-hero-{home,signature,black-rose,love-hurts,kids}-{390,1440}.png`. Current extra-width evidence uses the same `current-hero-*` naming at 320, 414 and 768. These are composition/legibility assessments of known presented frames, not loop-seam, timing, nine-scene, GPU or physical-device certification.

## Current additional-width art direction

- **320px:** Home typography remains readable and its two main actions stack instead of colliding. Signature, Black Rose and Kids also stack commerce/story actions, while Love Hurts fits both on one row. All five maintain distinct title and body hierarchy. Artwork cropping is substantial at the side monuments, but does not newly truncate the separate product/collection decision text. Home's three-world navigation falls below the shown viewport; this is not an all-content-above-fold claim.
- **414px:** Main hero and character controls remain distinct; all four collection titles and action pairs fit without visible overlap. The additional width exposes more side artwork and allows longer supporting lines. Love Hurts' story heading fits on a single line in this capture. Kids' face/identity remain prominent without the Pause control covering the face.
- **768px:** Collection heroes switch to a wide environmental frame and a divided heading/action layout. Signature and Love Hurts titles stay on one line; Black Rose and Kids use two lines, with their CTAs separated in the right column. These are readable breakpoint transformations, not clipping. Home retains its left text/guide hierarchy over the larger bridge/star field; the star is deliberately cropped at the right edge. No new overlap or illegible decision text was visible.

The inspected 1440px collection viewports devote most of their height to the film, leaving primary collection headings/actions below the first viewport. The accepted baseline does the same. Home's desktop Pause control sits at the lower captured edge in both versions. This review records retained composition rather than silently treating every offscreen control as visible.

## Engine evidence boundary

Root additionally supplied `firefox-heroes.json` with five actual-frame capture passes and `webkit-playback.json` documenting all five preserved-poster plus explicit-play successes. WebKit's initial autoplay path returned `NotAllowedError`; its passing path is **poster/policy fallback plus manual play**, not autoplay success. These artifacts do not establish physical Safari or iOS device performance. This reviewer did not independently launch those engines.

## Existing mobile imagery independently inspected

Paths below are under `.artifacts/v2-cinematic-finalization-20260906/`.

- **Quick View 320:** `commerce/current-quick-view-320.png`. Full selected product figure remains legible. Long product name wraps without overlapping price or options. The subdued unavailable/unchosen Add to Cart state differs clearly from the active continuation action. The panel is vertically long; this still does not prove that its final secondary action is reachable by touch/keyboard. The matched 390 review in `independent-review.md` remains BETTER for the inspected settled state.
- **Directory 320/390:** `commerce/current-navigation-{320,390}.png`. Primary destinations retain a readable single vertical hierarchy. The selected Signature campaign still is large enough to read as a collection preview. Two collection-link columns remain distinct below it. At both widths the collection choices extend below the screenshot; this is a scrollable-directory composition, not an all-options-above-fold claim.
- **Kids card 390:** `commerce/current-card-kids-capsule-390.png`. Complete portrait/frame, readable product name, complete `$65.00`, availability and both native actions. No clipping visible in this image.
- **Jersey card 320:** `commerce/current-card-320.png`. Complete framed portrait, wrapped product identity, price and availability, explicit pre-order edition, and both actions remain visible. This is a card capture, not proof of full-grid rhythm.
- **Skyy chat 390:** `skyy/current-390-chat.png`. Frontal character is retained with visible Pause, Minimize and Close. Transcript, quick responses and input remain readable. The input is near the lower edge of this 900px-high capture; this is not proof at a shorter handset viewport or at 320px.

## Corrected assessment: Love Hurts mobile card

The initial tool-rendered image view appeared to show truncated labels. Root challenged the result, and this reviewer reopened the exact absolute `commerce/current-card-love-hurts-390-recheck.png` at original detail. The actual verified image unambiguously shows complete `$75.00`, `Available`, and `Quick view` labels. Its SHA-256 at verification was `911f891c79136dbd0345a8b222a1439b69d0a8d390486cac42f3290bb557e813`.

The earlier WORSE finding is **withdrawn**. The verified current card is **EQUIVALENT** to the accepted before image for artwork, framing, identity and readable commerce controls. No stored-file or source defect was established. The discrepancy in the earlier image-tool presentation is not assigned a speculative cause.
