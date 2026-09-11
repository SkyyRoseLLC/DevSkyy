# V2 readiness visual freeze review

**Final verdict: EQUIVALENT for all 22 reviewed route/state comparisons, with the two original desktop capture mismatches closed by explicit matched-state follow-up.** Sixteen original pairs were exact pixel matches, four mobile collection arrows were visually equivalent with tiny raster differences, and Home/Quick View desktop now have independently inspected byte-identical matched-state screenshots. This is a bounded visual freeze, not performance, temporal-animation or full release certification.

## Scope and method

Read-only independent inspection of `paired-visual-final/comparison.json`, `results.json`, and screenshots. All six non-identical pairs were viewed side by side in sequence using the native image inspection tool. Representative identical Shop, PDP and Checkout screenshots were also viewed. The remaining identical pairs rely on the supplied exact pixel comparison, not an invented claim that every screenshot received separate human inspection. No browser or source edits were used.

The manifest has 22 pairs at 390 and 1440 widths. Six pairs have non-null pixel bounds; 16 are exactly identical according to comparison.json. All 44 results report zero errors and scrollWidth no greater than viewport width. These captures establish those states and viewports only, not every scroll depth, animation frame, browser or physical device.

## Per-pair result

| Pair | Pixel difference bounds | Verdict |
|---|---|---|
| black-rose-settled-1440 | none | EQUIVALENT — exact captured pixels |
| black-rose-settled-390 | [318, 571, 328, 578] | EQUIVALENT — tiny arrow raster difference |
| cart-settled-1440 | none | EQUIVALENT — exact captured pixels |
| cart-settled-390 | none | EQUIVALENT — exact captured pixels |
| checkout-settled-1440 | none | EQUIVALENT — exact captured pixels |
| checkout-settled-390 | none | EQUIVALENT — exact captured pixels |
| home-settled-1440 | [154, 652, 274, 888] | EQUIVALENT — matched-state closeout below |
| home-settled-390 | none | EQUIVALENT — exact captured pixels |
| kids-settled-1440 | none | EQUIVALENT — exact captured pixels |
| kids-settled-390 | [293, 571, 304, 578] | EQUIVALENT — tiny arrow raster difference |
| love-hurts-settled-1440 | none | EQUIVALENT — exact captured pixels |
| love-hurts-settled-390 | [270, 571, 280, 577] | EQUIVALENT — tiny arrow raster difference |
| pdp-settled-1440 | none | EQUIVALENT — exact captured pixels |
| pdp-settled-390 | none | EQUIVALENT — exact captured pixels |
| shop-filters-1440 | none | EQUIVALENT — exact captured pixels |
| shop-filters-390 | none | EQUIVALENT — exact captured pixels |
| shop-quick-view-1440 | [811, 694, 922, 746] | EQUIVALENT — matched-state closeout below |
| shop-quick-view-390 | none | EQUIVALENT — exact captured pixels |
| shop-settled-1440 | none | EQUIVALENT — exact captured pixels |
| shop-settled-390 | none | EQUIVALENT — exact captured pixels |
| signature-settled-1440 | none | EQUIVALENT — exact captured pixels |
| signature-settled-390 | [299, 596, 310, 603] | EQUIVALENT — tiny arrow raster difference |

## Four mobile collection arrows

Black Rose difference is limited to x318–328/y571–578; Kids x293–304/y571–578; Love Hurts x270–280/y571–577; Signature x299–310/y596–603. Eyes-on inspection found the same downward-arrow meaning, scale, aligned CTA, underline, spacing, reading order and surrounding typography. At displayed viewport size the differences do not degrade legibility, affordance or composition. Verdict: EQUIVALENT, explicitly not pixel-identical.

The root reports independent font-outline and advance-width reproduction for all 11 subset glyphs at weights 100/400/700/900. This review does not independently repeat that font-program proof. The screenshot verdict is consistent with equivalent glyph appearance and limited rasterization differences; it does not invent the precise rasterization cause.

## Original Home desktop mascot mismatch — superseded by closeout

The baseline screenshot visibly shows the static mascot around x154–274/y652–888. The current screenshot does not show her in that region, although the presence status and Ask Skyy/Dismiss controls remain. Hero imagery, headline, body copy, navigation and CTAs otherwise compare identically. This is a real visible difference in the supplied evidence. It may be asynchronous poster/render readiness, but that cause is not demonstrated by these files.

Captured DOM records are not temporally aligned enough to settle it: baseline mascot-stage y is 624.484375, current y is 588.5, while the screenshot difference bounds exclude the status/control area. That discrepancy supports a capture-state issue but does not prove the current mascot eventually appears. Required closeout: capture both versions after the same explicit character/poster-ready condition, using the intended reduced-motion state, and include readiness, image complete/naturalWidth, computed visibility/opacity and screenshot taken at that state. Preserve the original mismatched evidence. Until then this pair is UNVERIFIED for visual freeze, not a confirmed permanent mascot regression.

## Original Shop Quick View desktop state mismatch — superseded by closeout

The only pixel difference is the Add to Cart button rectangle x811–922/y694–746. Baseline shows a subdued outline/gray label; current shows rose fill with black label. Both screenshots show “Choose an option” in the size control. Product image, price, text, modal framing, quantity and other controls remain the same.

The captured button DOM records have identical class (`single_add_to_cart_button button alt`), dimensions, font metrics and computed text color, despite the visible button colors differing. The records do not include disabled/aria-disabled, selected variation, background, opacity or an atomic screenshot timestamp. Therefore this set cannot distinguish transition/loading timing from a selection-state presentation regression. Required closeout: repeat at explicitly matched native variation state, record selected attributes/variation ID, disabled and aria-disabled state and computed background/opacity, then capture. Confirm a missing selection still cannot purchase through the native form. Until then the pair is UNVERIFIED for visual freeze; a commerce failure is not inferred from button color alone.

## Protected media and limits

The root reports 397 protected media hashes unchanged, covering the accepted visual systems. That provenance evidence supports retention but is separate from this screenshot review. This set covers Home plus four collection arrivals, merchandise cards, PDP, Cart, Checkout, filters and Quick View; it is not independent temporal review of all nine Scroll World films or every hero frame. No media redesign is requested.

Keep the original mismatched evidence and diagnosis as an audit trail. The matched-state evidence below closes both requested confirmations. This review modifies only this document.


## Final matched-state closeout — independently reviewed

All eight follow-up screenshots were opened with `view_image`. The four baseline/current pairs visibly preserve the same mascot identity/position and Quick View presentation. Independent file-byte equality and SHA256 checks also confirm exact PNG equality per pair; this is stronger than relying only on the supplied null pixel boxes. `matched-states/comparison.json` reports stable before/after DOM snapshots and zero errors for all four states.

- **Home 1440:** Both versions show the decoded static mascot poster with natural width 330 and opacity 1. Dataset state is show/static/reduced, visibility visible, chat closed, motion paused; canvas is hidden. Stage rectangle matches exactly at x57.59375, y624.484375, width448, height288. The prior absent-character frame is not reproduced at this explicit ready state. Visual closeout: EQUIVALENT.
- **Quick View empty:** Both show no size, variation 0, `aria-disabled=true`, native `disabled wc-variation-selection-needed` classes, background rgb(22,22,22), border rgb(128,128,128), opacity1. The native DOM `disabled` property is false in both; this report does not falsely claim an HTML-disabled button. State representation and visual closeout: EQUIVALENT.
- **Quick View selected:** Both select M, variation44, `aria-disabled=false`, standard purchase class and rose rgb(183,110,121) background/border. Visual closeout: EQUIVALENT.
- **Quick View reset:** Both return to empty size/variation, `aria-disabled=true`, native selection-needed classes and dark button background. The focused select and button presentation match. Visual closeout: EQUIVALENT.

These screenshots prove matched presentation and the recorded native states; separate commerce tests remain responsible for purchase rejection/acceptance behavior. The original screenshots were taken at unmatched readiness/control presentation and should not be used as a settled-state regression verdict. The exact original timing sequence is not reconstructed here.

| Matched state | Identical baseline/current PNG SHA256 |
|---|---|
| home | `60ea64b0a2bec6962423ce3a748af4e729b5ce84e042b168a63938c69a95ed1a` |
| qv-empty | `76eee019f03740d14810c5874b924d9b7a465bf6cd7798a13ba1f7d5b0ed3887` |
| qv-selected | `9e158bfe64cec757b794516b8aad5a918c44ad72b6b0004e117135f6468e4735` |
| qv-reset | `c942e9f8d861f6c412144a1138f83029325e0bcd278954e21fea7086fcdec5c3` |

Four tiny mobile arrow raster differences remain explicitly non-identical; there is no claim that the original 22 PNG pairs are all byte- or pixel-identical. The final equivalence verdict combines original evidence with the matched-state closeout, retaining the same breakpoint, browser, scroll-depth and temporal-review limits stated above.

## Final fresh-capture addendum — 2026-09-06

Independent eyes-on review covered all 26 fresh PNGs: ten `browser-final/current-hero-{home,signature,black-rose,love-hurts,kids}-{390,1440}.png`, eight `skyy-review/{poster,walk,idle,chat}-{390,1440}.png`, and eight baseline/current `transaction-states-final` PNGs. This addendum preserves the preceding 22-comparison equivalence verdict. No browser, capture, source change, or performance run was performed for this review.

The five fresh route heroes retain the branded artwork and readable navigation. Mobile Home preserves the headline, copy, both primary actions, mascot and Ask/Dismiss controls without a visible primary-control collision. Mobile collection captures retain the collection title, action and handoff copy, including arrows. Desktop collection art occupies the large intended hero area, with further content below this viewport. Kids captures show a moving-character video frame rather than the seated poster; a different playback frame alone is not evidence of replaced media. Desktop Home's Pause motion control meets the bottom edge of the 900-pixel screenshot; this viewport crop does not establish a document clipping defect. These current-only stills supplement, rather than replace, the prior paired evidence and protected-media hash receipt.

Skyy's poster, walk-labelled and idle-labelled captures at both widths show the character at a similar location and scale, with pose changes and the Pause character control appearing in the live states. Chat presents the character inside the dialog, readable response text, suggested actions, focused question field, Ask, minimize and close controls. Mobile controls fit within the captured dialog. The desktop chat transcript shows a partly clipped earlier line at its upper scroll boundary below the guide copy, while mobile displays the latest response; these are candid limits of the captured transcript state, not evidence that every conversation line is simultaneously visible. The final reply and contact link are readable. No temporal smoothness, full walk cycle or rig-quality certification is inferred from these stills.

`skyy-review/recordings.json` reports zero captured errors at both widths and final state `renderer=3d`, `presence=live`, `chat=open`, `location=dialog`, `visibility=visible`, `motionPaused=false`, `actionPhase=idle`, `conversation=talking`. Its `modelRequests=0` counter is not accepted as proof of zero network model loads or a one-request guarantee: the request predicate may not match the served URL. Separate runtime profiles own model-request evidence. The WebM files are recording artifacts; this addendum's eyes-on conclusions are based on the PNG frames and recorded state, not a frame-by-frame video review.

All four fresh transaction pairs are visually equivalent and independently verified byte-identical. Cart retains quantity 2, the invalid-coupon error, item image/title, pricing, quantity/removal and coupon controls. Desktop retains the totals panel and checkout action; mobile totals continue below the captured viewport. Checkout retains the same responsive billing layout and coupon notice; desktop shows the same order summary and fixture notice that no payment methods are available. That notice limits this evidence to checkout presentation/review, not completed payment processing. `transaction-states-final/results.json` records zero errors, matching invalid-coupon notice and quantity, plus no reported checkout violations or horizontal overflow for both modes and widths.

| Fresh transaction pair | Identical baseline/current PNG SHA256 |
|---|---|
| Cart invalid coupon, 390 | `5c7c2d9845a44378523482a7a6d975595db74ed79e7e74cbd4b30886f4515421` |
| Cart invalid coupon, 1440 | `2d60cb87e7adceb3189d4579a1106e08241ed5b63b40ea20c6a010194124832a` |
| Checkout review, 390 | `cc4164bc9965fd11f330552345f06a343860b3e79914aa60368b85e1e20b146c` |
| Checkout review, 1440 | `db77c1a8dadc7fdab40517e2f82c6db558bb719090b0ddc214aa459329963c92` |

Final bounded assessment: no new visual regression is established by these fresh captures. Transaction equivalence is exact for the four supplied pairs; active video/character states are visually coherent in the reviewed frames and are not claimed pixel-identical across time. Performance, field INP, GPU behavior, payment-provider readiness and unreviewed browser/device combinations remain outside this visual approval.
