# Independent staged About and rotating-mark verification

Status: PASS for scoped feature delivery; founder visual review remains required.

Target: https://staging-7e48-skyyrose.wpcomstaging.com/about/

Execution began only after root reported `INSTALLED_VERIFIED_536_AND_ABOUT`, exact 536-file runtime parity and About content hash verification. This browser report does not independently recertify the filesystem. No source, database, staging configuration, cache, orders, payments or chat submissions were modified.

## Rotating header and footer

Eight Chromium browser cases passed: 320, 390, 768, 1024, 1440px normal motion; 390px reduced motion; 390px simulated Save-Data; and 390px intercepted WebM failure.

All ten normal header/footer video instances passed transparent-corner alpha=0, changing decoded pixels across frames, active playback and looping. Correct `skyyrose-logo-optimized-384w.webm` source observed. Header image geometry stayed identical before/after activation. No horizontal overflow was measured. Footer video was absent before approaching the footer, then activated. Reduced motion and Save-Data requested no WebM and created no video. Controlled video failure selected the existing animated WebP fallback.

The brand links preserve the SkyyRose home accessible name. Screenshots show contained header/footer marks at narrow widths, with no clipping or navigation overlap.

Do not interpret this as full-page zero CLS: initial shift observations included 0.00931 at 320px and approximately 0.00655 at 768px, attributed to About copy or unidentified nodes; none identified the logo. Identical logo boxes and zero logo-attributed entries support the scoped geometry result. These are functional observations, not isolated canonical performance runs. Request interception deliberately delayed normal WebM delivery by 700ms to exercise the swap.

Evidence: `.artifacts/v2-consolidated-staging-20260906/about-logo/integrated-validation.json`, eight pairs of `integrated-*-header.png` / `integrated-*-footer.png`.

## About

Five widths passed: 320, 390, 768, 1024, 1440px. The same daughter portrait loads at its 724px native width. All four collection graphics load: Signature founder-supplied SR/rose, Black Rose star, Love Hurts star/heart and founder-supplied Kids/Heir graphic. No hero/scene substitutes appear in this section. Oakland crops remain present. One meaningful H1 reads `The Story of SkyyRose` across widths. No horizontal overflow or page JavaScript errors were recorded. Full About text is identical across viewport records and the retired tagline is absent.

No YouTube interview iframe, Skyy GLB, Three.js or Draco request appeared before interview intent. Clicking the Blox button twice leaves exactly one correct YouTube privacy-enhanced iframe. Actual player text includes Corey Foster's spoken introduction; playback timers advanced to 1–2 seconds at 390/768/1024/1440px. At 320px the sampled player already contained spoken `Hi,` text, without an elapsed timer in the recorded snapshot. No claim of full-length viewing is made.

WordPress.com transforms the official interview thumbnail through `i0.wp.com` at 480×360. This is observed platform image delivery, not a theme source change.

Evidence: `.artifacts/v2-consolidated-staging-20260906/about-logo/new-browser.json`, five initial and five full-page `new-*.png` captures. Visual inspection included the 390px hero, 320px footer and complete 1440px page.

## Platform visual observation

WordPress.com injects a Likes iframe (`widgets.wp.com/likes/master.html`) and sharing/Like controls following native About content. On the 320px footer capture, its white strip contrasts with the dark page. This is separate from the Blox iframe: no interview iframe was present before intent. Surface for founder/platform review; no removal or configuration change was performed. Do not report total iframe count as zero.

## B13 measurement review

Read-only review of the explicit `isHero` flags in the observer and video snapshot, nearest hero-image poster selection, and summarizer selection by `isHero` found no actionable defect in those bounded changes. This correctly avoids treating new rotating logo videos as cinematic hero videos. This is not approval of unrelated measurement code or the eventual performance result.
