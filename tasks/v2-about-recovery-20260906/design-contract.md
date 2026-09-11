# About recovery presentation contract

Date: 2026-09-06. Mode: recovery / presentation-only design-system audit. Status: FOUNDER_REVIEW_REQUIRED; builder handoff BLOCKED pending content inventory and media proof. No theme changes performed by this contract author.

## Authority and scope

The attached founder brief takes priority: recover original stories and interview before changing content. The supplied mockup is a presentation reference, not a source for copy, people, chronology, Fox 2 attribution, generated signature, press credentials, or collection artwork. Its editorial scale contrast, substantial image/text pairings, and chapter pacing may inform composition. Its European serif headline treatment and pervasive gold must not silently replace existing Archivo/Hanken / rose-gold house canon.

Sources inspected: `docs/theme-team-charter.md`; `CLAUDE.md` section 6; `docs/brand/visual-references.md`; both theme.json files; `wordpress-theme/skyyrose-flagship/data/collections/signature/identity.json`; V2 `assets/css/design-tokens.css`, `assets/css/content-page.css`, `functions.php`, `page.php`, `template-parts/v2-about.php`; legacy `template-about.php`, About partials, about.css/about.js. `.wolf/memory.md` was absent. Skills applied: `.claude/skills/design-system/SKILL.md` (audit only), `.claude/skills/luxury-design-taste/SKILL.md`. No external framework/API claims are certified here; this is a repo-grounded design proposal, not an API implementation guide.

Global header/footer including rotating identity, all five heroes, nine scenes, cards, Quick View, and Ask Skyy behavior are frozen. About content must not depend on 3D. No staging/DB writes, uploads, publishing or paid generation.

## Concrete ownership defect

`wordpress-theme/skyyrose-flagship-2/page.php:27` routes About directly to `template-parts/v2-about.php`; the native `the_content()` branch runs only for other page types. Thus editing About page copy does not control this surface. The partial contains replacement narrative plus hardcoded press/interview relationships. Legacy `wordpress-theme/skyyrose-flagship/template-about.php` holds substantive origin paragraphs, founder quote, timeline and community text not preserved in this V2 partial. Presence in legacy code proves historical existence, not necessarily founder authorship or factual truth; recovery inventory determines eligible content.

The V2 interview currently has an iframe with an immediate `src`; `loading=lazy` is proximity loading, not explicit user-intent loading. Both poster and iframe are rendered. Recovery should retain the source link independently of embed success.

## Smallest maintainable architecture

1. Preserve current source and content exports/hashes. Build a local, importable Gutenberg content document from inventory-approved verbatim strings. Do not write it to WordPress automatically. Each block/section maps back to inventory ID and original source.
2. Add an About-only native-content path in `template-parts/v2-about.php` after approved content is available. A thin theme wrapper owns layout. Native Group, Heading, Paragraph, Quote, Image, Media/Text and link blocks own copy, captions and sequence. Keep existing historical fallback until content migration is separately accepted, with deterministic explicit selection to avoid duplicated stories.
3. Expose interview title, outlet, source URL, approved poster/media attachment and context as editor-maintained content. Prefer an editor-maintainable structured block/pattern or small validated field interface; do not require founder PHP edits or raw embed code. A plain source-link representation must remain valid if JavaScript is unavailable. Existing YouTube ID is a recovery lead, not sufficient proof of broadcaster identity.
4. Extend the current route enqueue pattern in `functions.php:359` with an About-only stylesheet/script, reusing design-token/theme dependencies and hash versions. Avoid broad extraction or deletion of shared `content-page.css` and `legacy-world-components.css` in this recovery pass. Scope all new selectors under one About root to avoid cross-route effects. Rebuild .min outputs after any source edits.
5. Full stories already canonical in Journal remain there. About uses source-approved excerpts and canonical links, not duplicated bodies. A missing Journal URL blocks that specific relationship rather than inventing a destination.

## Brand thesis and recognition

An Oakland-rooted family archive told through actual founder language, documented media and editorial photographs. Recognition devices: (1) real Oakland context; (2) founder/daughter relationship only where recovered and approved; (3) concrete-dark ground with one house accent; (4) strong Archivo chapter headings with Hanken reading text; (5) meaningful collection connections using existing approved imagery and routes. No fake archive years or arbitrary chapter counts.

Canonical map: `--sr2-void`, `--sr2-ink`, `--sr2-muted`, `--sr2-line`, `--sr2-accent`/`--sr2-rose`; `--sr2-font-display` Archivo, `--sr2-font-body` Hanken/Inter fallback, `--sr2-font-ui` Anton for sparse utility labels, `--sr2-font-caps` Cinzel only existing engraved-cap roles. Reuse `--sr2-content`, `--sr2-wide`, `--sr2-page-pad`, spacing scale, `--sr2-header`, `--sr2-ease`, existing timing. Do not regenerate tokens or reintroduce Playfair, Cormorant, Bebas or Yellowtail. Collection accents stay within collection-owned artwork rather than coloring every About chapter differently.

## Composition and responsive contract

Derive chapters from recovered content. The documented legacy sequence is a candidate provenance record, not a mandatory fixed new layout. Use a lead portrait/title pairing, substantial reading chapters, featured interview, a restrained publication list and supported collection/Journal destinations. Alternate image scale and text alignment without putting every story in equal cards. No prose truncation on mobile, no fixed-height text clipping, no line-clamp, no horizontal story-only carousel.

At 1440: asymmetric two-column lead, 55–70-character reading measure, intentional full-width media intervals. At 768: reduce side-by-side complexity before narrowing prose; maintain DOM reading order. At 390: one-column sequence with full text, meaningful headings, intrinsic media reservation and clear play/source actions. Do not crop identifying faces, captions or interview lettering merely to force uniform artwork ratios. Editorial originals have their own provenance; product truth rules apply when depicting actual products, not as an automatic disqualification of editorial founder photographs.

## Interview and motion states

Poster state: local approved responsive image with measured intrinsic ratio, lazy below fold; outlet/title/context and source link readable without JavaScript. No iframe src, player SDK, video download or autoplay before intent. If an approved poster is absent, use a typographic source-link presentation rather than an invented TV still.

Intent: keyboard-accessible explicit Play button, discernible name, short loading status. Mount only the allowlisted authoritative embed; move focus intentionally to the available player without trapping the page. Success: maintain source link, title and captions/transcript link where actually available. Error/offline/blocked embed: readable failure status and working original-source link, not a blank permanent rectangle or an endless spinner. Allow repeat attempt without mounting duplicate players. No unauthorized local hosting of TV footage.

Reduced motion / Save-Data: retain still poster, disable decorative reveals/parallax; user may intentionally play interview. Muted decorative media, if any is independently approved, must remain a still by default in reduced motion. All content initially visible without JS; no hidden-text reveal dependencies. No added full-page scrolling controller or timeline animation runtime.

## Verification matrix and content-loss gate

| Surface/state | 390 | 768 | 1440 | Required evidence |
|---|---|---|---|---|
| Old/recovered, current and candidate | capture | capture | capture | Label source/time; never call reconstructed HTML a historic screenshot |
| Full narrative/quotes | full text | full text | full text | Exact text/provenance diff; every inventory item disposition |
| Poster/intent/player | capture | capture | capture | No player network before intent; actual playback where accessible |
| JS disabled/reduced motion | capture | capture | capture | All stories readable; source link usable; static presentation |
| Keyboard/loading/error | test | test | test | Focus, names, status, retry/source fallback |
| Media/crops/links | test | test | test | Real editorial asset IDs; no broken endpoints or distorted subjects |
| Global systems | smoke | smoke | smoke | Rotating shell, Ask Skyy intent, unaffected routes |

WCAG AA text contrast must be measured, not inferred from token name. Touch actions at least 44 CSS px where practical, visible focus, logical h1/h2/h3, descriptive figure captions without redundant alt text, no horizontal overflow at 200% zoom. Performance captures must distinguish local from staging and include LCP/CLS/long tasks, image candidate size and pre-intent third-party requests. Reserve every media box. Record unsupported environment cases UNVERIFIED.

No visual score is assigned before rendering. Hard-fail scan: proposed architecture excludes generic icon-card rows, centered gradient hero, universal cards, invented claims, fabricated scarcity, and mockup-derived family/press identity; rendered scan remains UNVERIFIED. Logo-off verdict UNVERIFIED. Token drift UNVERIFIED for implementation (no tokens edited). Accessibility UNVERIFIED for implementation. Capture paths pending. Independent approver pending; contract author cannot approve own pixels.

Builder handoff: BLOCKED until root's recovery inventory establishes copy/media authority and interview identity. This contract does not authorize source rewrites or staging content migration. Root may prepare an isolated review candidate after recovery evidence is complete, preserving uncertain material in the audit and marking unresolved dependencies honestly.

## Inventory-complete implementation addendum

Root authorized bounded source integration after `archive-inventory.md` and `media-inventory.md` were read. Founder subsequently explicitly confirmed “the blox video is the interview”; the recovered Ja11W-g34Zo identity is now founder-confirmed, and the mockup's Fox 2 label is inapplicable rather than a publication blocker. Poster provenance remains separate; root payload uses the official oEmbed-associated thumbnail without footage rehosting.

Changed only assigned source paths: V2 `template-parts/v2-about.php` (native-content opt-in when serialized core/group has sr2-about-archive class), `functions.php` (About-only stylesheet/script enqueue), `assets/css/about-archive.css`, `assets/js/about-archive.js`. Existing non-opt-in About body remains byte-preserved below the new guard. Existing dirty global-shell/rotation work was not reverted. No .min outputs edited; root owns build.

CSS class interface: outer core/group `.sr2-about-archive`; direct lead group `.sr2-about-lead` accepts copy group plus native image; chapters `.sr2-about-chapter` contain `.sr2-about-reading` groups; native Quote supports `.sr2-about-quote`; `.sr2-about-film` contains native poster image, title/context, a button/group `.sr2-about-play` with authoritative source href and a separate original source link. `.sr2-about-press-list` and `.sr2-about-timeline` support child core/groups; `.sr2-about-worlds` supports core/columns and native images; `.sr2-about-continue` contains native buttons. Text stays fully present at every width. No new font/token system or global selector was added.

Intent controller preserves modified/new-tab navigation. It validates HTTPS YouTube watch/short URLs and 11-character IDs, creates one privacy-enhanced iframe only after activation, keeps source links, reports timeout/error/loading status honestly, and focuses the available iframe. Iframe load is explicitly not interpreted as successful cross-origin playback. Failure visibility of the provider's own cross-origin error UI remains a browser verification requirement.

Source syntax checks: both touched PHP files PASS `php -l`; JS PASS `node --check`. Root owns source review, minification, actual native-block candidate capture and browser/performance verification. No rendered APPROVED verdict is inferred from syntax.

Refinement before root browser testing: the official 4:3 thumbnail is letterboxed with `object-fit: contain` in a reserved 16:9 slot. Intent mounts the iframe over that same slot; it does not append another full media panel. Native figcaption and source fallback remain outside the player overlay. No-JS poster uses the same reserved aspect ratio, and no provider iframe is created until intent. JS syntax rechecked PASS.
