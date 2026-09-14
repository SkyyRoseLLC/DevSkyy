# SkyyRose V2 — visual three-way recovery inventory

**Status: HISTORICAL THREE-WAY INVENTORY. Current integration authority comes from the founder directive and explicit file allowlist, not the recovery recommendations below.**

The founder rejected the Phase 3B visual direction and the subsequent creative-elevation proposal. That proposal is superseded; it is not a recovery target. This inventory uses the recovered staging work itself as visual evidence and preserves the repaired commerce foundation.

## Current founder approval mapping

The latest mandatory-systems directive (`1b40f8ab-1e7e-4841-8a8e-e7c8f70efe4b/pasted-text.txt`) supersedes blanket restoration and the earlier selective-system wording. Animated heroes, scroll-world collection CTA scenes, Ask Skyy, and the paid product-card system are mandatory. Their exact approved media, current commerce truth, accessibility, and file allowlist still govern each implementation. A mandatory system does not authorize every historical variant.

Town Line is **PRESERVED FOR PRE-ORDER IMPLEMENTATION**. Preserve legitimate assets, code and manifests; do not implement its final experience in this integration. Other ambiguous recovered variants remain **HOLD FOR FOUNDER REVIEW**. The historical `classification` fields below retain source-analysis meaning only; `current_approval_mapping` in the JSON distinguishes present authority. Runtime edits are separately recorded in [founder-visual-allowlist.json](founder-visual-allowlist.json), with pre-integration hashes captured by root before implementation.

Ask Skyy's approved current source is `assets/models/skyy-mascot.glb`: one 18-joint skin, embedded texture images, and six populated clips named `Skyy_Idle`, `Skyy_Walk`, `Skyy_Wave`, `Skyy_Talk`, `Skyy_Joy`, and `Skyy_Exit` (54 channels each). The current canonical fallback is `assets/sot/images/mascot/skyy-canonical-v2-512w.webp`, resolved through the SOT URI helper. The implementation ownership is the five allowlisted mascot PHP/CSS/JS files; root owns local Three dependencies, configuration, build and integrity pins. Local source checks are not rendered/founder approval.

## Evidence states

- **A — recovered staging:** evidence commit `bb0e6eecf0af54b3f0c14e5eadb7b138530ce09b`; archive SHA-256 `3850af0ae0caa20a49e8ec4f029ef1543be4ac5ef65202b2a1a0336293f09f4a`; exact extracted theme at `.artifacts/v2-visual-recovery-20260905/recovered/skyyrose-flagship-2/`.
- **C — repaired Phase 2:** `5de8e2f3eb40827a052996f72bfb290a95bd600a`.
- **B — repaired Phase 3B:** `fec4e9339cad7077bcc812876668c128eb93b6e4`.
- **D — recovery candidate:** current scoped implementation is tracked separately in the root allowlist/ledger. This inventory compares A/C/B only and claims no D pixel approval or score.

Comparison is from exact A files and Git objects for C/B, not current working-tree guesses. See [visualsystems.json](visualsystems.json) for all 32 system records and all 447 recovered paths, their Git blob identities, both transition results, and C/B release classifications. Git blob SHA-1 is a content identity here; the recovered archive's separate SHA-256 remains the archive authority. The required `.wolf/memory.md` is absent from this checkout.

**Source wiring is not proof that a script initialized, an image resolved, or a video played on staging.** Root owns fresh A/B captures and runtime diagnosis. All rows have rendered verification pending. References below are theme-relative unless explicitly under `tools/`. Database content, plugin output, upload attachments and files outside the 447-file theme archive need separate runtime/media evidence. Do not classify their absence from this archive as deletion.

## What the file comparison actually proves

| Finding | Verified count / consequence |
| --- | --- |
| Recovered paths | 447 |
| Byte-identical A→C paths | 406 |
| Byte-identical A→B paths | 394 |
| Recovered paths absent from B source | **0** |
| Recovered paths included in B release allowlist | 374 |
| Recovered paths source-only in B | 73 |
| Release-classification changes C→B among recovered paths | **0** |

There was significant visual removal, but it cannot accurately be described as 447-file asset deletion. The dominant mechanisms are replacement of active markup, early return to a different collection renderer, removal of scene-specific CSS and controllers, and exclusion of retained systems from route loading. All 73 recovered source-only classifications already existed by C. This does not prove that every A referenced media file exists: a template may depend on uploads or external/shared SOT content outside the recovered archive.

Of the source-only paths, 21 are asset-side entries: four PNG scene authoring masters; six collection-hero MP4 masters; QA contact sheets/metadata and a replica review manifest. The remaining 52 are internal documents/scripts. The exact list and reasons are in JSON. Recovered delivery variants remain separate from masters. Restore a verified delivery dependency if needed; do not indiscriminately put authoring/QA material in the runtime package.

## Phase attribution

**A→C:** Home's scene, filmstrip, opening portals, Royal Procession, featured spread and Town Line markup remain. Its diff changes fulfillment/reservation language. Phase 2 introduces PDP commerce-media rejection/schema/variation guards, scoped restoration, a compact missing-image state, search grouping and overlay/mascot safety. Mobile mascot proactive entrance is suppressed and its positioning becomes non-obstructive. The retired Cormorant fallback is removed from hero-commerce scene CSS. These are not all Phase 3B edits.

**C→B spans Phase 3A and 3B:** shell commit `69249828b` replaces the visual navigation/footer and adds the shared accessible interface; card commit `55807b574` replaces the statue/nameplate presentation; Shop commit `03947e829` changes the archive; `67b2cd856` replaces the PDP and main collection experience; `fec4e9339` replaces Home and performs the final orphan/style/controller cleanup. The table's transitions distinguish these from earlier repairs. A compiled-only change is not assumed to mean a visual redesign when its source stayed identical.

## System-by-system reconciliation

The founder's combined classification **RESTORE VISUALLY / REIMPLEMENT TECHNICALLY** is kept as one label. Decisions below are recovery recommendations, not completed work or permission to copy whole files. Motion labels are provisional source assessments; no system is called SAFE TO RESTORE without runtime verification.

### 01. Home Bay Bridge scene, atmosphere, veil and overlaid headline

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY** · Motion: **RESTORE WITH REFACTOR**. State: `markup_replaced_and_css_controller_removed`.

- **A→C:** A rendered layered scene and foreground copy. C retains structure; only commerce wording changes elsewhere.
- **C→B:** C→B fec4e9339 replaces full composition with .sr2-archive-arrival, detached image and SKYYROSE masthead; old scene CSS and headline controller removed.
- **Evidence:** `front-page.php: .sr-house-hero`; `assets/css/theme.css: .sr-house-hero__scene/__atmosphere/__veil`; `assets/js/theme.js: [data-hero-headline]`.
- **Recovery constraint:** Recover exact source image/layer relationships around current accessible shell and critical-image delivery; do not copy global old theme CSS.

### 02. Three opening on-model looks and moving filmstrip

**RESTORE + REFACTOR** · Motion: **RESTORE WITH REFACTOR**. State: `unwired_and_controller_removed`.

- **A→C:** A/C show SG005 BR004 LH004 scenes as editorial links and duplicated aria-hidden loop copies.
- **C→B:** C→B fec4e9339 removes markup and heroModelLoop controller; sources persist.
- **Evidence:** `front-page.php: $hero_models/.sr-house-filmstrip`; `assets/js/theme.js: heroModelLoop`; `assets/css/theme.css: .sr-home__hero-model-track`.
- **Recovery constraint:** Restore editorial use within its own authority; reduced motion/Save-Data yields static visible originals; no duplicate eager transfers or fake PDP approval.

### 03. Opening three product portals

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY**. State: `markup_replaced`.

- **A→C:** A/C render Signature, Black Rose and Love Hurts product portals.
- **C→B:** Card replaced at 55807b574; Home three-product portal row removed at fec4e9339.
- **Evidence:** `front-page.php: $opening_cast/.sr-house-portals`; `template-parts/commerce/product-card.php`.
- **Recovery constraint:** Recreate collection-owned staging with current canonical card truth, derivative and finally protections.

### 04. Kids Royal Procession: Invitation, Guardians, Heir

**RESTORE + REFACTOR** · Motion: **RESTORE WITH REFACTOR**. State: `retained_sources_unwired_css_removed`.

- **A→C:** A/C Home explicitly invokes three-panel procession; PHP/JS byte-identical A/C/B.
- **C→B:** C→B fec4e9339 removes Home invocation, conditional JS enqueue and 103 .sr-kids-procession occurrences from theme CSS.
- **Evidence:** `template-parts/home/kids-capsule-reveal.php: data-house-royal-procession`; `assets/js/kids-capsule-reveal.js`; `assets/css/theme.css: .sr-kids-procession`; `functions.php: skyyrose2_assets`.
- **Recovery constraint:** Recover three chapter layering, portrait/product dialogue and native chapter controls; preserve real product facts and mobile/keyboard access.

### 05. Signature featured-product threshold and secondary detail images

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY**. State: `markup_and_css_replaced`.

- **A→C:** A/C contain vertical Signature label, large primary and two smaller detail attachments. C changes fulfillment copy only.
- **C→B:** C→B fec4e9339 replaces with one shared SG005 card and story.
- **Evidence:** `front-page.php: .sr-house-featured__vertical/__hero/__summary/__details`.
- **Recovery constraint:** Restore spread around permission-aware current delivery; A raw get_image_id/get_gallery_image_ids is not safe to restore.

### 06. Father/daughter legacy portrait composition

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY**. State: `restyled_not_asset_loss`.

- **A→C:** A/C contain founder/daughter portrait and promise text.
- **C→B:** C→B preserves image/text but changes scale, surrounding sequence and composition.
- **Evidence:** `front-page.php: .sr-house-legacy versus .sr2-archive-legacy`.
- **Recovery constraint:** Use A/B capture to recover intended intimacy, portrait relationship and depth; retain accurate links/alt.

### 07. Pre-order confidence block

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY**. State: `visual_block_removed_claims_repaired_earlier`.

- **A→C:** A contains reservation/fulfillment promises; C corrects unsupported claims while retaining block.
- **C→B:** C→B Home removes block.
- **Evidence:** `front-page.php: .sr-house-confidence`; `template-parts/v2-preorder.php`.
- **Recovery constraint:** Recover useful staging only if founder wants it; keep C/B truthful standard-payment/shipping wording.

### 08. Navigation collection-monument reel

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY** · Motion: **RESTORE WITH REFACTOR**. State: `markup_css_controller_removed`.

- **A→C:** A/C include four frames, prev/next/pause/count/progress and 6500ms intent.
- **C→B:** C→B shell commit 69249828b replaces markup; initHeaderReel later removed from remaining film controller.
- **Evidence:** `functions.php: skyyrose2_header/skyyrose2_header_world_frames`; `assets/js/house-of-roses-motion.js: initHeaderReel`; `assets/css/theme.css: .sr2-header-worlds`.
- **Recovery constraint:** Recover image-led navigation inside B overlay manager; current inert, focus return, Escape, scroll lock and current-page semantics remain owner.

### 09. Animated header/footer brand artwork

**RESTORE + REFACTOR** · Motion: **RESTORE WITH REFACTOR**. State: `unwired_controller_removed_assets_retained`.

- **A→C:** A/C swap still to animated WebP; footer waits for intersection; Save-Data/reduced motion use still.
- **C→B:** C→B new global shell drops animation attrs; theme.js loader removed.
- **Evidence:** `functions.php: skyyrose2_header/skyyrose2_footer data-brand-animation`; `assets/js/theme.js: loadAnimation`.
- **Recovery constraint:** Restore exact existing animation only with viewport/preferences gating and fixed geometry; no font recreation.

### 10. Header conceal-on-scroll

**KEEP CURRENT**. State: `behavior_repaired`.

- **A→C:** A autohides header; C adds overlay/focus repairs.
- **C→B:** C→B removes autohide in accessible shell while retaining scrolled styling.
- **Evidence:** `assets/js/theme.js: previousY/header is-hidden`; `assets/css/theme.css: .sr2-header.is-hidden`.
- **Recovery constraint:** Preserve current reachable navigation and no focus disappearance; recovered appearance can use background/scrolled state.

### 11. Approved statue frame, adult top crop and inscription

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY**. State: `visual_staging_removed`.

- **A→C:** A/C have full-height statue layer; photo window top8%, left17%, width66%, height64%; eligible adult tops scale1.5 from top; collection-specific engraved nameplate.
- **C→B:** C→B 55807b574 replaces with full 2:3 static photo, normal-flow index/SKU/nameplate; A card PHP unchanged through C.
- **Evidence:** `template-parts/commerce/product-card.php: $is_top/$card_crop/$frame_uri`; `assets/css/theme.css: [data-card-frame="v2-statue"], __frame-label, __inscription`; `data/approved-card-fronts.json`.
- **Recovery constraint:** Recover approved frame/crop/inscription intent with readable larger garments and native actions outside obstruction; retain B derivative/loading/Quick View/global-product finally logic.

### 12. Shop staging and archive composition

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY**. State: `layout_replaced`.

- **A→C:** A/C use previous native archive with approved-frame cards.
- **C→B:** C→B 03947e829 introduces compact filtered archive and current grid; card change occurs separately.
- **Evidence:** `woocommerce/archive-product.php`; `woocommerce/content-product.php`; `assets/css/theme.css versus shop-page.css`.
- **Recovery constraint:** Recover visual staging while keeping validated native GET filters, sorting, pagination, counts and product order.

### 13. PDP portal exhibition, emblem and gallery depth

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY**. State: `visual_replaced_with_prior_commerce_repair`.

- **A→C:** A uses portal surface, contextual emblem/story, large gallery and spacious purchase column. A→C changes media resolver, missing frame size and exception cleanup, not wholesale visual template.
- **C→B:** C→B 67b2cd856 replaces portal with .sr2-pdp-product--editorial and scoped 7/5 CSS; moves native excerpt after purchase.
- **Evidence:** `woocommerce/single-product.php`; `template-parts/commerce/product-hero.php: .sr2-pdp-product--portal`; `assets/css/theme.css: .sr2-pdp-product--portal/__summary`.
- **Recovery constraint:** Restore portal depth around current native hooks/forms/schema/media guards; native detail/lightbox/variation caches and no-JS visibility stay current.

### 14. Signature architectural Golden Gate world

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY** · Motion: **RESTORE WITH REFACTOR**. State: `route_bypass`.

- **A→C:** A/C show full monument entry and multi-scene commerce chapters with Golden Gate/Oakland origin.
- **C→B:** C→B 67b2cd856 route returns shared world.php early, bypassing old hero/rail; content and scene files remain.
- **Evidence:** `functions.php: skyyrose2_collections signature`; `template-collection.php: hero/world rail`; `template-immersive-signature.php`.
- **Recovery constraint:** Recover monument scale, chapter environments and layer relationships from A; keep product authority and native scrolling.

### 15. Black Rose moonlit Bay Bridge, salon/court and product layers

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY** · Motion: **RESTORE WITH REFACTOR**. State: `route_bypass`.

- **A→C:** A/C show nocturnal hero, scene chapters and approved/placeholder-aware composites, then Town Line.
- **C→B:** C→B shared world.php nocturne uses contained hero, products/story and static directory; old scene renderer bypassed.
- **Evidence:** `template-collection.php`; `functions.php: black-rose world definition`; `template-parts/commerce/hero-composed-scene.php`; `inc/hero-commerce-scenes.php`.
- **Recovery constraint:** First-three recovery priority: preserve dramatic world staging, localized light and faithful layer relationships; keep existing scene status distinctions.

### 16. Love Hurts aisle, Beast/rose and fractured scene treatment

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY** · Motion: **RESTORE WITH REFACTOR**. State: `route_bypass`.

- **A→C:** A/C own aisle/Beast/rose world and scene-specific source contracts. A→C removes retired Cormorant fallback in hero-commerce-scenes.css.
- **C→B:** C→B world.php devotion replaces broad scene sequence with contained image and limited offset/rule variation.
- **Evidence:** `template-collection.php`; `functions.php: love-hurts definition`; `inc/hero-commerce-scenes.php`; `template-immersive-love-hurts.php`.
- **Recovery constraint:** Recover dramatic expressive spatial treatment without restoring retired font or weakening contrast/commerce.

### 17. Kids Heir throne and future-facing world

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY** · Motion: **RESTORE WITH REFACTOR**. State: `collection_route_bypass_immersive_retained`.

- **A→C:** A/C preserve Heir throne/media and dedicated immersive world; Home procession is separate.
- **C→B:** C→B collection uses inheritance variant and two-card edit; immersive template/source remain identical.
- **Evidence:** `template-collection.php`; `functions.php: kids-capsule definition`; `template-immersive-kids-capsule.php`; `template-parts/immersive/world.php`.
- **Recovery constraint:** Recover throne/product relationship and generational visual story; distinguish dedicated world from removed Home procession.

### 18. Four existing approved hero clips and localized overlay motion

**RESTORE + REFACTOR** · Motion: **RESTORE WITH REFACTOR**. State: `retained_but_route_unwired`.

- **A→C:** A manifest marks all four founder_approved and binds source/web hashes. Source manifest/controller unchanged A/C/B.
- **C→B:** C→B shared collection branch skips old hero video markup and its conditional script/style load.
- **Evidence:** `data/collection-hero-motion.json`; `functions.php: skyyrose2_collection_hero_motion`; `assets/js/collection-scene-motion.js`; `template-collection.php: data-scene-motion-toggle`.
- **Recovery constraint:** Reuse exact authorized delivery variants and source/hash guards, poster fallback, user control, reduced motion and Save-Data; do not call masters missing when derivatives are packaged.

### 19. Scene backplate, protected model, furniture/statue and hotspot composition

**RESTORE + REFACTOR** · Motion: **RESTORE WITH REFACTOR**. State: `partially_unwired_not_asset_loss`.

- **A→C:** A includes approved/review-candidate/placeholder branches, model_layers, suppress_model_layers_for_placeholder and hero_composed decisions. C preserves these with font correction.
- **C→B:** C→B leaves helpers/layer data intact but main collection bypasses them; dedicated immersive consumers persist.
- **Evidence:** `template-parts/commerce/hero-composed-scene.php`; `inc/hero-commerce-scenes.php`; `data/collection-scene-motion.json`; `assets/css/hero-commerce-scenes.css`.
- **Recovery constraint:** Restore each scene under its existing scene/editorial status; do not treat opening rejection as a universal scene ban or candidate state as commerce approval.

### 20. Four dedicated immersive routes, DOM chapters and Three.js enhancement

**KEEP CURRENT** · Motion: **RESTORE WITH REFACTOR**. State: `runtime_retained_discovery_reduced`.

- **A→C:** All four templates, shared wrapper and source CSS/JS identical A/C/B; minified artifacts rebuilt A→C.
- **C→B:** No source removal C→B. Collection page no longer displays explicit Enter full scene action; entry discovery changed.
- **Evidence:** `template-immersive-*.php`; `template-parts/immersive/world.php`; `assets/css/immersive.css`; `assets/js/immersive.js`.
- **Recovery constraint:** Keep actual wrapper/source and recover entry links after route check. Reprofile enhancement cost and keyboard; do not rebuild or label absent simply because collection page changed.

### 21. Town Line train film, scroll chapters and jersey reveal

**RESTORE + REFACTOR** · Motion: **RESTORE WITH REFACTOR**. State: `retained_renderer_and_media_unwired`.

- **A→C:** A/C render fictional train poster/film with controls, transcript, chapter links and optional card grid. Explicit data-media-status=founder-review-candidate.
- **C→B:** C→B Home and Black Rose use template-parts/commerce/town-line.php static directory; original function/film controller remain but enqueue excludes new collection routes.
- **Evidence:** `functions.php: skyyrose2_render_black_rose_jersey_series`; `assets/js/house-of-roses-motion.js: initFilm`; `assets/video/skyyrose-tour-around-the-bay.*`; `assets/css/legacy-world-components.css`.
- **Recovery constraint:** Recover filmed editorial composition and original truthful candidate context, not product approval. Keep current canonical series order and destinations; do not autoplay on constrained/mobile preferences.

### 22. World/rail/Town Line CSS moved out of global stylesheet

**RESTORE + REFACTOR**. State: `css_retained_conditionally_excluded`.

- **A→C:** A/C rules live in theme.css.
- **C→B:** C→B extracts legacy world/rail/jersey/film selectors; B does not enqueue for Home, editorial collections, PDP or Shop, while legacy consumers still load it.
- **Evidence:** `assets/css/legacy-world-components.css`; `functions.php: skyyrose2_assets $page_styles`.
- **Recovery constraint:** Extract restored systems into explicit route bundles preserving cascade; existence in a compatibility file is not active styling on target routes.

### 23. House/Home and Royal Procession CSS deleted after temporary extraction

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY**. State: `css_removed`.

- **A→C:** A/C have 39 sr-house-hero and 103 sr-kids-procession occurrences in source and served minified CSS.
- **C→B:** B contains zero of both selectors in source theme.css and no final legacy-home.css package.
- **Evidence:** `A/C assets/css/theme.css: .sr-house-hero, .sr-kids-procession`; `B assets/css/home-page.css`.
- **Recovery constraint:** Recover the exact A component rules from evidence and reconcile semantic tokens/cascade; do not assume compatibility CSS contains these removed systems.

### 24. Skyy character, guide/recall and mobile presence

**RESTORE + REFACTOR** · Motion: **RESTORE WITH REFACTOR**. State: `behavior_changed_in_phase2_not_removed_in3B`.

- **A→C:** A fixed proactive guide. A→C stops unsolicited mobile entrance, avoids prompts during overlays, repairs Escape/focus, makes mobile guide relative and hides during dialog/nav.
- **C→B:** Mascot CSS/JS unchanged C→B; B still enqueues off checkout. Shell integration later protects overlays.
- **Evidence:** `template-parts/skyy-mascot.php`; `assets/js/{mascot,mascot-loader,skyy-3d}.js`; `assets/css/mascot.css`; `functions.php: skyyrose2_assets`.
- **Recovery constraint:** Restore meaningful visual presence where capture warrants while keeping mobile non-obstruction, explicit user agency and overlay ownership. Do not claim mascot asset/controller deletion.

### 25. Monument/crop/spatial type and licensed font assets

**RESTORE VISUALLY / REIMPLEMENT TECHNICALLY**. State: `roles_and_composition_replaced`.

- **A→C:** A→C removes retired font fallback and adjusts token/contrast/layer behavior. Font files retained.
- **C→B:** C→B adds semantic role system and replaces large scene-specific type with common editorial roles; font files not removed.
- **Evidence:** `assets/css/design-tokens.css`; `theme.json`; `A theme.css portal/house/inscription selectors`.
- **Recovery constraint:** Recover scale, engraved inscription, offset and image/artwork relationships with current licensed fonts/contrast. Do not restore Cormorant or fake asset-backed scripts as text.

### 26. Opening approval records and PDP image safeguards

**KEEP CURRENT**. State: `engineering_improved_not_approval_removed`.

- **A→C:** Opening manifest identical A/C/B. A→C introduces commerce resolver and explicit rejected-state guard/schema/variation consistency.
- **C→B:** C→B adds native gallery/cache/AJAX wrapper and hash-bound responsive delivery.
- **Evidence:** `data/opening-product-media.json`; `functions.php: skyyrose2_product_commerce_media`; `inc/pdp-media-delivery.php`.
- **Recovery constraint:** Preserve current guards. Separate Home editorial portraits, approved card fronts and approved opening/PDP contexts; report individual authority gaps instead of blanket erasure.

### 27. Recovered approved card fronts, scene imagery and delivery variants

**RESTORE EXACTLY**. State: `assets_retained`.

- **A→C:** Recovered media/source bindings remain available; A/C/B manifests listed are identical.
- **C→B:** No recovered path removed from B source; 374/447 paths release-included.
- **Evidence:** `data/approved-card-fronts.json`; `data/collection-hero-motion.json`; `data/collection-scene-motion.json`; `recovered_file_inventory`.
- **Recovery constraint:** When rewiring, preserve exact authorized pixels and bindings; responsive derivatives may preserve same pixels under current hash contract. This classification does not mean copy files or approve candidates.

### 28. Authoring masters/QA assets excluded from release

**KEEP CURRENT**. State: `source_only_since_canonical_boundary`.

- **A→C:** 73 recovered paths source-only by C, including21 assets plus internal documents/scripts.
- **C→B:** C/B release classifications identical for all447 recovered paths; no C→B exclusion.
- **Evidence:** `tools/v2-source-certification/package-boundary.json at C and B`; `recovered_file_inventory`.
- **Recovery constraint:** Keep masters/QA out of runtime when existing declared delivery files suffice; inspect exact dependency if a restored renderer needs an excluded file. No blanket unexclude.

### 29. Legacy Black Rose preorder hotspot salon inside false branch

**DISCARD**. State: `already_disabled_in_A`.

- **A→C:** A already disables this legacy branch; old controller exists but this PHP path cannot render it. C keeps false branch.
- **C→B:** C→B retires interactive-scene controller; cannot count this as removal of active staging scene.
- **Evidence:** `page.php: if(false) at recovered lines27/65`; `assets/js/theme.js: [data-interactive-scene]`.
- **Recovery constraint:** Discard claim that enabling this is restoration of the recovered rendered experience. Keep evidence available; any new activation requires separate media/product review.

### 30. Depth cards, product reels, hero video and Bay map hooks

**UNKNOWN**. State: `retired_code_actual_A_usage_unknown`.

- **A→C:** Controller source exists but no matching A PHP markup found; injected database/plugin content not included in447filearchive.
- **C→B:** C→B removes these controllers.
- **Evidence:** `A/C assets/js/theme.js: [data-depth-card], setupProductReel/[data-product-reel], [data-hero-video], [data-bay-map]`.
- **Recovery constraint:** Do not automatically reactivate or call them staging losses; require A rendered DOM or database-backed evidence.

### 31. Bag/search/Quick View/account/checkout behavior

**KEEP CURRENT**. State: `repaired_engineering`.

- **A→C:** A bag is a direct cart link, search dialog present; C corrects search grouping, media/checkout truth and focus.
- **C→B:** C→B shell overlay manager and native bag add reliable state/focus/inert; card and PDP improvements retain Woo authority.
- **Evidence:** `inc/global-shell.php`; `assets/js/theme.js: overlays`; `template-parts/commerce/search-dialog.php`; `Woo native forms`.
- **Recovery constraint:** Recover surrounding visual art direction around these controllers; never restore fake reservation/payment messages, old unsafe search or broken variation wiring.

### 32. About, journal, pre-order and miscellaneous editorial surfaces

**KEEP CURRENT**. State: `shared_style_delta`.

- **A→C:** A uses dedicated active partials; old about/preorder drafts already false. C repairs commerce language.
- **C→B:** No wholesale replacement of active partials C→B, though shared type/card/shell and CSS extraction affect appearance.
- **Evidence:** `template-parts/v2-about.php`; `template-parts/journal-press-fallback.php`; `template-parts/v2-preorder.php`; `page.php`.
- **Recovery constraint:** Retain current behavior; compare any affected route after shared visual recovery. Do not mistake stale false drafts for active templates.

## Important non-losses and uncertain dependencies

1. The four dedicated immersive PHP templates, their shared wrapper, source immersive CSS/JS, collection-scene motion source and Kids procession PHP/JS survive. Restoring their discovery/loading is different from reconstructing nonexistent files.
2. The old Home, procession, navigation-reel and PDP-portal CSS is not all hidden in compatibility CSS. A/C `theme.css` contain 39 `sr-house-hero`, 103 `sr-kids-procession`, 52 `sr2-header-worlds` and 20 `sr2-pdp-product--portal` occurrences; B contains zero of those in source `theme.css`. Their A/C minified CSS contains the same counts. Final B has no `legacy-home.css`. Recover these component styles deliberately from A rather than merely enqueueing `legacy-world-components.css`.
3. The compatibility file retains actual world/rail/Town Line selectors such as `.sr2-worlds__rail`, `.sr2-world--commerce` and film/jersey presentation. B's `skyyrose2_assets` deliberately skips it on Home, editorial collections, PDP and Shop; legacy consumers still load it.
4. A `page.php` already wraps old preorder salon and about drafts in `if ( false )`. The removed `data-interactive-scene` controller therefore cannot by itself establish a rendered staging feature. No A PHP consumer was found for old depth-card, product-reel, hero-video or Bay-map hooks; database/plugin-provided markup remains UNKNOWN.
5. The four hero-motion manifest entries are marked `founder_approved` and source/web-hash bound. That does not prove every video was successfully loaded in A. Retain their exact guards and check declared derivative files. Town Line explicitly uses `founder-review-candidate` and a previsualization transcript: restoration of its existing editorial use must not silently change that status.
6. The accepted 33 card fronts, the on-model Home editorial portraits, composed scene layers and opening/PDP approval are distinct authorities. No broad approval/rejection inference is valid merely because two contexts share a SKU. The A card frame's real photo window/crop/inscription treatment is recoverable presentation; its raw attachment fallback/eager policy is not the engineering foundation to copy.

## Recovery constraints and first checkpoint

Recover **Home → Black Rose → PDP** first, using A as the composition reference and B's commerce foundation. Stop expansion for the founder's direct A/B/D comparison before other worlds are propagated. The document does not prescribe a new brand direction, additional fonts, replacement campaign media or new Phase 3C motion.

Restore the scene's valuable layers and relationships before optimizing that restored implementation. Then measure responsive source choice, conditional route bundles, video/poster loading, below-fold requests, native-scroll behavior, reduced motion, Save-Data, CPU/memory and image visibility before JS. Targets of roughly Home mobile 4.45–4.80s and PDP mobile 5.04s are comparison references, not permission to flatten the recovered composition again; final recovery cannot certify catastrophic regressions.

Protect B's product/variation identity, native cart validation and line items, fragments, checkout, account/search, standard-payment and preorder-truth wording, schema, media/cache/AJAX guards, focus/inert/scroll ownership and reproducible packaging. If A styling assumes unsafe structure, reproduce the appearance around those repaired behaviors. Never restore a whole `functions.php`, theme stylesheet or commerce template to obtain one effect.

Root's pending comparison set: Home, Shop, PDP and all four worlds at 390/1440 plus navigation/search/bag. Each material row above must be paired with actual screenshot/DOM/media evidence before being described as recovered. This author performed no browser, build, runtime edit, restoration, commit or deployment.
