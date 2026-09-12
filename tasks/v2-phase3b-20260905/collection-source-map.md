# Phase 3B collection source map — Signature first

Read-only preparation against the accepted Phase3A descendant and concurrent Phase3B checkout. Governing inputs: founder Phase3B attachment `d1afc6fe-6c6a-4ca6-ac93-6236ac9d609c/pasted-text.txt` and `tasks/v2-phase3b-20260905/design-contract.md`. No runtime, manifest, build, database, media, branch, or commit change was made by this census. All theme-relative paths below start at `wordpress-theme/skyyrose-flagship-2/`. Line references are discovery pointers and can move as the parent implements Shop/PDP.

## Recommended seam

**Replace the shared page composition in `template-collection.php`, starting with Signature, while preserving the collection definitions, route resolver, exact product-resolution helpers, and media-authority manifests.** A shared arrival → short story → canonical product edit → optional verified editorial image → next-world index is sufficient. Render natural vertical DOM first; omit the old pinned/horizontal attributes from the replacement. Render/review Signature before extending controlled variants to other collections.

Separate “current source is wired,” “manifest records founder authorization,” “asset hash matches,” and “this reviewer has seen pixels.” The asset inventory below verifies exact local bytes/dimensions; it does not grant new approval. Existing approved cards and editorial scene states remain independent contexts.

## Routing and rendering map

| Source pointer | Current behavior | Preservation / replacement boundary |
|---|---|---|
| `functions.php:584`, `skyyrose2_collections()` | One array supplies four worlds' names, copy, hero paths, lockups, artifact/atmosphere, lookbook, fallback world sequence and portal frames | Keep stable data keys consumed by header, cards, PDP, Home and preloads. A new composition variant may be keyed here or in a small presentation adapter; do not create four duplicate template stacks. |
| `functions.php:927`, `skyyrose2_collection_url()` | Sanitizes a supported slug and delegates to marketplace page resolver | Keep working real page URLs/fallbacks. Do not build `/collections/…` ad hoc or pass multi-segment strings to the wrong helper. |
| `functions.php:1589`, `skyyrose2_collection_template()` | `template_include` filter selects shared template for exact child-page URI `collections/{slug}` even without manual template assignment | Preserve parent/URI isolation. An unrelated page named “signature” must not inherit this renderer. |
| `template-collection.php:12–27` | Selects collection by page slug; defaults Signature when directly assigned to an unknown slug. Resolves hero, hero motion and layered commerce scenes | Explicitly distinguish canonical route from fallback/manual assignment in new route predicate. Do not silently show Signature with a non-Signature `data-collection`. |
| `template-collection.php:28–86` | Global header + main collection context + Woo notices; huge hero first, separate identity/three CTAs after hero, repeated introduction | Replace arrival as one compact indexed identity/image composition with immediate Shop access. Keep Woo notices, one H1, main/skip target and local collection tokens. |
| `template-collection.php:88–172` | Pinned/horizontal world stack, commerce scene products/candidate notices and optional composed scenes | Retire from new Signature composition; do not delete its source helpers/manifests or expand candidate approvals. Optional scene inclusion needs explicit state and exact cast. |
| `template-collection.php:174–188` | Native product edit and Black Rose-only Jersey Series subchapter | Consume the evolved canonical card and live query. Keep Jersey isolation; static Town Line composition may replace its current film treatment later without changing product membership. |
| `template-collection.php:190–205` | Repeated manifesto, atmosphere/logo, lookbook and more CTA | Reduce repetitive story. Use at most a supported, verified image/story spread after commerce; do not call arbitrary old lookbook imagery current product proof. |
| `template-collection.php:207–219` | Three other-world links via canonical URL helper and per-world data context | Preserve useful progression; one next-world emphasis plus full collection index can replace equal links. Use the existing tokens, no parallel palette. |

**Route gating risk:** current collection asset enqueue and performance-preload logic use `is_page_template('template-collection.php')`, whereas route selection also happens later through `template_include`. An automatically resolved collection child page can therefore render the template while its stored `_wp_page_template` remains default. Use a common exact-route predicate for rendering/enqueue/preload decisions, and test manually assigned and automatically resolved pages. The existing performance test only stubs explicit template membership; it does not prove inferred routes receive identical assets.

## Product and commerce authority

| Helper/source | Verified contract | Use in replacement |
|---|---|---|
| `functions.php:1072`, `skyyrose2_get_products()` | Woo published product query by category, then generated presentation-registry membership; queries full collection before applying limit so excluded Jersey products cannot starve the Black Rose core rail; pre-order uses registry flag | Preserve registry/category isolation and post-filter limit semantics. It returns Woo objects; native card handles visible/price/status/action. Do not replace with a static SKU-price array. |
| `functions.php:1150`, `skyyrose2_get_products_by_skus()` | Exact SKU lookup; visible/published product; actual category membership; matching presentation collection; exact returned SKU check | Use for fixed authored Signature featured edit. Missing products remain missing; do not substitute another SKU to fill a composition. |
| `functions.php:1129`, `skyyrose2_collection_scene_product()` | Cycles through first12 filtered products using chapter modulo | Suitable only for generic supporting editorial products. It does not preserve a locked scene cast and should not replace exact scene SKU bindings. |
| `functions.php:1194`, `skyyrose2_product_cards()` | Live query + featured fallback + current empty message + one canonical loop renderer | Reuse/new wrapper can add heading/index/loading context and concise empty state. First-card index must not make every below-fold edit high priority. |
| `functions.php:1461`, `skyyrose2_render_product_loop_card()`; `template-parts/commerce/product-card.php` | Canonical card entry, global Woo product scoping, native action and real product metadata | Consume root's current Phase3B card contract. Do not copy its markup into collection templates or undo concurrent Shop/card evolution. |
| `functions.php:185`, `skyyrose2_resolve_commerce_scene_products()` | Ordered exact SKU slots; missing slots explicit; per-product availability; aggregate ready/partial/unavailable is resolution completeness, not “all products in stock” | Preserve truthful slot semantics if any existing scene is reused. No fictional combined bundle or “complete look” purchase action. |
| `functions.php:222`, `skyyrose2_scene_product_action_label()` | Native purchasability/stock and pre-order presentation determine label; variable pre-order uses choose-options language | Keep transactional meaning. Phase3B does not authorize changing pre-order fulfillment/payment mechanics. |
| `data/product-presentation-registry.json`; `inc/approved-card-fronts.php`; opening/PDP media resolver in functions | Generated presentation registry,33 independently accepted card fronts, separate16stale/9missing-front/5rejected/3approved editorial/PDP truth states | No authority collapse. Collection product cards can use accepted card-context images; approved campaign/scene use requires its own source evidence. |

The helper’s full-collection query is acceptable for the current small catalog but not an unbounded replacement for Shop pagination. Preserve the repaired native Shop query for “View all” links. A collection edit can show a bounded representative set and route to the real filtered Shop URL established by root's current implementation.

## Layered scene authority — do not flatten

`skyyrose2_collection_commerce_scenes()` (`functions.php:131`) resolves in this exact order:

1. `data/scene-narrative-blueprints.json`: collection `commerce_scene_chapters`, exact scene IDs/SKU casts, plate paths/approval strings. Missing/malformed/unapproved chapter returns the legacy fallback worlds for the entire collection.
2. `skyyrose2_founder_scene_placeholders()` (`functions.php:78`): `data/founder-selected-theme-placeholders-v1.json` schema and **exact asset SHA256** checks. A founder-selected placeholder is not final product scene approval.
3. `skyyrose2_apply_hero_commerce_scene()` (`inc/hero-commerce-scenes.php:77`): `data/hero-commerce-scenes-c1.json`; requires local wiring, matching collection, path prefix/existence. Sets `LOCAL_WIRED_COMPOSITION_CANDIDATE_NEEDS_REVIEW`, can replace exact product bindings, and suppresses duplicated model overlays because models are baked into the composition. Local candidate wiring does not imply final visual approval.
4. `skyyrose2_apply_collection_scene_motion()` (`inc/hero-commerce-scenes.php:16`): `data/collection-scene-motion.json`; requires founder-approved-visual and local-wiring flags, matching collection and allowed existing local paths. Sets approved-local-motion state, changes still poster and removes candidate variants/review message. Exact current video hash verification is captured in the table below; unlike hero-motion helper, this runtime helper does not itself compare recorded video hashes.
5. `template-parts/commerce/hero-composed-scene.php`: shows responsive still/optional movie with full-frame composition and separately resolved live product links. Emits `candidate` versus `founder-approved-local-motion` state. The runtime state is not a new product-media or deployment approval.

Do not “simplify” by using the first `asset` found in a manifest or the legacy `world[]` images as approved garment proof. Do not remove candidate notices while retaining a candidate image. For the first static Signature pass, use the verified configured monument arrival and actual canonical product cards; only introduce a supported scene after its state and cast are reviewed in context.

## Hero and responsive source authority

`skyyrose2_collection_hero_motion()` (`functions.php:756`) independently reads `data/collection-hero-motion.json`, requires `ai_motion.status=founder_approved`, verifies exact original source-path correspondence to the configured1440w hero, checks source SHA256, then validates both MP4/WebM path prefixes, hashes and maximum web-asset bytes. This is stronger than filename-based reuse and should remain intact even if the new arrival intentionally uses the still only.

The configured still roots are distinct: Signature Golden Gate, Black Rose Bay Bridge, Love Hurts rose aisle/cathedral, Kids Heir throne. Eye inspection in this census confirms Signature640w shows the two gold monuments and bridge, and confirms a real **Love Hurts responsive mismatch**: configured mobile640w shows a single star/heart monument under an arch with the Golden Gate visible, while desktop1440w shows the cathedral aisle, script monument, star monument and central back-turned figure. These are different scenes, not differently sized crops.

A matching existing `love-hurts-rose-aisle-monuments-v3-640w.webp` is present. It is inventoried below as an **unbound existing derivative**, not silently swapped. The narrow fix is to bind the same verified cathedral scene across breakpoints (after parent confirms context), keeping the matching preload function aligned. No new image generation is needed.

The table's “hash match” for source masters refers to the existing hero-motion manifest. Responsive derivative hashes were computed from current bytes; the motion manifest does not separately certify each derivative's hash. Images are not classified as approved merely because the file exists.

| World / role | Theme-relative asset path | Dimensions / bytes | SHA256 / recorded-source match |
|---|---|---|---|
| signature / source master | `assets/sot/images/hero/signature-golden-gate-monuments-v2.webp` | 1672×941 / 149,386B | `f8bb7d2a1572f575e4f9e4e45edebf6115b4339708e64f95c3bb251813fc0fd8`; matches manifest |
| signature / hero | `assets/sot/images/hero/responsive/signature-golden-gate-monuments-v2-1440w.webp` | 1440×810 / 115,506B | `927bf1bf6ea9ea7d7e1baf22311045ab8b51551f9e643faeb88536e477809554` |
| signature / hero_tablet | `assets/sot/images/hero/responsive/signature-golden-gate-monuments-v2-1024w.webp` | 1024×576 / 70,526B | `08564ea7aced4b1911709186fe8c55891730810ca6588dc0b8b1f3a09c9b5419` |
| signature / hero_mobile | `assets/sot/images/hero/responsive/signature-golden-gate-monuments-v2-640w.webp` | 640×360 / 33,434B | `f3666eda781e94e06e27560bd5b57dc8bf24732d0c105a0241d49be8a9709d06` |
| black-rose / source master | `assets/sot/images/hero/black-rose-bay-bridge-monuments-v4.webp` | 1672×941 / 141,154B | `e5ebd5a0b28043137d2c5a2c99efe99cc63704c2157719793e95f6d7d0fc2ca9`; matches manifest |
| black-rose / hero | `assets/sot/images/hero/responsive/black-rose-bay-bridge-monuments-v4-1440w.webp` | 1440×810 / 108,926B | `ae49e7f146c900891467c511604639eb338ab79d10e210c8813f41acdb96b9bc` |
| black-rose / hero_tablet | `assets/sot/images/hero/responsive/black-rose-bay-bridge-monuments-v4-1024w.webp` | 1024×576 / 63,508B | `0f2b7e412dcfc56680cf442eef46829e0a5960485b153489fa8b448d6796cce0` |
| black-rose / hero_mobile | `assets/sot/images/hero/responsive/black-rose-bay-bridge-monuments-v4-640w.webp` | 640×360 / 28,500B | `5abe63100fba3f69773c7db9548d48ddf04cd18c9962419ca888801a0696eb08` |
| love-hurts / source master | `assets/sot/images/hero/love-hurts-rose-aisle-monuments-v3.webp` | 1672×941 / 242,636B | `000ec8308a8d72daa691ca8712ef751d7d8e9b9805a4b08060ebc9a59a2bc3a3`; matches manifest |
| love-hurts / hero | `assets/sot/images/hero/responsive/love-hurts-rose-aisle-monuments-v3-1440w.webp` | 1440×810 / 181,918B | `b9362ef7ea80740603abfa0b95e973bd3f0593c9d7af47c8f78392db4c5b2062` |
| love-hurts / hero_tablet | `assets/sot/images/hero/responsive/love-hurts-rose-aisle-monuments-v3-1024w.webp` | 1024×576 / 108,290B | `732cff974d835bbe66630fa3817b876222c9ac71133bc6e177570fa7e22d3875` |
| love-hurts / hero_mobile | `assets/sot/images/hero/responsive/love-hurts-golden-gate-monument-v2-640w.webp` | 640×360 / 26,472B | `1d3f83ae8d22a3cb1a47b487b41c0050d3944254fb72fdf996a6e6b499f73a0f` |
| kids-capsule / source master | `assets/sot/images/hero/kids-capsule-heir-throne-v3.webp` | 1672×941 / 142,022B | `fda7df7f2f0391e10c8e93121b0757242010d4c49e69d79beb8827a32d0b7936`; matches manifest |
| kids-capsule / hero | `assets/sot/images/hero/responsive/kids-capsule-heir-throne-v3-1440w.webp` | 1440×810 / 108,766B | `549a800235f9a50de6e61f1d58b69ca94f2ae7d2c26a09d0711a5a2b6e51ac4c` |
| kids-capsule / hero_tablet | `assets/sot/images/hero/responsive/kids-capsule-heir-throne-v3-1024w.webp` | 1024×576 / 66,610B | `af745ca34335422542cf5524d7d4348b2b5bdb236d802bda72c93007f68521f3` |
| kids-capsule / hero_mobile | `assets/sot/images/hero/responsive/kids-capsule-heir-throne-v3-640w.webp` | 640×360 / 33,402B | `0e3da0bcecffd44c073becd947c32df1cedd7dfdd278de1c6f656f36ae57452c` |
| love-hurts / unbound matching mobile derivative | `assets/sot/images/hero/responsive/love-hurts-rose-aisle-monuments-v3-640w.webp` | 640×360 / 46,998B | `063b7fe9ce19b1c1ca948ece5ff6919757187686c4ecdc71e9e022f06f755b44` |

### Signature supporting imagery currently declared

These are source references, not a fresh garment-fidelity or rights approval. Use only after checking editorial usage context.

| Role | Theme-relative path | Dimensions / bytes | SHA256 |
|---|---|---|---|
| lockup | `assets/sot/images/lockups/signature-lockup.webp` | 1600×540 / 148,060B | `12ffac7935de8f8ce0607d3f652abef6e28c7acbb38726dad460930b80b198e1` |
| artifact | `assets/sot/images/logos/sr-monogram-rose-gold.webp` | 720×720 / 31,900B | `a67a4efc3811828f6bcea872422a76d85fb9afd929266256c71fc4c0ce7b1d1d` |
| atmosphere | `assets/sot/images/logos/rose-gold-rose.webp` | 1544×2036 / 86,184B | `2f98fd671e121057b73caa8a1052a30d6afddf1cbe75cd11b25f356f70bfc1db` |
| lookbook | `assets/sot/images/lookbook/lb-rose-hoodie-beanie-960w.webp` | 960×1280 / 72,002B | `d8704e75d28007e44f2dd665f8db07cc6e409b834539c33a00488c3418608fcf` |
| lookbook_mobile | `assets/sot/images/lookbook/lb-rose-hoodie-beanie-480w.webp` | 480×640 / 36,350B | `a6de7bc27b01f461032a62c9168e7549348f0dd9f055710823b4ff716e23a17c` |

### Current exact scene bindings and final motion poster

This reflects manifest precedence, not a new runtime-browser evaluation. All nine motion records declare founder visual approval and local wiring. Kids has no commerce_scene_chapters and uses its legacy world array. Hashes below are actual still-poster bytes.

| Collection / scene | Effective exact SKU bindings | Effective poster under assets/scroll-world/ | Size / bytes / SHA256 |
|---|---|---|---|
| signature / SIG-COMMERCE-1 | sg-009, sg-007 | `generated-candidates/hero-commerce-c1/sig-commerce-1-natural-font-h1-1672w.webp` | 1671×941; 255,858B; `a42deb80c50e5caecc989b2ac43982e4c2aa8e6abb2b91c7c10be1d34bfc9026` |
| signature / SIG-COMMERCE-2 | sg-013, sg-014, sg-006 | `generated-candidates/hero-commerce-c1/sig-commerce-2-natural-dress-h1-1672w.webp` | 1672×941; 310,334B; `df2f79e55c5cff38dbac7efae3cd7258685cd546c4713fa413ec9b6f0640111e` |
| signature / SIG-COMMERCE-3 | sg-001, sg-005, sg-003, sg-002, sg-015 | `motion/collection-scenes-k1/sig-commerce-3-motion-k1-poster.webp` | 1671×942; 263,964B; `eef20813c482773ff3217cd1f367c61bbd9c31734e45ff252ea1f510783bc452` |
| black-rose / BR-COMMERCE-1 | br-001, br-002 | `motion/collection-scenes-k1/br-commerce-1-archive-4-poster.webp` | 960×1707; 133,642B; `b29c1c71c8754a2c716771dcf701ec880438f423e305f6fdb097d0765a8cdc51` |
| black-rose / BR-COMMERCE-2 | br-005, br-007, br-004 | `generated-candidates/hero-commerce-c1/br-commerce-2-hero-wall-statue-i3-1672w.webp` | 1672×941; 289,988B; `33d013d846261c82b14a3e015f630f3e8ef9e54a9f5e0592ce50e289fa1da338` |
| black-rose / BR-COMMERCE-3 | br-008, br-009, br-010, br-011, br-012 | `generated-candidates/founder-commerce-scenes-v1/black-rose/br-commerce-3-five-jersey-lounge-a-black-founder-approved-v1.png` | 1536×1024; 2,482,505B; `90aa66aa432bc9176b393dfbb7ce15d507860e47c970650337f78c4b1a480683` |
| love-hurts / LH-COMMERCE-1 | lh-004, lh-002, lh-006 | `motion/collection-scenes-k1/lh-commerce-1-archive-4-poster.webp` | 960×540; 113,872B; `6a5fbe9d202f0e47492ded17b4ea381311b41412fc3625c949874da1392d3446` |
| love-hurts / LH-COMMERCE-2 | lh-003 | `generated-candidates/hero-commerce-c1/lh-commerce-2-hero-composed-c1-1672w.webp` | 1672×941; 340,456B; `bf251eba7a2366a8d133e10a7bd548f3c6d89d2137da8263a418e761bfdefa54` |
| love-hurts / LH-COMMERCE-3 | lh-005 | `motion/collection-scenes-k1/lh-commerce-3-archive-4-poster.webp` | 960×1280; 87,158B; `ddff5431de7614583429a92d4c51ebab87dc0b174ff1292b997004aa7b8dfc5c` |

### Signature optional motion assets — source preservation only

Static-first replacement does not require these video transfers. These are already-authorized existing files; preserve their records and rollback availability.

| Context | Theme-relative video | Bytes | SHA256 matches declared record |
|---|---|---:|---|
| Signature hero | `assets/video/collection-heroes/approved/signature/web/signature-golden-gate-monuments-motion-v1.mp4` | 855,303 | True; `423323ae440372f72b4b0336116e49f42dafec6b39ff58bf08c5973ebcecc8e6` |
| Signature hero | `assets/video/collection-heroes/approved/signature/web/signature-golden-gate-monuments-motion-v1.webm` | 472,889 | True; `e0e57f545959a946b71ca2125d9dbc46756778510e55dc59ef65ac1e79127a02` |
| SIG-COMMERCE-1 | `assets/scroll-world/motion/collection-scenes-k1/sig-commerce-1-motion-k1-1080p.mp4` | 3,033,789 | True; `72538af2cca074b3d8d8f5252e20793390936e3844c345184682b2b15e8b9de7` |
| SIG-COMMERCE-1 | `assets/scroll-world/motion/collection-scenes-k1/sig-commerce-1-motion-k1-720p.mp4` | 1,222,252 | True; `aef4de2a77fc1c0e6be98398636e1b1f5cd6ed9b5e314eac60f87dd5cd2c3ef2` |
| SIG-COMMERCE-2 | `assets/scroll-world/motion/collection-scenes-k1/sig-commerce-2-motion-k1-1080p.mp4` | 3,529,477 | True; `da10d2d6f09594a8cdc1e0172878451a6a92c76242330a563e9ee22dab262eb7` |
| SIG-COMMERCE-2 | `assets/scroll-world/motion/collection-scenes-k1/sig-commerce-2-motion-k1-720p.mp4` | 1,548,209 | True; `f9e6b2fb536d75d14fcc477bbcf8eedc406ec9222988a8e8c22895477d8b72ba` |
| SIG-COMMERCE-3 | `assets/scroll-world/motion/collection-scenes-k1/sig-commerce-3-motion-k1-1080p.mp4` | 3,476,074 | True; `dcc1cea5b7606ee8ee66074d91da4829394ca5f76cb12d8524b6d4d73735bcbe` |
| SIG-COMMERCE-3 | `assets/scroll-world/motion/collection-scenes-k1/sig-commerce-3-motion-k1-720p.mp4` | 1,589,393 | True; `80cba25c744d4bf823dffa2f2761a7de6d3d466703e12f6c066b2f4f4011f981` |

### Authority record fingerprints

| Data file | SHA256 at census |
|---|---|
| `data/collection-hero-motion.json` | `c6d99877bff4f1d97303cd56cf96f91cd3492e2ca6e5c220ecba375e4b9437cf` |
| `data/collection-scene-motion.json` | `23f51f54fefc0d37e91ea3c76fecfbf0d2ba6553ce0c0c0c4d31d3e3c91892ef` |
| `data/scene-narrative-blueprints.json` | `1f4bfa4d778e1383a2d04d3a9406b3b251de9e06acfd341a5885bc95b0a0f240` |
| `data/hero-commerce-scenes-c1.json` | `cdfd57623707f60f06f911177c458848044bc5bdb7dd182a3d22b11cbc97dd9e` |
| `data/founder-selected-theme-placeholders-v1.json` | `7072cd6bb81783ba6890ddac521073667c12785801a77473fb99cf5652d234ff` |
| `data/hero-aspect-remakes-v3-manifest.json` | `f49dab7a430b075f867059f232398496b96292700b36a3b45169329b89ec35d4` |
| `data/approved-card-fronts.json` | `c9a35409c5af2fac958bc429c798f6ff23fee5a4e03db2e41dd604a1b96d841c` |
| `data/opening-product-media.json` | `6de9889f180edf3b5ba86ee308f710649e114facbcb5830bb7b89eac1332abae` |
| `data/product-presentation-registry.json` | `cf14a9dc84560658af94df2b2bdb5dec177b3ce8289f1364af684ec0efcacc12` |

## Asset and interaction dependency map

| Dependency | Current activation / responsibility | Narrow static-first treatment |
|---|---|---|
| Common `theme.js` overlay coordinator | Header/nav, bag, search and native dialogs across routes | Preserve unchanged. A collection filter or gallery must use this coordinator rather than creating another body lock/focus trap. |
| `theme.js:299–394`, `[data-scene-motion]` | Collection hero effects, lazy source attachment, pause/resume, visibility/reduced-motion/Save-Data; video error fallback | New static arrival can omit these attributes/video/effects. Keep controller for any still-existing consumer until all routes migrate. Do not autoplay old video merely because the helper resolves it. |
| `theme.js:395–539`, `setupPinnedWorld`; `:542–588`, horizontal rail | Converts desktop≥1200/fine-pointer world into sticky stage with synthetic document height and translated rail; compact/reduced fallback is still a horizontal scroller | Omit `data-horizontal-world`, `data-scroll-world-pinned`, `data-scroll-world-stage`, `data-horizontal-rail` from the new vertical collection modules. This retires behavior on the new page without changing unrelated legacy consumers. Native vertical chapter anchors replace next/previous rail controls. |
| `theme.js:740`, `[data-hero-depth]` | Pointer/scroll-dependent hero depth | Omit for the first static composition. If later reintroduced, no geometry initialization or content hidden before JS. |
| `assets/js/collection-scene-motion.js` | `[data-collection-scene-motion]` videos; selects mobile/slow-network source; one most-visible scene active; user pause, Save-Data/reduced-motion and tab lifecycle | Do not enqueue for a collection with no such videos. Preserve existing script/manifests for retained legacy or explicitly opted-in scene modules. Static image render must not depend on the script. |
| `assets/js/house-of-roses-motion.js` | Legacy header reel and Town Line `[data-house-film]`; own cinematic/scroll film logic | Accepted header no longer needs the old reel. Signature static collection needs neither. Black Rose may keep static Town Line story/products without invoking final advanced film motion. Do not delete globally until Home and any legacy consumer are checked. |
| `assets/css/theme.css` | `.sr2-worlds*`111–155, `.sr2-collection-hero*`194–254, identity/intro256+, manifesto/crossnav, scene/product portal styles and late responsive overrides | Move/rewrite collection composition into one conditional collection CSS source; remove replaced selectors only after checking dedicated immersive and content pages. Keep shared `.sr2-products`, controls, cards, tokens and global overlay rules. Preserve late responsive cascade ordering during extraction. |
| `assets/css/hero-commerce-scenes.css` | Full-frame composed scene presentation, separate product links | Needed only when rendering `hero-composed-scene.php`. Do not load in first static arrival/product edit if no composed scene is present. |
| `assets/css/collection-scene-motion.css` | Video layering, motion toggles | Needed only with the matching motion module. No static-first reason to force it into every collection. |
| `functions.php` enqueue branch around357 | Scene CSS/JS applies to shared collection and three dedicated immersive templates; Kids immersive omitted from this branch | Replace route test with truthful actual module/route presence. Keep other route dependencies stable; do not assume every immersive template shares the same module stack. |
| `template-collection.php:22–25` | Conditionally enqueues `skyyrose2-hero-commerce-scenes` directly with hardcoded `.min.css`/`1.0.0-c1`, duplicating the normal helper-owned handle | Retire this exceptional registration from the new composition; use the standard source/min selector and content hash version through the existing asset owner. Avoid “clean” changes that break retained legacy scenes. |
| `inc/performance.php:215–260` | Matching responsive hero preloads; native eager/decode handling | New collection arrival must declare one responsive media contract reused for preload and `<picture>`. Preserve source verification; no unconditional second hero preload. Ensure automatically resolved routes receive matching hints. |

Dedicated immersive entrypoints (`template-immersive-signature.php`, Black Rose/Love Hurts/Kids equivalents, `template-parts/immersive/world.php`) are **separate surfaces**. Replacing collection commerce is not authorization to add WebGL/Three.js or rebuild final immersive systems. Keep existing destinations valid, but the new collection should not require following an immersive CTA before reaching products.

## Signature-first implementation plan within the authorized sequence

1. Finish root's card/Shop/PDP components first. The collection consumes their real APIs, native URLs and loading arguments.
2. Resolve Signature only through its supported route. Keep the shared template dispatch, header/footer, `data-collection=signature`, one H1 and Woo notices. Add a focusable `main`/anchor target where needed for skip and chapter navigation.
3. Arrival: index/name/short source-backed line, a bounded Golden Gate monument picture and direct Shop Signature action in the first mobile viewport. Use actual 640/1024/1440 image dimensions/aspect from this inventory; do not use a full-screen decorative plane that pushes all identity below the fold.
4. Product edit: canonical cards from the published Signature query; a feature uses exact SKU lookup if an authored cast is selected. Reserve full readable garment framing; no scene-image product substitution. Show a real native filtered Shop destination for the full set.
5. Short origin spread: existing Signature manifesto/origin wording, optionally a reviewed supporting image. Avoid repeating the same manifesto in three sections or inventing material/craft facts.
6. Optional editorial chapter: use a specifically reviewed still from the resolved Signature scene inventory only if it adds meaning. Keep exact SKU bindings and independent purchasability links. The first implementation can omit all scene videos and provide a stronger static layout.
7. Next-world continuation: controlled house architecture rather than three equal giant previews. Preserve collection navigation and focus/contrast.
8. Render Signature at320/360/375/390/414/768/1024/1280/1440/1728 with reduced motion and delayed JS. Confirm native scroll/reverse/refresh-midpage and first-view media budget. Independent review before other worlds.
9. Extend one shared module grammar through bounded variants: Black Rose asymmetry/silver, Love Hurts a controlled editorial offset/crimson with responsive correction, Kids future-owner/rose. Commerce alignment remains stable. Homepage consumes this mature system last.

## Specific risks and evidence gates

- **Love Hurts world drift is visually confirmed**, not a filename hypothesis. Existing matching rose-aisle640w is46,998B; the currently bound GoldenGate variant is26,472B. Fixing the identity mismatch legitimately adds20,526B to that one mobile hero versus the incorrect scene. Offset with smaller critical CSS/lazy secondary media; do not keep wrong-world imagery solely to improve a score.
- **Approval-state ambiguity:** C1 records may state `founder_final_visual_approval=false` while a later motion record authorizes the clip/poster locally. Evaluate the effective layered record, not just one file. Retain exact authorization scope and candidate provenance. `hero-aspect-remakes-v3-manifest.json` explicitly marks four older statue remakes rejected/unwired; do not reuse those assets as attractive background alternatives.
- **Large optional video cost:** Signature's three scene clips total several MB even at720p (see table). Static-first composition can avoid all three downloads; no need to invent another motion framework. Existing hero loops are separately approved but still must meet new performance priorities.
- **Aspect correctness:** actual Signature scene posters include1671×941 and1671×942 despite nominal1672w names. Runtime motion helper reads actual image dimensions; a new static renderer should preserve accurate reserved aspect rather than copy filename-derived dimensions.
- **Missing collection products:** preserve returned empty/missing state and a real Shop/collection route. Current “Next pieces entering the world soon” suggests future availability without data; an honest neutral empty-state is a safe content refinement.
- **Unbounded first-screen text:** current collection shop copy mentions technical Woo authority and duplicates generic gender-neutral prose. Preserve factual meaning but do not expose implementation machinery to customers; editorial copy should support identity or purchase decisions.
- **Broad CSS deletion:** old world/rail/product class names occur outside the shared collection page, including dedicated immersive pages and Town Line. Remove only styles proven superseded for all consumers, or retain legacy route assets while new collection CSS is conditional.
- **Build reproducibility:** runtime PHP hashes, source/min/generated output records, build inputs and package boundary must be updated by root for actual authorized changes. Preserve media/registry hashes above; a new module does not justify repinning unrelated assets. Rebuild through pinned toolchain and repeat package parity checks.
- **Verification coverage:** existing `scripts/test-performance.php` checks explicit collection-template preload branches; add inferred-route coverage. Existing exact-SKU/registry/media tests remain required. Add meaningful tests for route isolation, one matching hero request, exact collection products, no eager below-fold competition, and unchanged commerce/account/search/checkout behavior—not snapshots that merely duplicate markup.

## Census outcome

One shared collection template and its data/commerce helper seams can support the requested architecture without new infrastructure. All four hero source masters match the existing motion-manifest hashes. Signature's two hero web-video files and six scene variants also match their declared hashes. The first Signature replacement should use the existing monument still, the evolved canonical card, live product queries, stable type geometry, and normal vertical reading. Broader world/media decisions remain with root and independent visual review; no new visual approval or implementation is claimed here.
