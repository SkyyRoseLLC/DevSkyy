# SkyyRose V2 experience architecture — OODA baseline

Baseline: aabd2bffdc1e322862acd05a5640c14cf00f3acf. Theme: wordpress-theme/skyyrose-flagship-2. The attached completion brief is implementation authority. This map precedes new feature edits.

## Retain

- `functions.php` bootstraps route-specific enqueues and the native WooCommerce integration; `inc/global-shell.php` owns the global header/concierge context.
- `assets/css/design-tokens.css` and `theme.json` are the existing runtime/editor token system. Extend these owners through the token generator; do not create a second token system.
- `assets/js/theme.js` owns overlays, rail navigation, and current card Quick View. Keep its modal focus, Escape, and native link fallbacks.
- `assets/js/collection-scene-motion.js`, `inc/hero-commerce-scenes.php`, and `data/approved-scroll-world-scenes.json` govern the exact nine approved scenes. Preserve all assets and approval bindings.
- `visual-recovery.js` retains approved animated heroes, decode-before-motion, reduced motion and Save-Data handling.
- `mascot-loader.js`, `mascot.js`, and `skyy-3d.js` provide deferred character, conversation and preserved model/animation runtime. No new character asset or rig work.
- Native Woo templates remain authoritative for product, variations, prices, stock, quantities, cart, checkout, account and notices. Product-card art and verified media remain unchanged.

## Observed gaps and decisions

1. Quick View currently copies display-only card facts and links to PDP. The brief now requires selection and purchase within Quick View. Reuse server-rendered native Woo purchase forms and native variation resolution; do not invent a JS cart or price model. Preserve link fallback and focus restoration.
2. Search dialog currently submits a native GET form. Add intent-driven, debounced previews derived from the existing native search route, with cancellation, bounded results, accessible count/loading/error and keyboard links. Keep ordinary form submission fully functional.
3. Mobile LCP remains above the prior release target. Establish matching baseline/candidate delivery before attributing changes. Preserve hero, card and character systems.
4. Existing static responsive runner omits 1024px and Firefox and runs reduced motion only. Add distinct coverage; do not relabel it as cinematic or full commerce evidence.
5. Motion has micro/interface/editorial/scene tokens; material and character categories need explicit mapping. Extend in a later controlled pass after census, without restyling founder assets.

## Component implementation contract for the first commerce pass

Keep current dialog composition, fonts, colors, card imagery and shared control styles. Use current interface timing, no added bounce/tilt/particles. Search uses standard result links and native keyboard order; announcements do not steal focus. Quick View retains product-image/copy hierarchy, native variation forms, real server actions and explicit loading/failure state. At small widths keep dialogs scrollable with visible exit, native 44px controls, and purchasing reachable. Disable optional transitions with reduced motion; request data only on user intent. Abort stale requests on closure and superseding input. Checkout remains quiet.

## Preservation and deferred boundaries

Town Line source and authoring videos remain preserved; no final Pre-Order invention. Deeper Blender refinement remains deferred. No deployment, promotion, remote writes or paid generation. Asset presence and historic test reports do not establish current certification.
