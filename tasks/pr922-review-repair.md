# PR 922 source repair

Starting PR head: `0feda3df9fcdd4c825ccdd58e56fe8a57dcf249f`. Integrated main:
`fa5677079b4e537747ea8337359f430982ebdbec`.

- Transparency inspection checks every decoded/composited animation frame; an
  opaque later frame fails the asset. CLI failures retain diagnostic tracebacks
  and a nonzero exit status.
- Ten offline tests cover static transparent/partial/opaque alpha,
  all-transparent animated WebP, later opaque frames, first opaque frame,
  missing/malformed assets, empty arguments, and multiple requested files.
  Fixtures inspect the decoded frame alphas to prove they reproduce the intended
  cases.
- Standalone V2 build requirements now declare Pillow 12.3.0 and README includes
  its installation step. No paid/provider operation is involved.
- Original homepage styling is preserved. Hero typography uses the public
  `--font-ui` alias. Three hex spellings were shortened without color changes.
- Desktop/mobile before-and-after captures revealed that the new vertical hero
  label overlapped the mobile collection CTA. At the existing 720px breakpoint
  the label now runs horizontally in the bottom margin below the actions.
- Main's exact Sharp pin and lockfile are retained. The collection-test
  substring selector is valid; no unnecessary script change was introduced.

Validation: ten Python tests, Ruff/Black, original theme PHP lint, scoped CSS
Stylelint, original canonical CSS build/check (61 outputs), and V2 clean
install, build and verification all pass. Minified files are current;
core.min.css uses six minified bundle segments and is not a pretty-printed
source file.

Visual evidence lives outside shipped source under `.artifacts/pr922-review`:
`before-1440.png`, `after-1440.png`, `before-390.png`, `after-390.png`,
corresponding hero captures, and `comparison.json`. The fixture uses the actual
original storefront template and existing assets with offline WordPress
adapters. It preserves four collection links, both hero actions and their ~48px
hit areas, and has zero document overflow at both sizes. Screenshots were
inspected. This comparison does not claim live WooCommerce/payment behavior or
product media approval. No deployment, paid call, image generation, or promotion
occurred.

Independent Python review and CSS simplifier review approved the scoped repair
before commit/push.

The first normal commit hook correctly rejected generated-token freshness:
Prettier normalized the canonical producer's whitespace, hex spelling, and font
quotes. The producer now emits formatter-stable CSS. A regression runs the real
producer, Prettier, and producer again, asserting byte idempotence and parsed
CSS values. An independent comparison against the original PR head found all 195
parsed declarations semantically unchanged. Typography and identity source JSON
are unchanged. The focused producer/drift/transparency suite passes 16 tests;
the formatter regression, root TypeScript check, ESLint, Ruff and Black pass.

The root Prettier exclusions now cover only original-theme generated minified
CSS under assets/css and style.min.css, owned by its canonical build-css.js.
Prettier fileInfo confirms source design-tokens.css remains formatted while both
generated locations are ignored. Canonical minification still checks all 61
outputs. Read-only public homepage inspection returned HTTP 200; this is
availability evidence only, not deployment or commerce certification.
