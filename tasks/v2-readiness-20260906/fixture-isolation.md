# Fixture theme-root isolation — V2 readiness

**Fixed in the four authorized local routers only.** The shared fixture symlink and database were not edited. No source-theme file or other worktree was modified. This is a fixture correction, not a performance optimization or creative change.

## Confirmed source leak

The initial HTTP Home response from each router contained `wp-fonts-local` referencing `assets/derived/fonts/archivo-normal-width.woff2`. That file is absent from the baseline/current overlays but is explicitly declared at line 194 of `/Users/theceo/.codex/worktrees/v2-completion-pr-20260906/DevSkyy/wordpress-theme/skyyrose-flagship-2/theme.json`. The shared fixture's `wp-content/themes/skyyrose-flagship-2` symlink currently resolves to that other worktree. Original HTML and router bytes were retained before mutation.

The actual core path in `/Users/theceo/.codex/worktrees/19db/DevSkyy/.artifacts/v2-phase3-20260905/wordpress` is:

1. `wp-includes/fonts/class-wp-font-face-resolver.php:28–36` reads `wp_get_global_settings()`; its variation branch also uses `WP_Theme_JSON_Resolver`.
2. `wp-includes/class-wp-theme-json-resolver.php:252–255` asks `wp_get_theme()->get_file_path('theme.json')` and reads that file.
3. `wp-includes/theme.php:117–133` implements `wp_get_theme()`, calling `get_raw_theme_root()`, rather than the router's filtered `get_theme_root()`.
4. `wp-includes/theme.php:697–702` returns literal `/themes` whenever there is at most one registered theme directory, even when the sole directory was the overlay. `wp_get_theme()` prepends `WP_CONTENT_DIR` and therefore resolves the shared symlink.
5. `wp-settings.php:571` registers `get_theme_root()`, so the existing filter can register only the overlay and still trigger that single-directory shortcut. Normal template paths can be correct while theme.json provenance is wrong.

This establishes a split source problem, not a missing approved font derivative. Creating the absent font or suppressing its 404 would hide the fixture fault.

## Minimal router correction

After loading the plugin API and before bootstrapping WordPress, each router now registers a `muplugins_loaded` callback at `PHP_INT_MIN` that registers both the existing fixture theme directory and the intended overlay directory. Two registered directories bypass Core's raw-root shortcut. Request-local `pre_option_stylesheet_root` and `pre_option_template_root` filters return the overlay root, so active-theme raw lookup resolves that exact registered root. Existing `theme_root` and `theme_root_uri` filters remain.

Core executes `muplugins_loaded` at wp-settings.php:548, before the normal registration at :571 and active theme setup at :705. `get_raw_theme_root()` reads the filtered active stylesheet/template root at theme.php:708–712. No option is written, no cache is flushed and no shared symlink is changed by this code. Other inactive themes remain discoverable; active-theme JSON and styles are isolated to the overlay.

The existing URL/asset-serving/gzip branches are unchanged. Router backups and before/after hashes are recorded in `audit/fixture-isolation/receipt.json`.

## Verification

All four routers pass `php -l`. For each port, HTTP Home HTML was saved after correction; extracted `wp-fonts-local` font URLs exactly match the four declarations in that port's intended theme.json, and every font URL returns HTTP 200 with recorded bytes/hash. The foreign `archivo-normal-width` reference is absent on all ports. Baseline CSS still has its original five registered face definitions; current design-token CSS now has the root implementer's four-face configuration. The four theme.json faces are a separate Core output and do not imply equal design-token declarations.

| Port | Router | Intended role | Font URL match | All font HTTP results |
|---|---|---|---|---|
| 18423 | `.artifacts/v2-readiness-20260906/baseline-router.php` | Immutable baseline source, uncompressed | Exact 4/4 | 200 |
| 18424 | `.artifacts/v2-readiness-20260906/baseline-gzip-router.php` | Immutable baseline source, gzip | Exact 4/4 | 200 |
| 18416 | `.artifacts/v2-cinematic-ooda-20260906/router.php` | Current source, uncompressed | Exact 4/4 | 200 |
| 18419 | `.artifacts/v2-cinematic-finalization-20260906/compressed-current-router.php` | Current source, gzip | Exact 4/4 | 200 |

The proof folder is `.artifacts/v2-readiness-20260906/audit/fixture-isolation/`: `before-PORT.html`, `after-PORT.html`, `router-PORT.before.php`, and `receipt.json` with expected/actual URLs, exact theme.json SHA256 and font fetch hashes.

## Measurement consequence

Prior results generated while the shared symlink leaked another candidate's theme.json must not be claimed as an isolated baseline/current comparison. The precise historical start of contamination was not established here. Repeat affected baseline/current measurements through the corrected routers with source hashes bound to each run. This correction produces no claim of LCP improvement; its purpose is trustworthy measurement.

The same contamination caveat applies to all prior readiness CSS ablation and font-use experiments; rerun acceptance checks on the isolated fixture before certifying gains or absence of regressions.
